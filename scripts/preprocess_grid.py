"""Preprocess the GRID audio-visual corpus for Visual-Forced-Alignment.

Input layout (raw GRID):
    raw_dir/<speaker>/<clip>.mpg
    raw_dir/<speaker>/align/<clip>.align

Output layout:
    out_dir/<speaker>/video/<clip>.mp4    # 96x96 mediapipe lip crop, 25 fps
    out_dir/<speaker>/align/<clip>.align  # copied through unchanged
    out_dir/{train,val,test}_split.json   # lists of video paths relative to out_dir

Usage:
    python scripts/preprocess_grid.py --raw_dir /path/to/grid_raw --out_dir ./GRID_processed

Optionally --speaker_independent to hold out whole speakers for val/test
(the standard GRID evaluation), instead of a random 80/10/10 clip split.
"""

import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress mediapipe C++ logging

import sys
# Ensure module resolution across repo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import random

import cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from data.preprocessing import extract_lip_frames


def save_video_from_frames(frames, output_path, fps=25):
    """Saves a list of numpy RGB frames to a video file."""
    if not frames:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        # Convert RGB back to BGR for OpenCV writing
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def make_splits(out_dir, speaker_independent=False):
    """Writes train/val/test split jsons (lists of paths relative to out_dir)."""
    all_videos = sorted(glob.glob(os.path.join(out_dir, '*', 'video', '*.mp4')))
    rel_videos = [os.path.relpath(v, out_dir) for v in all_videos]

    if speaker_independent:
        speakers = sorted({p.split(os.sep)[0] for p in rel_videos})
        random.seed(42)
        random.shuffle(speakers)
        test_speakers = set(speakers[:2])
        val_speakers = set(speakers[2:4])
        train_videos = [p for p in rel_videos if p.split(os.sep)[0] not in test_speakers | val_speakers]
        val_videos = [p for p in rel_videos if p.split(os.sep)[0] in val_speakers]
        test_videos = [p for p in rel_videos if p.split(os.sep)[0] in test_speakers]
    else:
        random.seed(42)
        random.shuffle(rel_videos)
        total_size = len(rel_videos)
        test_size = max(1, int(total_size * 0.1))
        val_size = max(1, int(total_size * 0.1))
        train_videos = rel_videos[:total_size - val_size - test_size]
        val_videos = rel_videos[total_size - val_size - test_size:total_size - test_size]
        test_videos = rel_videos[total_size - test_size:]

    for name, split in [("train", train_videos), ("val", val_videos), ("test", test_videos)]:
        with open(os.path.join(out_dir, f"{name}_split.json"), "w") as f:
            json.dump(split, f)

    print(f"Data split saved in {out_dir}:")
    print(f" - Train: {len(train_videos)}")
    print(f" - Val:   {len(val_videos)}")
    print(f" - Test:  {len(test_videos)}")


def preprocess(raw_dir, out_dir, mp_model_path, speaker_independent=False):
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return

    print("Initializing MediaPipe Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path=mp_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    speakers = sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and d != "align"
    )
    print(f"Found {len(speakers)} speakers in '{raw_dir}'.")

    for speaker in speakers:
        speaker_raw_dir = os.path.join(raw_dir, speaker)
        raw_align_dir = os.path.join(speaker_raw_dir, "align")
        videos = sorted(glob.glob(os.path.join(speaker_raw_dir, "*.mpg")))
        print(f"Processing speaker: {speaker} ({len(videos)} videos)")

        out_video_dir = os.path.join(out_dir, speaker, "video")
        out_align_dir = os.path.join(out_dir, speaker, "align")
        os.makedirs(out_video_dir, exist_ok=True)
        os.makedirs(out_align_dir, exist_ok=True)

        for vid_path in tqdm(videos, desc=f"Preprocessing {speaker}"):
            base_name = os.path.splitext(os.path.basename(vid_path))[0]
            src_aln_path = os.path.join(raw_align_dir, f"{base_name}.align")
            if not os.path.exists(src_aln_path):
                continue

            out_vid_path = os.path.join(out_video_dir, f"{base_name}.mp4")
            out_aln_path = os.path.join(out_align_dir, f"{base_name}.align")

            if not os.path.exists(out_aln_path):
                import shutil
                shutil.copy2(src_aln_path, out_aln_path)

            # Skip video if it's already completely extracted
            if os.path.exists(out_vid_path):
                continue

            # GRID is natively 25 fps, so frame indices are preserved by the
            # re-encode. Any frames dropped by face-tracking fallback are
            # absorbed at training time by rescaling the align times to the
            # actual frame count (see GRIDDataset).
            cropped_frames = extract_lip_frames(vid_path, detector=detector)
            if cropped_frames:
                save_video_from_frames(cropped_frames, out_vid_path, fps=25)

    make_splits(out_dir, speaker_independent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GRID corpus (mediapipe lip cropping).")
    parser.add_argument('--raw_dir', required=True, help='Raw GRID root (contains speaker dirs).')
    parser.add_argument('--out_dir', default='./GRID_processed', help='Output root for processed data.')
    parser.add_argument('--model_path', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'face_landmarker.task'),
        help='Path to mediapipe face_landmarker.task')
    parser.add_argument('--speaker_independent', action='store_true',
                        help='Hold out whole speakers for val/test instead of a random clip split.')
    args = parser.parse_args()

    preprocess(args.raw_dir, args.out_dir, args.model_path, args.speaker_independent)

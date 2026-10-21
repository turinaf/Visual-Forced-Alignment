"""Single-sample inference: align one preprocessed GRID clip to a transcript.

Given a lip-cropped video (96x96 mp4 from scripts/preprocess_grid.py) and a
word transcript, runs the trained model and prints per-word start/end times
via viterbi forced alignment.

Usage:
    python inference.py \
        --video /path/to/GRID_processed/s01/video/bbalg1n.mp4 \
        --transcript "bin blue at f two now" \
        --checkpoint /path/to/grid_v1.pkl \
        --grid_root /path/to/GRID_processed

    # Or derive the transcript (and ground-truth comparison) from a .align file:
    python inference.py --video ... --align /path/to/bbalg1n.align --checkpoint ...
"""

import argparse
from itertools import groupby

import cv2
import numpy as np
import torch

from data import consts
from data.grid_dataset import CROP_SIZE, SIL, build_vocab, get_grid_labels, parse_align_file
from models.phoneme_split_net import PhonemeSplitDecoderNet
from scripts.viterbi_align import viterbi_align
from utils import data_augment_util as transforms

FPS = 25


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('--video', required=True, help='path to preprocessed 96x96 lip mp4')
    parser.add_argument('--transcript', help='space-separated words, e.g. "bin blue at f two now"')
    parser.add_argument('--align', help='optional .align file; used for the transcript and '
                                       'ground-truth comparison (overrides --transcript)')
    parser.add_argument('--checkpoint', required=True, help='path to trained .pkl checkpoint')
    parser.add_argument('--grid_root', required=True,
                        help='preprocessed GRID root (rebuilds the vocab so token ids match training)')
    parser.add_argument('--use_bound_pred', action='store_true',
                        help='feed predicted boundary probs into the viterbi aligner')
    return parser.parse_args()


def read_video(video_path):
    """Same decode + center-crop as GRIDDataset in test mode: (1, T, 88, 88)."""
    cap = cv2.VideoCapture(video_path)
    img_tensor_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        img_tensor_list.append(torch.from_numpy(np.transpose(img, (2, 0, 1))).float())
    cap.release()
    if not img_tensor_list:
        raise RuntimeError(f"could not read any frames from {video_path}")
    img_sequence = torch.stack(img_tensor_list, 1)
    return transforms.CenterCropping(crop_size=CROP_SIZE)(img_sequence)


def transcript_to_ids(words, vocab):
    unknown = [w for w in words if w != "sil" and w not in vocab]
    if unknown:
        raise SystemExit(f"words not in GRID vocabulary: {unknown}")
    return [vocab[w] if w != "sil" else SIL for w in words]


def token_spans(frame_to_label):
    """Groups a frame-level label sequence into (token_id, start_frame, end_frame)."""
    spans = []
    start = 0
    for token_id, frames in groupby(frame_to_label):
        length = sum(1 for _ in frames)
        spans.append((token_id, start, start + length))
        start += length
    return spans


def main():
    args = arg_parse()
    if args.align is None and not args.transcript:
        raise SystemExit("provide either --transcript or --align")
    torch.cuda.set_device(int(args.gpu))

    vocab = build_vocab(args.grid_root)
    num_class = len(vocab) + 4  # PAD + SIL + words + BOS + EOS
    bos, eos = num_class - 2, num_class - 1
    id_to_word = {token_id: word for word, token_id in vocab.items()}
    id_to_word[SIL] = "sil"

    if args.align is not None:
        entries = parse_align_file(args.align)
        words = [w for w, _, _ in entries if w != "sil"]
    else:
        words = args.transcript.split()
    word_ids = transcript_to_ids(words, vocab)

    video = read_video(args.video)
    T = video.shape[1]
    if len(word_ids) > T:
        raise SystemExit(f"transcript has {len(word_ids)} words but video has only {T} frames")

    model = PhonemeSplitDecoderNet(num_class=num_class)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda().eval()

    video = video.unsqueeze(0).cuda()                      # (1, 1, T, 88, 88)
    transcript = torch.tensor([bos] + word_ids + [eos]).unsqueeze(0).cuda()
    frame_mask = torch.ones((1, T), dtype=torch.bool, device=video.device)

    with torch.inference_mode():
        frame_logits, _, boundary_probs = model.predict(
            video, transcript, label_mask=(transcript != 0), frame_mask=frame_mask)

    labels = [SIL] + word_ids + [SIL]  # leading/trailing silence, as in training
    bound_pred = boundary_probs[0].cpu().numpy() if args.use_bound_pred else None
    frame_to_label = viterbi_align(frame_logits[0].cpu(), labels, bound_pred=bound_pred)

    print(f"\n{args.video}  ({T} frames @ {FPS} fps)")
    print("word      start (s)   end (s)")
    for token_id, start, end in token_spans(frame_to_label):
        print(f"{id_to_word.get(token_id, f'<{token_id}>'):<10}{start / FPS:6.2f}      {end / FPS:6.2f}")

    if args.align is not None:
        gt_frame_label, _, _ = get_grid_labels(entries, vocab, T)
        gt_spans = [(id_to_word.get(t, f'<{t}>'), s) for t, s, _ in token_spans(gt_frame_label)]
        pred_spans = [(id_to_word.get(t, f'<{t}>'), s) for t, s, _ in token_spans(frame_to_label)]
        if len(gt_spans) == len(pred_spans):
            errs = [abs(p_start - g_start) for (_, p_start), (_, g_start) in zip(pred_spans, gt_spans)]
            acc = sum(1 for a, b in zip(frame_to_label, gt_frame_label) if a == b) / len(gt_frame_label)
            mean_err, median_err = np.mean(errs), np.median(errs)
            print(f"\nground-truth comparison: frame acc {acc:.4f}, "
                  f"boundary MAE mean {mean_err:.2f} frames ({mean_err * 1000 / FPS:.0f} ms), "
                  f"median {median_err:.1f} frames ({median_err * 1000 / FPS:.0f} ms)")
        else:
            print(f"\nground-truth comparison skipped: {len(pred_spans)} predicted spans "
                  f"vs {len(gt_spans)} ground-truth spans")


if __name__ == "__main__":
    main()

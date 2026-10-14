import json
import os

import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import data_augment_util as transforms

CROP_SIZE = [88, 88]
SIL_TOKEN = "sil"

PAD = 0
SIL = 1


def parse_align_file(path):
    """Parses a GRID .align file into a list of (word, start, end) tuples.

    Times are audio sample points at GRID's native 25 kHz sampling rate:
    value / 25000 = seconds, and since GRID video is 25 fps, exactly
    1000 samples correspond to one video frame (value / 1000 = frame).
    """
    entries = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            start, end, word = int(parts[0]), int(parts[1]), parts[2]
            entries.append((word, start, end))
    return entries


def build_vocab(root):
    """Builds a word->id mapping from every align file in all splits.

    ID layout: PAD=0, SIL=1, words=2..K+1 (sorted), BOS=K+2, EOS=K+3.
    """
    words = set()
    for split in ["train", "val", "test"]:
        split_file = os.path.join(root, f"{split}_split.json")
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            for rel in json.load(f):
                entries = parse_align_file(_align_path(rel, root))
                words.update(w for w, _, _ in entries if w != SIL_TOKEN)
    sorted_words = sorted(words)
    vocab = {w: i + 2 for i, w in enumerate(sorted_words)}
    return vocab


def _align_path(rel_video_path, root):
    """video/xxx.mp4 -> align/xxx.align (paths are relative to the data root)."""
    rel = rel_video_path.replace("video", "align").replace(".mp4", ".align")
    return os.path.join(root, rel)


def get_grid_labels(entries, vocab, frame_len):
    """Converts align entries to (frame_label, phoneme_label, phoneme_label_with_sil).

    GRID timing is 25 kHz audio samples with 25 fps video, so the exact
    conversion is frame = t // 1000 (HTK convention: a label owns frames in
    [floor(start/1000), floor(end/1000))). GRID aligns typically end ~0.02s
    before the video does (74.5 vs 75 frames); extra tail frames are just
    silence, so only rescale when the video is *shorter* than the align
    timeline implies — i.e. frames were dropped during preprocessing.
    """
    total_units = entries[-1][2]
    units_per_frame = 1000.0
    if frame_len < total_units / units_per_frame - 1:
        units_per_frame = total_units / frame_len

    def to_frame(t):
        return int(t // units_per_frame)

    frame_label = [SIL] * frame_len
    phoneme_label = []            # word sequence, no silences
    phoneme_label_with_sil = []   # SIL inserted wherever speech is absent
    pre_end = 0
    for word, start, end in entries:
        if word == SIL_TOKEN:
            continue
        start = to_frame(start)
        end = min(to_frame(end), frame_len)
        if start >= frame_len:
            break
        phone_id = vocab[word]

        phoneme_label.append(phone_id)
        if start > pre_end:
            phoneme_label_with_sil.append(SIL)
        pre_end = end
        phoneme_label_with_sil.append(phone_id)

        for i in range(start, end):
            frame_label[i] = phone_id

    # GRID aligns mark silence explicitly as sil entries, which are skipped
    # above, so also close off any trailing silence after the last word.
    if pre_end < frame_len:
        phoneme_label_with_sil.append(SIL)

    return frame_label, phoneme_label, phoneme_label_with_sil


class GRIDDataset(Dataset):
    """GRID audio-visual dataset over the output of scripts/preprocess_grid.py.

    Returns the same tuple as LRS2Dataset:
        (img_sequence, frame_label, phoneme_label, phoneme_label_with_sil)
    """

    def __init__(self, mode, root="./GRID_processed/"):
        super().__init__()
        assert mode in ["train", "val", "test"]
        self.root = root
        self.mode = mode

        with open(os.path.join(root, f"{mode}_split.json")) as f:
            self.file_list = json.load(f)

        # Vocabulary is built over ALL splits so token ids are consistent.
        self.vocab = build_vocab(root)
        # PAD + SIL + words + BOS + EOS
        self.num_class = len(self.vocab) + 4
        self.bos = self.num_class - 2
        self.eos = self.num_class - 1

        self.train_transform = transforms.Compose([
            transforms.RandomCropping(crop_size=CROP_SIZE),
            transforms.RandomFlipping(p=0.5),
            transforms.TimeMasking(video_fps=25),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCropping(crop_size=CROP_SIZE)
        ])

    def __len__(self):
        return len(self.file_list)

    def _read_video(self, video_path):
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
        img_sequence = torch.stack(img_tensor_list, 1)  # (channels, frames, H, W)
        return img_sequence, len(img_tensor_list)

    def __getitem__(self, index):
        rel = self.file_list[index]
        video_path = os.path.join(self.root, rel)
        align_path = _align_path(rel, self.root)

        img_sequence, frame_len = self._read_video(video_path)
        if self.mode in ["train"]:
            img_sequence = self.train_transform(img_sequence)
        elif self.mode in ["test", "val"]:
            img_sequence = self.test_transform(img_sequence)

        entries = parse_align_file(align_path)
        frame_label, phoneme_label, phoneme_label_with_sil = get_grid_labels(
            entries, self.vocab, frame_len)

        frame_label = torch.tensor(frame_label)
        phoneme_label = torch.tensor([self.bos] + phoneme_label + [self.eos])
        phoneme_label_with_sil = torch.tensor([self.bos] + phoneme_label_with_sil + [self.eos])
        return img_sequence, frame_label, phoneme_label, phoneme_label_with_sil


def collate_fn(batch):
    img_batch, frame_label_batch, phoneme_label_batch, w_sil_batch = [], [], [], []
    for img_sequence, frame_label, phoneme_label, phoneme_label_with_sil in batch:
        img_batch.append(img_sequence.permute(1, 0, 2, 3))
        frame_label_batch.append(frame_label)
        phoneme_label_batch.append(phoneme_label)
        w_sil_batch.append(phoneme_label_with_sil)

    img_sequence = pad_sequence(img_batch, batch_first=True, padding_value=0)
    img_sequence = img_sequence.permute(0, 2, 1, 3, 4)
    frame_label = pad_sequence(frame_label_batch, batch_first=True, padding_value=0)
    phoneme_label = pad_sequence(phoneme_label_batch, batch_first=True, padding_value=0)
    phoneme_label_with_sil = pad_sequence(w_sil_batch, batch_first=True, padding_value=0)
    return img_sequence, frame_label, phoneme_label, phoneme_label_with_sil

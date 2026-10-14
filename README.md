# Visual-Forced-Alignment

Visual forced alignment: aligning phoneme-level transcripts to the frames of a talking-face video using only **visual (lip-reading) information** — no audio required.

Given a mouth-region video and its phoneme transcript, the model learns to produce a **frame-level phoneme alignment** (which phoneme is being articulated in each video frame), analogous to what traditional audio-based forced alignment (e.g. HTK, MFA) does with speech, but driven entirely by the visual stream.

This is useful for building phoneme-level supervision for lip-reading / talking-head generation, and for aligning data where audio is unavailable, noisy, or unreliable.

## How it works

The model (`models/phoneme_split_net.py`, `PhonemeSplitDecoderNet`) is trained jointly with three objectives:

1. **Frame-level phoneme classification** — a per-frame cross-entropy loss predicts the phoneme spoken in each frame (the core alignment signal).
2. **Phoneme sequence prediction** — a Transformer decoder autoregressively predicts the phoneme sequence (with silences) from the visual features, ensuring the visual representation actually encodes the full transcript.
3. **Boundary prediction** — a per-frame binary loss predicts phoneme-transition frames.

At inference time, `scripts/viterbi_align.py` runs a **Viterbi-style dynamic-programming alignment** over the frame-level posteriors, constrained to the known phoneme sequence and biased by the predicted boundary probabilities, to output the final frame-to-phoneme alignment.

### Architecture

```
mouth ROIs (B, 1, T, H, W)
   │
   ▼
Visual front-end: 3D Conv (spatiotemporal) + ResNet-18 backbone  →  per-frame features (B, T, 512)
   │
   ▼
Positional encoding
   │
   ▼
Conformer encoder with cross-attention to phoneme embeddings (4 layers)
   │
   ├──► frame_linear: per-frame phoneme logits      (loss 1)
   ├──► Transformer decoder (autoregressive phoneme-seq prediction)  (loss 2)
   └──► bound_linear: per-frame boundary logit      (loss 3)
```

- **Visual front-end** (`models/resnet_new.py`): 3D convolution + ResNet-18 over the grayscale mouth ROI, producing one 512-d embedding per frame at 25 fps.
- **Conformer cross-attention encoder** (`models/conformer_xattn.py`): 4-layer Conformer (from [lucidrains/conformer](https://github.com/lucidrains/conformer)) where video frames attend to the phoneme label sequence.
- **Phoneme decoder**: standard `nn.TransformerDecoder` (2 layers) with greedy/beam decoding (`utils/sequence_decoder.py`).

The label space is 51 phonemes + padding + SIL + BOS/EOS (`data/consts.py`), i.e. 55 classes. Phoneme IDs in the alignment files are offset by `index_start=2` to make room for `PAD=0` and `SIL=1`.

## Data

Training uses the **LRS2** dataset with the following layout (paths configured in `data/LRS2_dataset.py`):

```
LRS2/
├── mouth_grey_96/main/<clip_id>/    # grayscale 96x96 mouth ROI frames (one image per video frame, 25 fps)
├── wav_align/main/<clip_id>.txt     # phoneme alignment files with timing, see format below
├── audio/main/                      # (referenced as label_root; alignment .txt used as labels)
└── data_split/{train,test}.txt      # clip ID lists per split
```

Each alignment file contains phoneme entries of the form:

```
<phoneme_id> <start_frame> <end_frame>
...
video_frames: <num_frames>
```

Data augmentation (`utils/data_augment_util.py`): random spatial cropping to 88×88, random horizontal flip, and temporal masking; center-cropping at test time.

## Usage

### Requirements

- Python 3 with **PyTorch** (CUDA), **torchvision**, **torchaudio**, **opencv-python**, **numpy**, **einops**, **editdistance**.

### Training

```bash
python train_phoneme_split.py \
    --gpu 0 \
    --batch_size 4 \
    --epoch_num 60 \
    --save_epochs 3 \
    --save_name my_experiment
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `2` | GPU id to use |
| `--batch_size` | `4` | Batch size |
| `--epoch_num` | `60` | Number of training epochs |
| `--save_epochs` | `3` | Interval (epochs) for test evaluation / logging |
| `--load_name` | `None` | Checkpoint name to resume from |
| `--save_name` | `test` | Experiment name; checkpoints saved to `./checkpoints/VFA/<save_name>/` |

Training uses the **Noam** learning-rate schedule (8 warm-up epochs) plus `ReduceLROnPlateau`, and logs frame accuracy, phoneme CER, and boundary accuracy to `./checkpoints/VFA/<save_name>/`. Every `--save_epochs` epochs it evaluates on the test split (frame CER/accuracy, phoneme CER, boundary accuracy).

### Alignment (inference)

After training, use the frame logits and boundary predictions with the Viterbi aligner:

```python
from scripts.viterbi_align import viterbi_align

# frame_logits: (T, num_class) per-frame phoneme logits from the model
# labels:       the known phoneme id sequence to align against (without BOS/EOS)
# bound_pred:   (T,) per-frame boundary probabilities (from bound_linear + sigmoid)
frame_to_label = viterbi_align(frame_logits, labels, bound_pred)
# -> list of length T mapping each frame index to a phoneme id
```

The DP enforces that the label sequence is consumed in order (each label occupies ≥1 frame) and forces a label transition when the boundary head fires with probability > 0.8.

## Project structure

```
├── train_phoneme_split.py       # training entry point
├── scripts/viterbi_align.py     # Viterbi frame-to-phoneme alignment
├── models/
│   ├── phoneme_split_net.py     # main model (front-end + Conformer + decoders)
│   ├── conformer_xattn.py       # Conformer encoder with cross-attention
│   └── resnet_new.py            # ResNet-18 visual backbone
├── data/
│   ├── consts.py                # constants (token ids, embed size, paths)
│   └── LRS2_dataset.py          # LRS2 dataset + label loading + collate
└── utils/
    ├── sequence_decoder.py      # greedy autoregressive decoding
    ├── data_augment_util.py     # video augmentations
    ├── optimizer_util.py        # Noam LR schedule wrapper
    └── logging_util.py          # logging helpers
```

## Notes & caveats

- The training script currently assumes a single CUDA GPU (`torch.cuda.set_device`) and fixed dataset paths — edit `data/LRS2_dataset.py` to point at your own LRS2 extraction.
- The phoneme ID space (51 phonemes) and label format follow the alignment files produced for LRS2; adapting to another corpus means changing the label files and `num_class` in `train_phoneme_split.py`.
- A known quirk in `PhonemeSplitDecoderNet.predict`: `ph_preds` returned by beam search is decoded per sample but the test loop assumes tensor shape — see `get_test_cer` in `train_phoneme_split.py` if you hit shape errors at evaluation time.

## License

MIT — see [LICENSE](LICENSE). © 2024 Yi He.

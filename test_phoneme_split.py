import argparse
from itertools import groupby

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import consts
from models.phoneme_split_net import PhonemeSplitDecoderNet
from scripts.viterbi_align import viterbi_align
from train_phoneme_split import calc_cer, dataset_factory, get_boundary_label

FPS = 25


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('--checkpoint', required=True, help='path to trained .pkl checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', choices=['LRS2', 'GRID'], default='GRID')
    parser.add_argument('--grid_root', default='./GRID_processed/')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_align', type=int, default=200,
                        help='number of test samples to force-align and score')
    parser.add_argument('--use_bound_pred', action='store_true',
                        help='feed predicted boundary probs into the viterbi aligner')
    parser.add_argument('--print_samples', type=int, default=3,
                        help='print word timing tables for the first N aligned samples')
    return parser.parse_args()


def token_spans(frame_to_label):
    """Groups a frame-level label sequence into (token_id, start_frame, end_frame)."""
    spans = []
    start = 0
    for token_id, frames in groupby(frame_to_label):
        length = sum(1 for _ in frames)
        spans.append((token_id, start, start + length))
        start += length
    return spans


@torch.no_grad()
def evaluate(model, dataloader, dataset, num_align, use_bound_pred, print_samples):
    cer_list, acc_list, ph_cer_list, bound_acc_list = [], [], [], []
    align_acc_list, boundary_err_list = [], []
    aligned = 0
    printed = 0
    model.eval()

    id_to_word = {token_id: word for word, token_id in dataset.vocab.items()}
    id_to_word[1] = "sil"

    for img_sequence, frame_label, ph_label, ph_label_w_sil in dataloader:
        img_sequence = img_sequence.cuda()
        frame_label = frame_label.cuda()
        ph_label = ph_label.cuda()
        ph_label_w_sil = ph_label_w_sil.cuda()
        mask = (ph_label != 0)
        frame_mask = (frame_label != 0)

        frame_logits, ph_preds, bound_preds = model.predict(
            img_sequence, ph_label, label_mask=mask, frame_mask=frame_mask)

        batch = frame_label.shape[0]
        boundary_label = get_boundary_label(frame_label).numpy()
        preds = frame_logits.argmax(-1).cpu().numpy()
        frame_label = frame_label.cpu().numpy()
        ph_label_w_sil = ph_label_w_sil.cpu().numpy()

        for i in range(batch):
            # --- decoding metrics (same as training-time eval) ---
            ph_len = np.sum(ph_label_w_sil[i] != 0)
            ph_truth = ph_label_w_sil[i][:ph_len][1:-1].tolist()
            ph_pred = ph_preds[i][1:-1].cpu().numpy().tolist()
            ph_cer_list.append(calc_cer(ph_pred, ph_truth))

            frame_len = np.sum(frame_label[i] != 0)
            bound_acc = np.sum(
                boundary_label[i][:frame_len]
                == (bound_preds[i][:frame_len] > 0.5).cpu().numpy()) / frame_len
            bound_acc_list.append(bound_acc)

            prediction = preds[i][:frame_len]
            truth = frame_label[i][:frame_len].tolist()
            acc_list.append(sum(1 for a, b in zip(prediction, truth) if a == b) / len(truth))
            cer_list.append(calc_cer(prediction, truth))

            # --- forced alignment metrics (first num_align samples only,
            # viterbi is per-sample and slow) ---
            if aligned >= num_align:
                continue
            labels = ph_label_w_sil[i][:ph_len][1:-1].tolist()
            if not (0 < len(labels) <= frame_len):
                continue
            bound_pred = bound_preds[i][:frame_len].cpu().numpy() if use_bound_pred else None
            frame_to_label = viterbi_align(
                frame_logits[i][:frame_len].cpu(), labels, bound_pred=bound_pred)

            # viterbi leaves frames at PAD(0) if the path degenerates
            if any(f == 0 for f in frame_to_label):
                continue
            aligned += 1

            gt_frames = truth
            align_acc_list.append(
                sum(1 for a, b in zip(frame_to_label, gt_frames) if a == b) / len(gt_frames))

            pred_spans = token_spans(frame_to_label)
            gt_spans = token_spans(gt_frames)
            if len(pred_spans) == len(gt_spans):
                errs = [abs(p_start - g_start)
                        for (_, p_start, _), (_, g_start, _) in zip(pred_spans, gt_spans)]
                boundary_err_list.extend(errs)

            if printed < print_samples:
                printed += 1
                print(f"\nSample {aligned}: {len(gt_frames)} frames @ 25 fps")
                for token_id, start, end in pred_spans:
                    word = id_to_word.get(token_id, f"<{token_id}>")
                    print(f"  {word:<8} {start / 25:6.2f}s - {end / 25:6.2f}s")

    print("\n===== test results =====")
    print(f"samples: {len(acc_list)}, force-aligned: {aligned}")
    print(f"frame acc: {np.mean(acc_list):.4f}, frame cer: {np.mean(cer_list):.4f}")
    print(f"phoneme cer with silence: {np.mean(ph_cer_list):.4f}")
    print(f"boundary predict acc: {np.mean(bound_acc_list):.4f}")
    if align_acc_list:
        print(f"alignment frame acc: {np.mean(align_acc_list):.4f}")
    if boundary_err_list:
        mean_err, median_err = np.mean(boundary_err_list), np.median(boundary_err_list)
        print(f"boundary MAE: mean {mean_err:.2f} frames ({mean_err * 1000 / FPS:.0f} ms), "
              f"median {median_err:.1f} frames ({median_err * 1000 / FPS:.0f} ms)")


def main():
    args = arg_parse()
    torch.cuda.set_device(int(args.gpu))

    _, test_dataset, collate_fn = dataset_factory(args.dataset, args.grid_root)
    model = PhonemeSplitDecoderNet(num_class=test_dataset.num_class)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda()
    print(f"loaded checkpoint: {args.checkpoint} "
          f"({len(test_dataset)} test samples, {test_dataset.num_class} classes)")

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)
    evaluate(model, dataloader, test_dataset, args.num_align,
             args.use_bound_pred, args.print_samples)


if __name__ == "__main__":
    main()

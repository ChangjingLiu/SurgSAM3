#!/usr/bin/env python3
"""Evaluate Medical-SAM3 2D checkpoint with per-class IoU and mIoU."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from sam3.model.model_misc import SAM3Output
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_miou_sam3_lora import (
    MIoUCOCODataset,
    finalize_segment_stats,
    masks_for_query,
    new_segment_stats,
    print_miou_report,
    segment_key_from_filename,
    update_segment_stats,
)
from infer_medsam3 import BPE_PATH, MEDICAL_SAM3_CHECKPOINT_2D, load_medsam3_checkpoint
from validate_sam3_lora import move_to_device


def evaluate_miou_medsam3(
    data_dir: str,
    checkpoint_path: str | None = None,
    num_samples: int | None = None,
    prob_threshold: float = 0.3,
    nms_iou: float = 0.7,
    miou_mode: str = "global",
    skip_no_gt: bool = True,
    json_out: str | None = None,
    by_segment: bool = True,
):
    os.chdir(Path(__file__).resolve().parent)

    checkpoint_path = checkpoint_path or MEDICAL_SAM3_CHECKPOINT_2D
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Medical-SAM3 2D checkpoint: {checkpoint_path}")

    print("Building SAM3 model (empty weights)...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path=None,
        load_from_HF=False,
        bpe_path=BPE_PATH,
        eval_mode=False,
    )
    print("Loading Medical-SAM3 checkpoint...")
    load_medsam3_checkpoint(model, checkpoint_path)

    model.to(device)
    model.eval()

    dataset = MIoUCOCODataset(Path(data_dir))
    n_images = len(dataset) if num_samples is None else min(num_samples, len(dataset))

    overall_stats = new_segment_stats()
    segment_stats: dict[str, dict] = defaultdict(new_segment_stats)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn_api(b, dict_key="input", with_seg_masks=True),
        num_workers=0,
    )

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(loader, total=n_images, desc="mIoU eval")):
            if batch_idx >= n_images:
                break

            datapoint = dataset[batch_idx]
            img_id = dataset.image_ids[batch_idx]
            file_name = dataset.images[img_id]["file_name"]
            segment_key = segment_key_from_filename(file_name)
            gt_masks = dataset.build_gt_class_masks(img_id)
            orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

            input_batch = move_to_device(batch_dict["input"], device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_outputs = list(outputs_iter)[-1][-1]

            pred_class_masks: dict[int, object] = {}
            for q_idx, query in enumerate(datapoint.find_queries):
                cat_id = int(query.inference_metadata.original_category_id)
                pred_class_masks[cat_id] = masks_for_query(
                    final_outputs["pred_logits"][q_idx].detach().cpu(),
                    final_outputs["pred_masks"][q_idx].detach().cpu(),
                    final_outputs["pred_boxes"][q_idx].detach().cpu(),
                    orig_h,
                    orig_w,
                    prob_threshold,
                    nms_iou,
                )

            update_segment_stats(
                overall_stats,
                dataset.category_ids,
                gt_masks,
                pred_class_masks,
                orig_h,
                orig_w,
                miou_mode,
            )
            if by_segment:
                update_segment_stats(
                    segment_stats[segment_key],
                    dataset.category_ids,
                    gt_masks,
                    pred_class_masks,
                    orig_h,
                    orig_w,
                    miou_mode,
                )

    overall_report = finalize_segment_stats(
        overall_stats,
        dataset.category_ids,
        dataset.categories,
        miou_mode,
        skip_no_gt,
    )
    results = {
        "checkpoint": checkpoint_path,
        "data_dir": str(data_dir),
        "miou_mode": miou_mode,
        "by_segment": by_segment,
        "overall": overall_report,
    }

    print_miou_report("Medical-SAM3 2D mIoU EVALUATION (overall)", overall_report)

    if by_segment and segment_stats:
        by_segment_reports = {}
        for segment_key in sorted(segment_stats):
            report = finalize_segment_stats(
                segment_stats[segment_key],
                dataset.category_ids,
                dataset.categories,
                miou_mode,
                skip_no_gt,
            )
            by_segment_reports[segment_key] = report
            print_miou_report(
                f"Medical-SAM3 2D mIoU EVALUATION — segment {segment_key}",
                report,
            )
        results["by_segment"] = by_segment_reports

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Medical-SAM3 2D mIoU evaluation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MEDICAL_SAM3_CHECKPOINT_2D,
        help=f"Path to Medical-SAM3 2D checkpoint (default: {MEDICAL_SAM3_CHECKPOINT_2D})",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--miou-mode", choices=("global", "per_image"), default="global")
    parser.add_argument("--include-no-gt-classes", action="store_true")
    parser.add_argument(
        "--no-by-segment",
        action="store_true",
        help="Disable per-clip/seq/video breakdown",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    evaluate_miou_medsam3(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        miou_mode=args.miou_mode,
        skip_no_gt=not args.include_no_gt_classes,
        json_out=args.json_out,
        by_segment=not args.no_by_segment,
    )


if __name__ == "__main__":
    main()

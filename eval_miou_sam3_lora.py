#!/usr/bin/env python3
"""Evaluate SAM3 LoRA with per-class IoU and mean IoU (mIoU) for semantic segmentation.

Checkpoint / LoRA modes:
  - Base SAM3 only: omit --weights (optional --config and/or --checkpoint)
  - SAM3 + LoRA: --config and --weights (base from config unless --checkpoint overrides)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pycocotools.mask as mask_utils
from sam3.model.model_misc import SAM3Output
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)
from torchvision.transforms import v2

from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters, load_lora_weights
from train_sam3_lora_native import resolve_sam3_checkpoint_path
from validate_sam3_lora import apply_sam3_nms, move_to_device

DEFAULT_SAM3_CHECKPOINT = "/mnt/data2_hdd/changjing/modelscope/facebook/sam3/sam3.pt"


def resolve_eval_checkpoint(
    config: dict | None,
    checkpoint_override: str | None = None,
) -> str:
    """CLI --checkpoint > config model.checkpoint_path > default sam3.pt."""
    if checkpoint_override is not None:
        path = str(Path(checkpoint_override).expanduser())
    elif config is not None:
        path = resolve_sam3_checkpoint_path(config)
    else:
        path = DEFAULT_SAM3_CHECKPOINT
    if not Path(path).exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {path}")
    return path


class MIoUCOCODataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        ann_file = self.data_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = sorted(self.images.keys())
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        self.category_ids = sorted(self.categories.keys())

        self.img_to_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"Loaded mIoU dataset from {self.data_dir}")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Categories ({len(self.category_ids)}): {self.categories}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _decode_ann_mask(self, ann: dict, height: int, width: int) -> np.ndarray:
        segmentation = ann.get("segmentation")
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)

        if isinstance(segmentation, dict):
            mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
        else:
            return np.zeros((height, width), dtype=np.uint8)

        if mask.shape[:2] != (height, width):
            mask = (
                np.array(PILImage.fromarray(mask.astype(np.uint8)).resize((width, height), PILImage.NEAREST)) > 0
            ).astype(np.uint8)
        return (mask > 0).astype(np.uint8)

    def build_gt_class_masks(self, img_id: int) -> dict[int, np.ndarray]:
        img_info = self.images[img_id]
        height, width = int(img_info["height"]), int(img_info["width"])
        class_masks: dict[int, np.ndarray] = {}
        for ann in self.img_to_anns.get(img_id, []):
            cat_id = int(ann["category_id"])
            mask = self._decode_ann_mask(ann, height, width)
            if cat_id not in class_masks:
                class_masks[cat_id] = mask.astype(bool)
            else:
                class_masks[cat_id] |= mask.astype(bool)
        return class_masks

    def __getitem__(self, idx: int) -> Datapoint:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.data_dir / img_info["file_name"]
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        pil_resized = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(pil_resized)

        anns = self.img_to_anns.get(img_id, [])
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        objects: list[Object] = []
        class_to_object_ids: dict[int, list[int]] = defaultdict(list)

        for obj_idx, ann in enumerate(anns):
            bbox_coco = ann.get("bbox")
            if not bbox_coco:
                continue
            cat_id = int(ann["category_id"])
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h
            box_tensor /= self.resolution

            segment = None
            segmentation = ann.get("segmentation")
            if segmentation:
                try:
                    mask_np = self._decode_ann_mask(ann, orig_h, orig_w)
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t, size=(self.resolution, self.resolution), mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5
                except Exception:
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] - box_tensor[0]) * (box_tensor[3] - box_tensor[1]),
                object_id=obj_idx,
                segment=segment,
            )
            objects.append(obj)
            class_to_object_ids[cat_id].append(obj_idx)

        image_obj = Image(data=image_tensor, objects=objects, size=(self.resolution, self.resolution))

        queries: list[FindQueryLoaded] = []
        for cat_id in self.category_ids:
            class_name = self.categories[cat_id]
            queries.append(
                FindQueryLoaded(
                    query_text=class_name.lower(),
                    image_id=0,
                    object_ids_output=class_to_object_ids.get(cat_id, []),
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=cat_id,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
            )

        return Datapoint(find_queries=queries, images=[image_obj], raw_images=[pil_image])


def masks_for_query(pred_logits, pred_masks, pred_boxes, orig_h, orig_w, prob_threshold, nms_iou):
    if len(pred_logits) == 0:
        return np.zeros((orig_h, orig_w), dtype=bool)

    filtered_masks, _, _ = apply_sam3_nms(
        pred_logits=pred_logits,
        pred_masks=pred_masks,
        pred_boxes=pred_boxes,
        prob_threshold=prob_threshold,
        nms_iou_threshold=nms_iou,
        max_detections=100,
    )
    if len(filtered_masks) == 0:
        return np.zeros((orig_h, orig_w), dtype=bool)

    masks_up = torch.nn.functional.interpolate(
        filtered_masks.unsqueeze(1).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return (masks_up > 0.5).any(dim=0).cpu().numpy().astype(bool)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return float("nan")
    return float(2 * inter / denom)


def segment_key_from_filename(file_name: str) -> str:
    """Map COCO file_name to a SurgTPGS-style segment id (e.g. 01_00080, seq_5)."""
    parts = file_name.split("__")
    if len(parts) >= 2 and parts[0].startswith("video") and parts[1].startswith("video"):
        video_num = parts[0].replace("video", "")
        clip_suffix = parts[1].split("_", 1)[-1]
        return f"{int(video_num):02d}_{clip_suffix}"

    for part in parts:
        if re.fullmatch(r"seq_\d+", part):
            return part

    for part in parts:
        m = re.fullmatch(r"[Vv]ideo(\d+)", part)
        if m:
            return f"Video{int(m.group(1)):02d}"

    return "unknown"


def new_segment_stats() -> dict:
    return {
        "global_inter": defaultdict(int),
        "global_union": defaultdict(int),
        "global_gt_pixels": defaultdict(int),
        "global_pred_pixels": defaultdict(int),
        "per_class_ious_per_image": defaultdict(list),
        "per_class_dice_per_image": defaultdict(list),
        "num_images": 0,
    }


def update_segment_stats(
    stats: dict,
    cat_ids: list[int],
    gt_masks: dict[int, np.ndarray],
    pred_class_masks: dict[int, np.ndarray],
    orig_h: int,
    orig_w: int,
    miou_mode: str,
) -> None:
    stats["num_images"] += 1
    for cat_id in cat_ids:
        gt = gt_masks.get(cat_id, np.zeros((orig_h, orig_w), dtype=bool))
        pred = pred_class_masks.get(cat_id, np.zeros((orig_h, orig_w), dtype=bool))
        if miou_mode == "global":
            stats["global_inter"][cat_id] += int(np.logical_and(pred, gt).sum())
            stats["global_union"][cat_id] += int(np.logical_or(pred, gt).sum())
            stats["global_gt_pixels"][cat_id] += int(gt.sum())
            stats["global_pred_pixels"][cat_id] += int(pred.sum())
        else:
            iou = compute_iou(pred, gt)
            if not np.isnan(iou):
                stats["per_class_ious_per_image"][cat_id].append(iou)
            dice = compute_dice(pred, gt)
            if not np.isnan(dice):
                stats["per_class_dice_per_image"][cat_id].append(dice)


def finalize_segment_stats(
    stats: dict,
    category_ids: list[int],
    categories: dict[int, str],
    miou_mode: str,
    skip_no_gt: bool,
) -> dict:
    per_class_iou: dict[int, float] = {}
    per_class_dice: dict[int, float] = {}
    for cat_id in category_ids:
        if miou_mode == "global":
            if stats["global_gt_pixels"][cat_id] == 0:
                if not skip_no_gt:
                    per_class_iou[cat_id] = 0.0
                    per_class_dice[cat_id] = 0.0
            else:
                if stats["global_union"][cat_id] > 0:
                    per_class_iou[cat_id] = (
                        stats["global_inter"][cat_id] / stats["global_union"][cat_id]
                    )
                dice_denom = (
                    stats["global_pred_pixels"][cat_id] + stats["global_gt_pixels"][cat_id]
                )
                if dice_denom > 0:
                    per_class_dice[cat_id] = (
                        2 * stats["global_inter"][cat_id] / dice_denom
                    )
        else:
            iou_values = stats["per_class_ious_per_image"].get(cat_id, [])
            if iou_values:
                per_class_iou[cat_id] = float(np.mean(iou_values))
            elif not skip_no_gt:
                per_class_iou[cat_id] = 0.0

            dice_values = stats["per_class_dice_per_image"].get(cat_id, [])
            if dice_values:
                per_class_dice[cat_id] = float(np.mean(dice_values))
            elif not skip_no_gt:
                per_class_dice[cat_id] = 0.0

    miou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0
    mdice = float(np.mean(list(per_class_dice.values()))) if per_class_dice else 0.0
    return {
        "num_images": stats["num_images"],
        "per_class_iou": {
            categories[cid]: round(per_class_iou[cid] * 100.0, 2)
            for cid in sorted(per_class_iou)
        },
        "miou_percent": round(miou * 100.0, 2),
        "per_class_dice": {
            categories[cid]: round(per_class_dice[cid] * 100.0, 2)
            for cid in sorted(per_class_dice)
        },
        "mdice_percent": round(mdice * 100.0, 2),
    }


def print_miou_report(title: str, report: dict) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Images: {report.get('num_images', '—')}")
    print("Per-class IoU (%):")
    for name, iou_pct in sorted(report["per_class_iou"].items(), key=lambda x: -x[1]):
        print(f"  {name:30s} {iou_pct:6.2f}")
    print("-" * 80)
    print(f"mIoU (%): {report['miou_percent']:.2f}")
    if "per_class_dice" in report:
        print("Per-class Dice (%):")
        for name, dice_pct in sorted(report["per_class_dice"].items(), key=lambda x: -x[1]):
            print(f"  {name:30s} {dice_pct:6.2f}")
        print("-" * 80)
        print(f"mDice (%): {report['mdice_percent']:.2f}")
    print("=" * 80)


def evaluate_miou(
    config_path, weights_path, data_dir, num_samples=None,
    prob_threshold=0.3, nms_iou=0.7, miou_mode="global",
    skip_no_gt: bool = True, json_out=None,
    by_segment: bool = True, checkpoint_override=None,
):
    # Relative paths (sam3/assets, configs) assume repo root as cwd.
    os.chdir(Path(__file__).resolve().parent)

    config = None
    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    use_lora = weights_path is not None
    if use_lora and config is None:
        raise ValueError("--config is required when --weights is set (LoRA structure)")

    checkpoint_path = resolve_eval_checkpoint(config, checkpoint_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Base checkpoint: {checkpoint_path}")
    if config_path:
        print(f"Config: {config_path}")

    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )

    if use_lora:
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)
        print(f"Loading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)
    else:
        print("Evaluating base SAM3 (no LoRA)")

    model.to(device)
    model.eval()

    dataset = MIoUCOCODataset(Path(data_dir))
    n_images = len(dataset) if num_samples is None else min(num_samples, len(dataset))

    overall_stats = new_segment_stats()
    segment_stats: dict[str, dict] = defaultdict(new_segment_stats)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
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

            with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
                final_outputs = list(outputs_iter)[-1][-1]

            pred_class_masks: dict[int, np.ndarray] = {}
            for q_idx, query in enumerate(datapoint.find_queries):
                cat_id = int(query.inference_metadata.original_category_id)
                pred_class_masks[cat_id] = masks_for_query(
                    final_outputs["pred_logits"][q_idx].detach().cpu(),
                    final_outputs["pred_masks"][q_idx].detach().cpu(),
                    final_outputs["pred_boxes"][q_idx].detach().cpu(),
                    orig_h, orig_w, prob_threshold, nms_iou,
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
        "data_dir": str(data_dir),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "lora_weights": weights_path,
        "miou_mode": miou_mode,
        "by_segment": by_segment,
        "overall": overall_report,
    }

    print_miou_report("mIoU EVALUATION (overall)", overall_report)

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
            print_miou_report(f"mIoU EVALUATION — segment {segment_key}", report)
        results["by_segment"] = by_segment_reports

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 mIoU evaluation (base SAM3 or SAM3 + LoRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Base SAM3 (default sam3.pt)\n"
            "  python eval_miou_sam3_lora.py --data_dir ./test\n\n"
            "  # Base SAM3 with explicit checkpoint\n"
            "  python eval_miou_sam3_lora.py --checkpoint /path/to/checkpoint_3D.pt --data_dir ./test\n\n"
            "  # Base from config model.checkpoint_path\n"
            "  python eval_miou_sam3_lora.py --config configs/foo.yaml --data_dir ./test\n\n"
            "  # SAM3 + LoRA\n"
            "  python eval_miou_sam3_lora.py --config configs/foo.yaml "
            "--weights outputs/best_lora_weights.pt --data_dir ./test"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config YAML (required with --weights; optional for base-only eval)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="LoRA weights .pt; omit to evaluate base SAM3 only",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Override base SAM3 checkpoint path "
            f"(default: config model.checkpoint_path or {DEFAULT_SAM3_CHECKPOINT})"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--miou-mode", choices=("global", "per_image"), default="global")
    parser.add_argument("--include-no-gt-classes", action="store_true")
    parser.add_argument(
        "--no-by-segment",
        action="store_true",
        help="Disable per-clip/seq/video breakdown (Cholec: 01_00080; EndoVis: seq_5)",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if args.weights and not args.config:
        parser.error("--config is required when --weights is set")
    if args.weights and not os.path.exists(args.weights):
        parser.error(f"LoRA weights not found: {args.weights}")

    evaluate_miou(
        config_path=args.config,
        weights_path=args.weights,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        miou_mode=args.miou_mode,
        skip_no_gt=not args.include_no_gt_classes,
        json_out=args.json_out,
        by_segment=not args.no_by_segment,
        checkpoint_override=args.checkpoint,
    )


if __name__ == "__main__":
    main()

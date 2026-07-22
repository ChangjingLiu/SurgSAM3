#!/usr/bin/env python3
"""
Medical-SAM3 2D inference.

Loads checkpoint_2D.pt (image-only format without detector. prefix) using the
same two-step pattern as Medical-SAM3 official inference/sam3_inference.py:
build empty model, then load custom checkpoint.

Usage:
    python infer_medsam3.py \
        --image path/to/image.png \
        --prompt "instrument-wrist" "kidney-parenchyma" \
        --output outputs/infer_medical_sam3_2d.png

    python infer_medsam3.py \
        --checkpoint /path/to/checkpoint_2D.pt \
        --image path/to/image.png \
        --prompt "Grasper" \
        --threshold 0.3 --nms-iou 0.7 \
        --output outputs/infer_grasper_2d.png
"""

from __future__ import annotations

import argparse
import os

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from infer_sam import SAM3LoRAInference

MEDICAL_SAM3_CHECKPOINT_2D = (
    "/mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_2D.pt"
)
BPE_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def load_medsam3_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load Medical-SAM3 checkpoint (detector-prefixed or image-only format)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    sample_key = next(iter(state_dict), "")
    if "detector." in sample_key:
        clean_state_dict = {
            k.replace("detector.", ""): v
            for k, v in state_dict.items()
            if "detector" in k
        }
        fmt = "detector-prefixed"
    else:
        clean_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("tracker.")
        }
        fmt = "image-only"

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    print(f"   Checkpoint format: {fmt}")
    print(f"   Loaded tensors: {len(clean_state_dict)}")
    if missing_keys:
        print(f"   Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"   Unexpected keys: {len(unexpected_keys)}")


class MedicalSAM3Inference(SAM3LoRAInference):
    """Medical-SAM3 2D inference (no LoRA)."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.use_base_model = True
        self.checkpoint_path = checkpoint_path or MEDICAL_SAM3_CHECKPOINT_2D
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("🔧 Initializing Medical-SAM3 (2D)...")
        print(f"   Device: {self.device}")
        print(f"   Checkpoint: {self.checkpoint_path}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        print("\n📦 Building SAM3 model (empty weights)...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path=None,
            load_from_HF=False,
            bpe_path=BPE_PATH,
            eval_mode=True,
        )

        print("💾 Loading Medical-SAM3 checkpoint...")
        load_medsam3_checkpoint(self.model, self.checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.use_manual_postprocess = True
        print("✅ Medical-SAM3 ready for inference!\n")


def main():
    parser = argparse.ArgumentParser(description="Medical-SAM3 2D inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MEDICAL_SAM3_CHECKPOINT_2D,
        help=f"Path to Medical-SAM3 2D checkpoint (default: {MEDICAL_SAM3_CHECKPOINT_2D})",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["object"],
        help='Text prompt(s), e.g. "Grasper" "Fat"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_medical_sam3_2d.png",
        help="Output visualization path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)",
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Show bounding boxes (default: False)",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5)",
    )
    args = parser.parse_args()

    inferencer = MedicalSAM3Inference(
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
    )

    results = inferencer.predict(args.image, args.prompt)
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks,
    )

    print("\n" + "=" * 60)
    print("📊 Summary:")
    for idx in sorted(k for k in results.keys() if k != "_image"):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result["num_detections"] > 0 and result["scores"] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Usage:
    # Text prompt inference
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Multiple prompts
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" "defect" "damage" \
        --output output.png

    # Original SAM3 (no LoRA)
    python3 infer_sam.py \
        --use-base-model \
        --image path/to/image.jpg \
        --prompt "Liver" "Gallbladder" \
        --output output_base.png

    # Medical-SAM3 base checkpoint
    python3 infer_sam.py \
        --use-base-model \
        --checkpoint /path/to/checkpoint_3D.pt \
        --image path/to/image.jpg \
        --prompt "instrument-wrist" \
        --output output_medical.png
"""

DEFAULT_SAM3_CHECKPOINT = "/mnt/data2_hdd/changjing/modelscope/facebook/sam3/sam3.pt"
MEDICAL_SAM3_CHECKPOINT_3D = (
    "/mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt"
)

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

MASK_ALPHA = 0.8

# Class name (lowercase) -> RGBA for mask overlay
CLASS_COLOR_RGBA = {
    # CholecSeg8K
    "abdominal wall": [230 / 255, 147 / 255, 166 / 255, MASK_ALPHA],       # #E693A6
    "grasper": [212 / 255, 242 / 255, 124 / 255, MASK_ALPHA],              # #D4F27C
    "l-hook electrocautery": [179 / 255, 226 / 255, 211 / 255, MASK_ALPHA],  # #B3E2D3
    "fat": [240 / 255, 210 / 255, 107 / 255, MASK_ALPHA],                  # #F0D26B
    "liver": [195 / 255, 95 / 255, 95 / 255, MASK_ALPHA],                    # #C35F5F
    "gallbladder": [95 / 255, 145 / 255, 210 / 255, MASK_ALPHA],             # #5F91D2
    "gastrointestinal tract": [130 / 255, 185 / 255, 125 / 255, MASK_ALPHA],  # #82B97D
    # EndoVis 2018 (softened from official label palette)
    "instrument-shaft": [110 / 255, 215 / 255, 110 / 255, MASK_ALPHA],       # #6ED76E
    "instrument-clasper": [95 / 255, 210 / 255, 220 / 255, MASK_ALPHA],      # #5FD2DC
    "instrument-wrist": [185 / 255, 240 / 255, 56 / 255, MASK_ALPHA],        # #B9F038
    "kidney-parenchyma": [245 / 255, 130 / 255, 85 / 255, MASK_ALPHA],       # #F58255
    "covered-kidney": [85 / 255, 115 / 255, 190 / 255, MASK_ALPHA],          # #5573BE
    "thread": [210 / 255, 180 / 255, 90 / 255, MASK_ALPHA],                  # #D2B45A
    "clamps": [95 / 255, 215 / 255, 165 / 255, MASK_ALPHA],                  # #5FD7A5
    "suturing-needle": [235 / 255, 230 / 255, 140 / 255, MASK_ALPHA],        # #EBE68C
    "suction-instrument": [165 / 255, 95 / 255, 195 / 255, MASK_ALPHA],      # #A55FC3
    "small-intestine": [175 / 255, 195 / 255, 75 / 255, MASK_ALPHA],         # #AFC34B
    "ultrasound-probe": [85 / 255, 215 / 255, 175 / 255, MASK_ALPHA],        # #55D7AF
}

FALLBACK_EDGE_COLORS = ["red", "blue", "green", "yellow", "cyan", "magenta"]
FALLBACK_RGBA = {
    "red": [1, 0, 0, MASK_ALPHA],
    "blue": [0, 0, 1, MASK_ALPHA],
    "green": [0, 1, 0, MASK_ALPHA],
    "yellow": [1, 1, 0, MASK_ALPHA],
    "cyan": [0, 1, 1, MASK_ALPHA],
    "magenta": [1, 0, 1, MASK_ALPHA],
}


def prompt_color(prompt: str, fallback_idx: int = 0) -> tuple[list[float], str]:
    """Return (mask_rgba, edge_color) for a text prompt."""
    key = prompt.strip().lower()
    if key in CLASS_COLOR_RGBA:
        rgba = CLASS_COLOR_RGBA[key]
        edge = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        return rgba, edge
    edge = FALLBACK_EDGE_COLORS[fallback_idx % len(FALLBACK_EDGE_COLORS)]
    return FALLBACK_RGBA[edge], edge


class SAM3LoRAInference:
    """SAM3 image model for inference (with optional LoRA weights)."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        use_base_model: bool = False,
        checkpoint_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize SAM3 for inference.

        Args:
            config_path: Path to training config YAML (required for LoRA mode)
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            use_base_model: Use original SAM3 without LoRA
            checkpoint_path: Path to base SAM3 checkpoint (default: sam3.pt)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to run on (default: "cuda")
        """
        self.use_base_model = use_base_model
        self.checkpoint_path = checkpoint_path or DEFAULT_SAM3_CHECKPOINT
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint_path}")
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if use_base_model:
            print("🔧 Initializing original SAM3 (no LoRA)...")
        else:
            if config_path is None:
                raise ValueError("--config is required unless --use-base-model is set")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            if weights_path is None:
                output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
                weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
                print(f"ℹ️  Auto-detected weights: {weights_path}")

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

            self.weights_path = weights_path
            print("🔧 Initializing SAM3 + LoRA...")

        print(f"   Device: {self.device}")
        print(f"   Base checkpoint: {self.checkpoint_path}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        if not use_base_model and self.checkpoint_path != DEFAULT_SAM3_CHECKPOINT:
            print(
                "⚠️  LoRA weights were trained on sam3.pt; using a different base "
                "checkpoint may give poor results."
            )

        print("\n📦 Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        if not use_base_model:
            print("🔗 Applying LoRA configuration...")
            lora_cfg = self.config["lora"]
            lora_config = LoRAConfig(
                rank=lora_cfg["rank"],
                alpha=lora_cfg["alpha"],
                dropout=0.0,
                target_modules=lora_cfg["target_modules"],
                apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
                apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
                apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
                apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
                apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
                apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
            )
            self.model = apply_lora_to_model(self.model, lora_config)
            print(f"💾 Loading LoRA weights from {weights_path}...")
            load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        ready_msg = "✅ SAM3 ready for inference!\n" if use_base_model else "✅ SAM3 + LoRA ready for inference!\n"
        print(ready_msg)

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        """
        Create a SAM3 datapoint from image and text prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        text_prompts: List[str]
    ) -> dict:
        """
        Run inference on an image with text prompts.

        Args:
            image_path: Path to input image
            text_prompts: List of text queries (e.g., ["crack", "defect"])

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...]},
                1: {'boxes': [...], 'scores': [...], 'masks': [...]}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        print(f"📷 Loaded image: {image_path}")
        print(f"   Size: {pil_image.size}")
        print(f"   Prompts: {text_prompts}")

        print("\n🔮 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt
            datapoint = self.create_datapoint(pil_image, [prompt])

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and convert from cxcywh to xyxy
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # Apply NMS to remove overlapping boxes
                keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
                boxes_xyxy = boxes_xyxy[keep_nms]
                kept_scores = kept_scores[keep_nms]
                num_keep = len(keep_nms)

                # Get masks and resize to original size
                if pred_masks is not None:
                    # Apply NMS filtering to masks too
                    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                    # Resize masks to original image size
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        masks_small.unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'num_detections': num_keep
                }
                print(f"   '{prompt}': {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'num_detections': 0
                }
                print(f"   '{prompt}': 0 detections")

        # Store original image for visualization
        results['_image'] = pil_image

        return results

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        prompt_keys = sorted([k for k in results.keys() if k != '_image'])
        total_detections = 0

        # Draw results for each prompt
        for prompt_idx, k in enumerate(prompt_keys):
            result = results[k]
            prompt = result['prompt']
            mask_rgba, edge_color = prompt_color(prompt, prompt_idx)

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            for det_idx in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None:
                    mask = masks[det_idx]
                    colored_mask = np.zeros((*mask.shape, 4))
                    colored_mask[mask] = mask_rgba
                    ax.imshow(colored_mask)

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[det_idx]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=edge_color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[det_idx] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=edge_color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

        ax.axis('off')

        legend_handles = []
        for idx, k in enumerate(prompt_keys):
            result = results[k]
            _, edge_color = prompt_color(result['prompt'], idx)
            n = result['num_detections']
            label = f"{result['prompt']} ({n})"
            legend_handles.append(
                patches.Patch(facecolor=edge_color, edgecolor=edge_color, alpha=0.7, label=label)
            )
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_handles), 4),
            fontsize=10,
            framealpha=0.9,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n✅ Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 inference (LoRA or base model)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (required unless --use-base-model)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 weights without LoRA"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_SAM3_CHECKPOINT,
        help=(
            "Path to base SAM3 checkpoint "
            f"(default: {DEFAULT_SAM3_CHECKPOINT})"
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer overlapping boxes)"
    )

    args = parser.parse_args()

    if not args.use_base_model and args.config is None:
        parser.error("--config is required unless --use-base-model is set")

    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        use_base_model=args.use_base_model,
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou
    )

    # Run inference
    results = inferencer.predict(args.image, args.prompt)

    # Visualize
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks
    )

    # Print summary
    print("\n" + "="*60)
    print("📊 Summary:")
    for idx in sorted([k for k in results.keys() if k != '_image']):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result['num_detections'] > 0 and result['scores'] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
    print("="*60)


if __name__ == "__main__":
    main()

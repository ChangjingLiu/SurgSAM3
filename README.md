# SurgSAM3

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Parameter-Efficient Adaptation of SAM3 for Prompt-Driven Surgical Concept Segmentation**

[Installation](#installation) • [Quick Start](#quick-start) • [Training](#training) • [Evaluation](#evaluation) • [Citation](#citation)

</div>

---

## Overview

**SurgSAM3** adapts [SAM3](https://huggingface.co/facebook/sam3) to multi-domain surgical segmentation using **LoRA** (Low-Rank Adaptation). Given a text prompt (e.g. `grasper`, `liver`), the model produces semantic masks for surgical instruments and anatomical structures.

**Supported benchmarks (test):**

- [CholecSeg8k](https://github.com/ternaus/robotic-surgery-segmentation) — laparoscopic cholecystectomy
- [EndoVis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) — instrument segmentation
- [CaDIS](https://cataracts-semantic-segmentation.grand-challenge.org/) — cataract surgery

### Key features

- Single-prompt training on a mixed surgical COCO dataset (one category name per query)
- ~1% trainable parameters vs. full SAM3 fine-tuning
- Unified mIoU / Dice evaluation across domains (`eval_miou_sam3_lora.py`)
- Multi-method radar plots (`plot_miou_radar_dirs.py`)
- Optional Medical-SAM3 2D/3D checkpoint as base model

### Acknowledgement

This project is built on top of [SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA) by Sompote et al. (KMUTT). We thank the authors for the LoRA training framework.

---

## Installation

### 1. Request SAM3 access

1. Request access at [facebook/sam3](https://huggingface.co/facebook/sam3)
2. Create a token at [Hugging Face Settings](https://huggingface.co/settings/tokens)

### 2. Clone and install

```bash
git clone https://github.com/ChangjingLiu/SurgSAM3.git
cd SurgSAM3
pip install -e .
hf auth login
```

### 3. Download checkpoints

Place checkpoints locally and update paths in the YAML configs:

| Model | Config key | Source |
|-------|------------|--------|
| SAM3 base | `model.checkpoint_path` in `configs/light_lora_config_r16_single_prompt_base.yaml` | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| Medical-SAM3 2D | `model.checkpoint_path` in `configs/light_lora_config_r16_single_prompt_medical_sam3.yaml` | [Medical-SAM3](https://modelscope.cn/models/ChongCong/Medical-SAM3) |

> Pre-trained LoRA weights will be released on Hugging Face (TBD).

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA (recommended)

---

## Data preparation

Training expects **COCO format** with one annotation file per split:

```
datasets/surgical_mix_coco_stride5_single_prompt/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
└── valid/          # optional
    ├── *.jpg
    └── _annotations.coco.json
```

Each image uses **single-prompt** supervision: one text query per category instance (category name from COCO `categories`).

Per-domain test sets (Cholec / EndoVis / CaDIS) should follow the same layout with `test/_annotations.coco.json`.

> Dataset download and preprocessing scripts are not included in this repo. Please prepare COCO exports following your benchmark licenses.

Set `training.data_dir` in the config to your dataset root before training.

---

## Quick Start

### Train SAM3 + LoRA (recommended)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sam3_lora_native.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --device 0
```

Multi-GPU:

```bash
python train_sam3_lora_native.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --device 0 1 2 3
```

### Train Medical-SAM3 2D + LoRA

```bash
python train_sam3_lora_native.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --device 0 1 2 3
```

Checkpoints are saved under `output.output_dir` in each config:

- SAM3 + LoRA: `outputs_r16_single_prompt_base/surgical_mix_lora_stride5/best_lora_weights.pt`
- Medical-SAM3 + LoRA: `outputs_r16_single_prompt_medical_sam3d/surgical_mix_lora_stride5/best_lora_weights.pt`

---

## Training

| Config | Base model | LoRA rank | Use case |
|--------|------------|-----------|----------|
| `configs/light_lora_config_r16_single_prompt_base.yaml` | SAM3 (`sam3.pt`) | 16 | Main surgical pipeline |
| `configs/light_lora_config_r16_single_prompt_medical_sam3.yaml` | Medical-SAM3 2D | 16 | Medical-domain prior |
| `configs/light_lora_config.yaml` | SAM3 | 16 | Legacy multi-query setup |

```bash
python train_sam3_lora_native.py --config <config.yaml> --device 0
```

Training monitors validation loss; full segmentation metrics are computed offline (see [Evaluation](#evaluation)).

**Multi-GPU:** pass multiple GPU ids to `--device`, e.g. `--device 0 1 2 3`. Effective batch size scales with the number of GPUs.

---

## Evaluation

`eval_miou_sam3_lora.py` reports per-class **IoU**, **Dice**, and **mIoU**.

### Base SAM3 (no LoRA)

```bash
python eval_miou_sam3_lora.py \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_base.json
```

### SAM3 + LoRA

```bash
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_lora.json
```

Repeat with EndoVis and CaDIS test directories.

### Medical-SAM3 2D (dedicated script)

```bash
python eval_miou_medsam3.py \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_medical_sam3_2d.json
```

### Medical-SAM3 3D checkpoint

```bash
python eval_miou_sam3_lora.py \
  --checkpoint /path/to/checkpoint_3D.pt \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_medical_sam3_3d.json
```

| Flag | Description |
|------|-------------|
| `--config` | Training config (required when using `--weights`) |
| `--weights` | LoRA weights; omit for base-model-only eval |
| `--checkpoint` | Override base SAM3 / Medical-SAM3 checkpoint |
| `--data_dir` | COCO test folder with `_annotations.coco.json` |
| `--json-out` | Save metrics JSON |

For COCO-style validation with mAP / cgF1, use `validate_sam3_lora.py` (upstream workflow).

---

## Visualization

Compare multiple methods with radar charts (IoU + Dice):

```bash
python plot_miou_radar_dirs.py \
  --sam3-dir results/sam3_base \
  --sam3-lora-dir results/sam3_lora \
  --medical-sam3-2d-dir results/medical_sam3_2d \
  --medical-sam3-3d-dir results/medical_sam3_3d \
  --medical-sam3-2d-lora-dir results/medical_sam3_2d_lora \
  --combined --metric both --min-max-iou 40 \
  --out results/miou_dice_radar.png
```

Each `--*-dir` should contain `miou_*_test*.json` files from evaluation.

---

## Inference

```bash
python infer_sam.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image path/to/frame.jpg \
  --prompt "grasper" \
  --output output.png
```

> Video inference is frame-by-frame; temporal post-processing is not included.

---

## Project structure

```
SurgSAM3/
├── configs/
│   ├── light_lora_config_r16_single_prompt_base.yaml
│   └── light_lora_config_r16_single_prompt_medical_sam3.yaml
├── train_sam3_lora_native.py      # Training
├── eval_miou_sam3_lora.py         # mIoU / Dice evaluation
├── eval_miou_medsam3.py           # Medical-SAM3 2D eval
├── plot_miou_radar_dirs.py        # Multi-method radar plots
├── infer_sam.py                   # Single-image inference
├── validate_sam3_lora.py          # mAP / cgF1 validation (upstream)
├── lora_layers.py                 # LoRA implementation
└── sam3/                          # SAM3 model code
```

---

## Troubleshooting

**Hugging Face access denied**

- Confirm SAM3 access is approved at [facebook/sam3](https://huggingface.co/facebook/sam3)
- Run `hf auth login` or set `HF_TOKEN`

**CUDA OOM**

- Use `batch_size: 1` and `gradient_accumulation_steps: 8` in the config
- Reduce LoRA rank or train on fewer GPUs

**Checkpoint not found**

- Set `model.checkpoint_path` in the YAML to your local SAM3 / Medical-SAM3 path

---

## Citation

If you use this code, please cite our paper and the upstream SAM3-LoRA project:

```bibtex
@article{liu2026surgsam3,
  title={Parameter-Efficient Adaptation of SAM3 for Prompt-Driven Surgical Concept Segmentation},
  author={Liu, Changjing and others},
  year={2026}
}

@software{sam3_lora,
  title={SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author={Sompote and AI Research Group, KMUTT},
  year={2025},
  url={https://github.com/Sompote/SAM3_LoRA}
}
```

Also cite SAM3, LoRA, and the surgical datasets you use.

---

## License

This project follows the upstream [Apache 2.0](LICENSE) license. SAM3 weights are subject to Meta's license on Hugging Face.

---

<div align="center">

**Repository:** https://github.com/ChangjingLiu/SurgSAM3

</div>

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

**SurgSAM3** adapts [SAM3](https://huggingface.co/facebook/sam3) to multi-domain surgical segmentation using **LoRA**. Given a text prompt (e.g. `grasper`, `liver`), the model produces semantic masks for surgical instruments and anatomical structures.

**Benchmarks:** [CholecSeg8k](https://github.com/ternaus/robotic-surgery-segmentation) · [EndoVis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) · [CaDIS](https://cataracts-semantic-segmentation.grand-challenge.org/)

Built on [SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA) (KMUTT). We thank the authors for the training framework.

---

## Installation

1. Request access at [facebook/sam3](https://huggingface.co/facebook/sam3) and create a [Hugging Face token](https://huggingface.co/settings/tokens).
2. Clone and install:

```bash
git clone https://github.com/ChangjingLiu/SurgSAM3.git
cd SurgSAM3
pip install -e .
hf auth login
```

3. Download [SAM3 weights](https://huggingface.co/facebook/sam3) and set `model.checkpoint_path` in `configs/light_lora_config_r16_single_prompt_base.yaml`.

> Pre-trained LoRA weights will be released on Hugging Face (TBD).

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA (recommended)

---

## Data preparation

COCO format with `_annotations.coco.json` per split:

```
datasets/surgical_mix_coco_stride5_single_prompt/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
└── valid/          # optional
    ├── *.jpg
    └── _annotations.coco.json
```

Single-prompt supervision: each query uses the category name from COCO `categories`. Set `training.data_dir` in the config before training.

---

## Quick Start

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sam3_lora_native.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --device 0
```

Multi-GPU: `--device 0 1 2 3`

Weights are saved to `outputs_r16_single_prompt_base/surgical_mix_lora_stride5/best_lora_weights.pt`.

---

## Training

```bash
python train_sam3_lora_native.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --device 0
```

Main config: `configs/light_lora_config_r16_single_prompt_base.yaml` (SAM3 base, LoRA rank 16, single-prompt).

Training monitors validation loss; segmentation metrics are computed offline.

---

## Evaluation

`eval_miou_sam3_lora.py` reports per-class **IoU**, **Dice**, and **mIoU**.

**Base SAM3:**

```bash
python eval_miou_sam3_lora.py \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_base.json
```

**SAM3 + LoRA:**

```bash
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /path/to/cholecseg8k_coco_stride5_benchmark/test \
  --json-out results/miou_cholec_test_lora.json
```

Repeat for EndoVis and CaDIS test sets. Key flags: `--config`, `--weights`, `--data_dir`, `--json-out`.

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

---

## Project structure

```
SurgSAM3/
├── configs/light_lora_config_r16_single_prompt_base.yaml
├── train_sam3_lora_native.py
├── eval_miou_sam3_lora.py
├── infer_sam.py
├── lora_layers.py
└── sam3/
```

---

## Citation

```bibtex
@misc{liu2026parameterefficientadaptationsam3promptdriven,
      title={Parameter-Efficient Adaptation of SAM3 for Prompt-Driven Surgical Concept Segmentation}, 
      author={Changjing Liu and Yiming Huang and Beilei Cui and Liangjing Shao and Long Bai and Yanheng Li and Haoxuan Che and Hongliang Ren},
      year={2026},
      eprint={2607.23694},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.23694}, 
}

@software{sam3_lora,
  title={SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author={Sompote and AI Research Group, KMUTT},
  year={2025},
  url={https://github.com/Sompote/SAM3_LoRA}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE). SAM3 weights are subject to Meta's Hugging Face license.

---

<div align="center">

**Repository:** https://github.com/ChangjingLiu/SurgSAM3

</div>

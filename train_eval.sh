# 第一次创建 zellij 窗口
zellij new -n sam3train
# 之后直接 attach 到窗口
zellij attach -c sam3train
# 查看窗口列表
zellij list-sessions
# 删除窗口
zellij kill-session sam3train



cd /mnt/data2_hdd/changjing/SAM3_LoRA
conda activate sam3_lora

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sam3_lora_native.py \
--config configs/light_lora_config.yaml \
--device 0

python train_sam3_lora_native.py \
--config configs/light_lora_config_r32.yaml \
--device 0

# medical-sam3+lora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sam3_lora_native.py \
--config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
--device 0 1 2 3

# sam3 +lora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sam3_lora_native.py \
--config configs/light_lora_config_r16_single_prompt_base.yaml \
--device 0 1 2 3

# 微调结果
# Cholec test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs/miou_cholec_test.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs/miou_endovis_test.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs/miou_cadis_test.json

# single prompt: medical-sam3 base (checkpoint from config, no LoRA)
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_base_lora/miou_cholec_test_medical_sam3_base.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_base_lora/miou_endovis_test_medical_sam3_base.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs_r16_single_prompt_base_lora/miou_cadis_test_medical_sam3_base.json

# single prompt: sam3 + LoRA
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/last_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_base/miou_cholec_test_lora.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/last_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_base/miou_endovis_test_lora.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_base.yaml \
  --weights outputs_r16_single_prompt_base/surgical_mix_lora_stride5/last_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs_r16_single_prompt_base/miou_cadis_test_lora.json

# single prompt: medical-sam3 base (checkpoint from config, with LoRA)
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --weights outputs_r16_single_prompt_medical_sam3_lora/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_medical_sam3_lora/miou_cholec_test_medical_sam3_base.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --weights outputs_r16_single_prompt_medical_sam3_lora/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs_r16_single_prompt_medical_sam3_lora/miou_endovis_test_medical_sam3_base.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r16_single_prompt_medical_sam3.yaml \
  --weights outputs_r16_single_prompt_medical_sam3_lora/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs_r16_single_prompt_medical_sam3_lora/miou_cadis_test_medical_sam3_base.json

# 微调结果r=32
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r32.yaml \
  --weights outputs_r32/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs_r32/miou_cholec_test.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r32.yaml \
  --weights outputs_r32/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs_r32/miou_endovis_test.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config_r32.yaml \
  --weights outputs_r32/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs_r32/miou_cadis_test.json

# 微调结果r=16
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs_r16/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs_r16/miou_cholec_test.json
# EndoVis test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs_r16/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs_r16/miou_endovis_test.json
# CaDIS test
python eval_miou_sam3_lora.py \
  --config configs/light_lora_config.yaml \
  --weights outputs_r16/surgical_mix_lora_stride5/best_lora_weights.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs_r16/miou_cadis_test.json

# sam3原版结果
cd /mnt/data2_hdd/changjing/SAM3_LoRA
conda activate sam3_lora

python eval_miou_sam3_lora.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs/miou_cholec_test_base.json

# EndoVis test
python eval_miou_sam3_lora.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs/miou_endovis_test_base.json

# CaDIS test
python eval_miou_sam3_lora.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs/miou_cadis_test_base.json

# Medical-SAM3 checkpoint_2D.pt (dedicated eval script)
python eval_miou_medsam3.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs/miou_cholec_test_medical_sam3_2d.json

python eval_miou_medsam3.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs/miou_endovis_test_medical_sam3_2d.json

python eval_miou_medsam3.py \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs/miou_cadis_test_medical_sam3_2d.json

# Medical-SAM3 checkpoint_3D.pt
python eval_miou_sam3_lora.py \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --json-out outputs/miou_cholec_test_medical_sam3_3d.json

python eval_miou_sam3_lora.py \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/endovis18_coco_stride5_benchmark/test \
  --json-out outputs/miou_endovis_test_medical_sam3_3d.json

python eval_miou_sam3_lora.py \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --data_dir /home/ren7/NAS2/changjing/Endo_VL/cadisv2_coco_stride5/test \
  --json-out outputs/miou_cadis_test_medical_sam3_3d.json

# ORD:  Radar chart: IoU + Dice side by side (filter classes with max IoU < 15%)
python plot_miou_radar.py \
  --outputs-dir outputs \
  --combined \
  --metric both \
  --min-max-iou 15 \
  --out outputs/miou_dice_radar_combined.png

# IoU only / separate panels
python plot_miou_radar.py \
  --no-combined \
  --metric iou \
  --outputs-dir outputs \
  --out outputs/miou_radar_overall.png

python plot_miou_radar.py \
  --outputs-dir outputs_r16 \
  --combined \
  --metric both \
  --min-max-iou 50 \
  --out outputs_r16/miou_dice_radar_combined_r16.png


python plot_miou_radar_dirs.py \
  --sam3-dir outputs_sam3\
  --sam3-lora-dir outputs_r16_single_prompt_base_lora \
  --medical-sam3-2d-dir outputs_medical_sam3_2d \
  --medical-sam3-3d-dir outputs_sam3_3d \
  --medical-sam3-2d-lora-dir outputs_r16_single_prompt_medical_sam3_lora \
  --combined --metric both --min-max-iou 40 \
  --out outputs_radar/miou_dice_radar_multi_dir.png


python plot_miou_radar_dirs.py \
  --sam3-dir outputs_sam3\
  --sam3-lora-dir outputs_r16_single_prompt_base_lora \
  --medical-sam3-2d-dir outputs_medical_sam3_2d \
  --medical-sam3-3d-dir outputs_sam3_3d \
  --combined --metric both --min-max-iou 40 \
  --out outputs_radar/miou_dice_radar_multi_dir.png
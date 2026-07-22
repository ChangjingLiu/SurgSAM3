cd /mnt/data2_hdd/changjing/SAM3_LoRA
conda activate sam3_lora
python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs_r16_single_prompt_base_lora/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_00240__frame_240_endo.png \
  --prompt "Liver" "Gallbladder" "Fat" "Gastrointestinal Tract" "Grasper" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_frame240.png

# test
export CUDA_VISIBLE_DEVICES=1
cd /mnt/data2_hdd/changjing/SAM3_LoRA
conda activate sam3_lora
python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/keyframe_0000.png \
  --prompt "Liver" "Gallbladder" "Fat" "Gastrointestinal Tract" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs_video01_00240/infer_cholec_frame240.png


# Original SAM3 (no LoRA)
python infer_sam.py \
  --use-base-model \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_00240__frame_240_endo.png \
  --prompt "Liver" "Gallbladder" "Fat" "Gastrointestinal Tract" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_frame240_base.png


cd /mnt/data2_hdd/changjing/SAM3_LoRA
conda activate sam3_lora

python compare_lora_base.py \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_00080__frame_100_endo.png \
  --data-dir /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --threshold 0.3 \
  --output outputs/compare_cholec_frame100.png



#对比
#video01_00240_grasper
  python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_00240__frame_240_endo.png \
  --prompt "Grasper" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_frame240_grasper.png

  python infer_sam.py \
  --use-base-model \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video02_15750 __frame_240_endo.png \
  --prompt "Grasper"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_frame240_base_grasper.png

# video01_15019
  python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_15019__frame_15019_endo.png \
  --prompt "Abdominal Wall" "Grasper" "L-hook Electrocautery" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video01_15019_Abdominal_Wall_Grasper_L-hook_Electrocautery.png

  python infer_sam.py \
  --use-base-model \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_15019__frame_15019_endo.png \
  --prompt "Abdominal Wall" "Grasper" "L-hook Electrocautery"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video01_15019_Abdominal_Wall_Grasper_L-hook_Electrocautery_base.png

    python infer_sam.py \
  --use-base-model \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video01__video01_15019__frame_15019_endo.png \
  --prompt "Abdominal Wall" "Grasper" "L-hook Electrocautery"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video01_15019_Abdominal_Wall_Grasper_L-hook_Electrocautery_medical_sam3_3d.png

# video12_15750_L-hook Electrocautery
  python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video12__video12_15750__frame_15750_endo.png \
  --prompt "Fat" "Grasper" "L-hook Electrocautery" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video12_15750_fat_grasper_L-hook_Electrocautery.png

  python infer_sam.py \
  --use-base-model \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video12__video12_15750__frame_15750_endo.png \
  --prompt "Fat" "Grasper" "L-hook Electrocautery"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video12_15750_fat_grasper_L-hook_Electrocautery_base.png

    python infer_sam.py \
  --use-base-model \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --image /home/ren7/NAS2/changjing/Endo_VL/cholecseg8k_coco_stride5_benchmark/test/video12__video12_15750__frame_15750_endo.png \
  --prompt "Fat" "Grasper" "L-hook Electrocautery"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_video12_15750_fat_grasper_L-hook_Electrocautery_medical_sam3_3d.png

#endovis_2018_release_2__seq_5__frame000
  python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis18_coco_stride5_benchmark/test/endovis_2018_release_2__seq_5__frame000.png \
  --prompt "instrument-wrist" "kidney-parenchyma" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_endovis_2018_release_2__seq_5__frame000_instrument-wrist_kidney-parenchyma.png

  python infer_sam.py \
  --use-base-model \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis18_coco_stride5_benchmark/test/endovis_2018_release_2__seq_5__frame000.png \
  --prompt "instrument-wrist" "kidney-parenchyma"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_cholec_endovis_2018_release_2__seq_5__frame000_instrument-wrist_kidney-parenchyma_base.png

# Medical-SAM3 checkpoint_3D.pt (base model, no LoRA)
  python infer_sam.py \
  --use-base-model \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis18_coco_stride5_benchmark/test/endovis_2018_release_2__seq_5__frame000.png \
  --prompt "instrument-wrist" "kidney-parenchyma" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_endovis_seq5_frame000_medical_sam3_3d.png

# Medical-SAM3 checkpoint_2D.pt (dedicated script, image-only weight format)
  python infer_medsam3.py \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis18_coco_stride5_benchmark/test/endovis_2018_release_2__seq_5__frame000.png \
  --prompt "instrument-wrist" "kidney-parenchyma" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_endovis_seq5_frame000_medical_sam3_2d.png


# seq 9
  python infer_sam.py \
  --config configs/light_lora_config.yaml \
  --weights outputs/surgical_mix_lora_stride5/best_lora_weights.pt \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis_2018_release_3/seq_9/left_frames/frame000.png \
  --prompt "instrument-shaft" "instrument-clasper" "instrument-wrist" "covered-kidney" \
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_endovis_seq9_frame000_instrument-shaft_instrument-clasper_instrument-wrist_covered-kidney.png

  python infer_sam.py \
  --use-base-model \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis_2018_release_3/seq_9/left_frames/frame000.png \
  --prompt "instrument-shaft" "instrument-clasper" "instrument-wrist" "covered-kidney"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_endovis_seq9_frame000_instrument-shaft_instrument-clasper_instrument-wrist_covered-kidney_base.png

    python infer_sam.py \
  --use-base-model \
  --checkpoint /mnt/data2_hdd/changjing/modelscope/ChongCong/Medical-SAM3/checkpoint_3D.pt \
  --image /mnt/data2_hdd/changjing/SAM3_LoRA/datasets/Endo_VL/endovis_2018_release_3/seq_9/left_frames/frame000.png \
  --prompt "instrument-shaft" "instrument-clasper" "instrument-wrist" "covered-kidney"\
  --threshold 0.3 \
  --nms-iou 0.7 \
  --output outputs/infer_endovis_seq9_frame000_instrument-shaft_instrument-clasper_instrument-wrist_covered-kidney_medical_sam3_3d.png

GPU_NUM=1
CFG="/home/ubuntu/roisul/Open-GroundingDino/config/cfg_odvg.py"
DATASETS="/home/ubuntu/roisul/Open-GroundingDino/config/datasets_mixed_odvg.json"
OUTPUT_DIR="/home/ubuntu/roisul/training_amz_250k_3001_mmcvfix/"

# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config_file ${CFG} \
    --datasets ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --pretrain_model_path /home/ubuntu/roisul/training_amz_250k_3001_mmcvfix/checkpoint.pth \
    --options text_encoder_type="/home/ubuntu/roisul/bert"

#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Basic parameters
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

# Distributed training parameters
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Model paths
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}

# LoRA parameters (using peft)
USE_LORA=${USE_LORA:-"false"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-"none"}
LORA_RESUME=${LORA_RESUME:-""}
MERGE_LORA=${MERGE_LORA:-"false"}

echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "GPU_NUM = $GPU_NUM"
echo "CFG = $CFG"
echo "DATASETS = $DATASETS"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "NNODES = $NNODES"
echo "NODE_RANK = $NODE_RANK"
echo "PORT = $PORT"
echo "MASTER_ADDR = $MASTER_ADDR"
echo "PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH"
echo "TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE"
echo ""
echo "LoRA Configuration (peft):"
echo "USE_LORA = $USE_LORA"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"
echo "LORA_DROPOUT = $LORA_DROPOUT"
echo "LORA_BIAS = $LORA_BIAS"
echo "LORA_RESUME = $LORA_RESUME"
echo "MERGE_LORA = $MERGE_LORA"
echo "=============================================="

# Build command
CMD="torchrun --nproc_per_node=${GPU_NUM} --master_port=${PORT}"

if [ "$NNODES" -gt 1 ]; then
    CMD="$CMD --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR}"
fi

CMD="$CMD main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CFG} \
    --datasets ${DATASETS} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --options text_encoder_type=${TEXT_ENCODER_TYPE}"

# Add LoRA parameters
if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora"
    CMD="$CMD --lora_r ${LORA_R}"
    CMD="$CMD --lora_alpha ${LORA_ALPHA}"
    CMD="$CMD --lora_dropout ${LORA_DROPOUT}"
    CMD="$CMD --lora_bias ${LORA_BIAS}"
    
    if [ -n "$LORA_RESUME" ]; then
        CMD="$CMD --lora_resume ${LORA_RESUME}"
    fi
    
    if [ "$MERGE_LORA" = "true" ]; then
        CMD="$CMD --merge_lora_after_train"
    fi
fi

echo "Executing: $CMD"
eval $CMD

# CUDA_VISIBLE_DEVICES=2,3 bash train_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output/0121_exp1
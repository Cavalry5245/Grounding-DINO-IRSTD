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
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Model paths
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"training_output_lora/lora_exp2_0105/merged_model.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}

# LoRA parameters (using peft)
USE_LORA=${USE_LORA:-"false"}
LORA_RESUME=${LORA_RESUME:-""}
MERGE_LORA=${MERGE_LORA:-"false"}

echo "=============================================="
echo "Testing Configuration"
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
    --eval \
    -c ${CFG} \
    --datasets ${DATASETS} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --options text_encoder_type=${TEXT_ENCODER_TYPE}"

# Add LoRA parameters if needed
if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora"
    
    if [ -n "$LORA_RESUME" ]; then
        CMD="$CMD --lora_resume ${LORA_RESUME}"
    fi
    
    if [ "$MERGE_LORA" = "true" ]; then
        CMD="$CMD --merge_lora_after_train"
    fi
fi

echo "Executing: $CMD"
eval $CMD
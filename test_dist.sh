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
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/lora_1230_exp1/merged_model.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}

# LoRA parameters (using peft)
USE_LORA=${USE_LORA:-"true"}
LORA_RESUME=${LORA_RESUME:-""}
MERGE_LORA=${MERGE_LORA:-"false"}

# ==================== 新增：扩展评估参数 ====================
# 一键启用所有扩展功能
EXTENDED_EVAL=${EXTENDED_EVAL:-"true"}

# 或者单独控制每个功能
SAVE_JSON=${SAVE_JSON:-"false"}       # 保存COCO格式JSON结果
PLOT_CURVES=${PLOT_CURVES:-"false"}   # 绘制PR/P/R/F1曲线
VISUALIZE=${VISUALIZE:-"false"}       # 可视化GT vs 预测对比图
SAVE_METRICS=${SAVE_METRICS:-"false"} # 保存详细指标到文件
# ===========================================================

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
echo ""
echo "Extended Evaluation Configuration:"
echo "EXTENDED_EVAL = $EXTENDED_EVAL"
echo "SAVE_JSON = $SAVE_JSON"
echo "PLOT_CURVES = $PLOT_CURVES"
echo "VISUALIZE = $VISUALIZE"
echo "SAVE_METRICS = $SAVE_METRICS"
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

# ==================== 新增：添加扩展评估参数 ====================
if [ "$EXTENDED_EVAL" = "true" ]; then
    CMD="$CMD --extended_eval"
else
    # 单独控制每个功能
    if [ "$SAVE_JSON" = "true" ]; then
        CMD="$CMD --save_json"
    fi
    
    if [ "$PLOT_CURVES" = "true" ]; then
        CMD="$CMD --plot_curves"
    fi
    
    if [ "$VISUALIZE" = "true" ]; then
        CMD="$CMD --visualize"
    fi
    
    if [ "$SAVE_METRICS" = "true" ]; then
        CMD="$CMD --save_metrics"
    fi
fi
# ==============================================================

echo ""
echo "Executing command:"
echo "$CMD"
echo ""

eval $CMD

# bash test_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_coco_test.json ./eval_output/1230_exp1/sirst
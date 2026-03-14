#!/bin/bash
# ============================================================
# train_lora.sh - 灵活的 LoRA / HF-LoRA 训练脚本
# ============================================================
# 用法示例：
#
# 1. 标准 LoRA 训练：
#    USE_LORA=true bash train_lora.sh 2 configs/cfg_odvg.py datasets/custom.json output/exp1
#
# 2. HF-LoRA 训练（Backbone 层用高频增强）：
#    USE_LORA=true USE_HF_LORA=true bash train_lora.sh 2 configs/cfg_odvg.py datasets/custom.json output/exp2
#
# 3. 自定义 HF-LoRA 目标层：
#    USE_LORA=true USE_HF_LORA=true HF_LORA_MODULES="qkv proj fc1 fc2" bash train_lora.sh 2 ...
#
# 4. 全量微调（不使用 LoRA）：
#    bash train_lora.sh 2 configs/cfg_odvg.py datasets/custom.json output/exp3
#
# 5. 从检查点恢复 HF-LoRA 训练：
#    USE_LORA=true USE_HF_LORA=true LORA_RESUME=output/exp2/lora_checkpoint_best bash train_lora.sh 2 ...
#
# 6. 评估模式：
#    USE_LORA=true USE_HF_LORA=true LORA_RESUME=output/exp2/lora_checkpoint_best EVAL_ONLY=true bash train_lora.sh 2 ...
#
# 7. 高 rank + 全部层 HF-LoRA（消融实验）：
#    USE_LORA=true USE_HF_LORA=true LORA_R=16 LORA_ALPHA=32 \
#    HF_LORA_MODULES="qkv proj fc1 fc2 sampling_offsets attention_weights value_proj output_proj" \
#    bash train_lora.sh 2 ...
# ============================================================

export TOKENIZERS_PARALLELISM=false

# ========================
# 基本参数（位置参数）
# ========================
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

if [ -z "$GPU_NUM" ] || [ -z "$CFG" ] || [ -z "$DATASETS" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash train_lora.sh GPU_NUM CFG DATASETS OUTPUT_DIR"
    echo ""
    echo "Example:"
    echo "  USE_LORA=true bash train_lora.sh 2 configs/cfg_odvg.py datasets/custom.json output/exp1"
    echo "  USE_LORA=true USE_HF_LORA=true bash train_lora.sh 2 configs/cfg_odvg.py datasets/custom.json output/exp2"
    exit 1
fi

# ========================
# 分布式训练参数
# ========================
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ========================
# 模型路径
# ========================
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}

# ========================
# LoRA 基础参数
# ========================
USE_LORA=${USE_LORA:-"false"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-"none"}
LORA_RESUME=${LORA_RESUME:-""}
MERGE_LORA=${MERGE_LORA:-"false"}

# LoRA 目标层（所有要注入 LoRA 的层）
# 默认覆盖 Backbone + Encoder/Decoder + Fusion + FFN
# LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"qkv proj fc1 fc2 sampling_offsets attention_weights value_proj output_proj v_proj l_proj values_v_proj values_l_proj out_v_proj out_l_proj linear1 linear2"}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"qkv query value v_proj values_v_proj values_proj"}

# 需要解冻的层（检测头等，不使用 LoRA 而是全量训练）
LORA_UNFREEZE_LAYERS=${LORA_UNFREEZE_LAYERS:-"class_embed bbox_embed label_enc"}

# ========================
# HF-LoRA 参数（创新点）
# ========================
USE_HF_LORA=${USE_HF_LORA:-"false"}

# HF-LoRA 目标层（在这些层使用高频增强，其余层使用标准 LoRA）
# 默认只在 Backbone 的核心层使用 HF-LoRA
HF_LORA_MODULES=${HF_LORA_MODULES:-"qkv fc1 fc2"}

# ========================
# 训练控制参数
# ========================
EVAL_ONLY=${EVAL_ONLY:-"false"}
SAVE_BEST_ONLY=${SAVE_BEST_ONLY:-"true"}
FIND_UNUSED_PARAMS=${FIND_UNUSED_PARAMS:-"true"}
SAVE_LOG=${SAVE_LOG:-"false"}
NUM_WORKERS=${NUM_WORKERS:-8}

# ========================
# 可选：覆盖 config 中的超参数
# ========================
# 这些参数通过 --options 传给 config，留空则使用 config 文件中的默认值
EPOCHS=${EPOCHS:-""}
LR=${LR:-""}
LR_DROP=${LR_DROP:-""}
BATCH_SIZE=${BATCH_SIZE:-""}
WEIGHT_DECAY=${WEIGHT_DECAY:-""}

# ============================================================
# 打印配置信息
# ============================================================

# 确定训练模式名称
if [ "$USE_LORA" = "true" ]; then
    if [ "$USE_HF_LORA" = "true" ]; then
        TRAIN_MODE="HF-LoRA (High-Frequency Enhanced LoRA)"
    else
        TRAIN_MODE="Standard LoRA"
    fi
else
    TRAIN_MODE="Full Fine-tuning (No LoRA)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Training Configuration                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Mode: ${TRAIN_MODE}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Basic:"
echo "║    GPU_NUM          = $GPU_NUM"
echo "║    CFG              = $CFG"
echo "║    DATASETS          = $DATASETS"
echo "║    OUTPUT_DIR        = $OUTPUT_DIR"
echo "║    PRETRAIN_MODEL    = $(basename $PRETRAIN_MODEL_PATH)"
echo "╠══════════════════════════════════════════════════════════╣"
if [ "$USE_LORA" = "true" ]; then
echo "║  LoRA:"
echo "║    LORA_R            = $LORA_R"
echo "║    LORA_ALPHA        = $LORA_ALPHA"
echo "║    SCALING           = $(echo "scale=2; $LORA_ALPHA / $LORA_R" | bc)"
echo "║    LORA_DROPOUT      = $LORA_DROPOUT"
echo "║    LORA_BIAS         = $LORA_BIAS"
echo "║    TARGET_MODULES    = $LORA_TARGET_MODULES"
echo "║    UNFREEZE_LAYERS   = $LORA_UNFREEZE_LAYERS"
if [ "$USE_HF_LORA" = "true" ]; then
echo "║  HF-LoRA (Innovation):"
echo "║    HF_LORA_MODULES   = $HF_LORA_MODULES"
fi
if [ -n "$LORA_RESUME" ]; then
echo "║  Resume:"
echo "║    LORA_RESUME       = $LORA_RESUME"
fi
echo "╠══════════════════════════════════════════════════════════╣"
fi
echo "║  Control:"
echo "║    EVAL_ONLY         = $EVAL_ONLY"
echo "║    SAVE_BEST_ONLY    = $SAVE_BEST_ONLY"
echo "║    MERGE_LORA        = $MERGE_LORA"
echo "║    FIND_UNUSED       = $FIND_UNUSED_PARAMS"
if [ -n "$EPOCHS" ]; then
echo "║    EPOCHS            = $EPOCHS"
fi
if [ -n "$LR" ]; then
echo "║    LR                = $LR"
fi
if [ -n "$BATCH_SIZE" ]; then
echo "║    BATCH_SIZE        = $BATCH_SIZE"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 构建命令
# ============================================================

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# torchrun 基础命令
CMD="torchrun --nproc_per_node=${GPU_NUM} --master_port=${PORT}"

if [ "$NNODES" -gt 1 ]; then
    CMD="$CMD --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR}"
fi

# main.py 基础参数
CMD="$CMD main.py \
    --output_dir ${OUTPUT_DIR} \
    -c ${CFG} \
    --datasets ${DATASETS} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --num_workers ${NUM_WORKERS} \
    --options text_encoder_type=${TEXT_ENCODER_TYPE}"

# ------ 可选超参数覆盖（通过 --options 传递）------
if [ -n "$EPOCHS" ]; then
    CMD="$CMD epochs=${EPOCHS}"
fi
if [ -n "$LR" ]; then
    CMD="$CMD lr=${LR}"
fi
if [ -n "$LR_DROP" ]; then
    CMD="$CMD lr_drop=${LR_DROP}"
fi
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD batch_size=${BATCH_SIZE}"
fi
if [ -n "$WEIGHT_DECAY" ]; then
    CMD="$CMD weight_decay=${WEIGHT_DECAY}"
fi

# ------ LoRA 参数 ------
if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora"
    CMD="$CMD --lora_r ${LORA_R}"
    CMD="$CMD --lora_alpha ${LORA_ALPHA}"
    CMD="$CMD --lora_dropout ${LORA_DROPOUT}"
    CMD="$CMD --lora_bias ${LORA_BIAS}"
    CMD="$CMD --lora_target_modules ${LORA_TARGET_MODULES}"
    CMD="$CMD --lora_unfreeze_layers ${LORA_UNFREEZE_LAYERS}"

    # LoRA 恢复
    if [ -n "$LORA_RESUME" ]; then
        CMD="$CMD --lora_resume ${LORA_RESUME}"
    fi

    # 训练后合并
    if [ "$MERGE_LORA" = "true" ]; then
        CMD="$CMD --merge_lora_after_train"
    fi

    # ------ HF-LoRA 参数 ------
    if [ "$USE_HF_LORA" = "true" ]; then
        CMD="$CMD --hf_lora_modules ${HF_LORA_MODULES}"
    fi
fi

# ------ 其他控制参数 ------
if [ "$EVAL_ONLY" = "true" ]; then
    CMD="$CMD --eval"
fi

if [ "$SAVE_BEST_ONLY" = "true" ]; then
    CMD="$CMD --save_best_only"
fi

if [ "$FIND_UNUSED_PARAMS" = "true" ]; then
    CMD="$CMD --find_unused_params"
fi

if [ "$SAVE_LOG" = "true" ]; then
    CMD="$CMD --save_log"
fi

# ============================================================
# 执行
# ============================================================
echo "Executing command:"
echo "$CMD"
echo ""
echo "Log file: ${OUTPUT_DIR}/info.txt"
echo "============================================================"
echo ""

eval $CMD

# ============================================================
# 训练完成后的提示
# ============================================================
EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "  Output: ${OUTPUT_DIR}"
    if [ "$USE_LORA" = "true" ]; then
        if [ -d "${OUTPUT_DIR}/lora_checkpoint_best" ]; then
            echo "  Best checkpoint: ${OUTPUT_DIR}/lora_checkpoint_best/"
            if [ -f "${OUTPUT_DIR}/lora_checkpoint_best/training_state.pth" ]; then
                echo "  (contains: lora_weights.pth, training_state.pth, lora_config.json)"
            fi
        fi
    fi
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "  Check log: ${OUTPUT_DIR}/info.txt"
fi
echo "============================================================"

exit $EXIT_CODE

# USE_LORA=true CUDA_VISIBLE_DEVICES=0,1 bash train_lora.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output_lora/lora_0304_exp2_S
# ```
# # USE_LORA=true \
# # USE_HF_LORA=true \
# # MERGE_LORA=true \
# # CUDA_VISIBLE_DEVICES=2,3 \
# # bash train_lora.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output_lora/lora_0304_exp2_S
# ```
# USE_LORA=true CUDA_VISIBLE_DEVICES=3,1 bash train_lora.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output/no_lora_exp1_D
# USE_LORA=true USE_HF_LORA=true CUDA_VISIBLE_DEVICES=2,3 bash train_lora.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output_lora/lora_0312_exp1VA-HF_D

#!/bin/bash
# ============================================================
# test_dist.sh - 灵活的 LoRA / HF-LoRA 测试评估脚本
# ============================================================
# 用法示例：
#
# 1. 标准 LoRA 评估（已合并权重的模型）：
#    USE_LORA=true bash test_dist.sh 2 configs/cfg_odvg.py datasets/test.json output/eval
#
# 2. HF-LoRA 评估（从检查点加载）：
#    USE_LORA=true USE_HF_LORA=true \
#    LORA_RESUME=output/hf_lora/lora_checkpoint_best \
#    bash test_dist.sh 2 configs/cfg_odvg.py datasets/test.json output/eval
#
# 3. 全量微调模型评估（不用 LoRA）：
#    USE_LORA=false bash test_dist.sh 2 configs/cfg_odvg.py datasets/test.json output/eval
#
# 4. 扩展评估（PR 曲线 + 可视化 + JSON）：
#    USE_LORA=true USE_HF_LORA=true EXTENDED_EVAL=true \
#    LORA_RESUME=output/hf_lora/lora_checkpoint_best \
#    bash test_dist.sh 2 configs/cfg_odvg.py datasets/test.json output/eval
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
    echo "Usage: bash test_dist.sh GPU_NUM CFG DATASETS OUTPUT_DIR"
    echo ""
    echo "Examples:"
    echo "  # 标准 LoRA 评估（已合并模型）"
    echo "  USE_LORA=true bash test_dist.sh 2 configs/cfg_odvg.py datasets/test.json output/eval"
    echo ""
    echo "  # HF-LoRA 评估（从检查点加载）"
    echo "  USE_LORA=true USE_HF_LORA=true LORA_RESUME=output/best bash test_dist.sh 2 ..."
    echo ""
    echo "  # 扩展评估（PR 曲线 + 可视化）"
    echo "  EXTENDED_EVAL=true bash test_dist.sh 2 ..."
    exit 1
fi

# ========================
# 分布式参数
# ========================
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ========================
# 模型路径
# ========================
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/lora_1230_exp1/merged_model.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}

# ========================
# LoRA 基础参数
# ========================
USE_LORA=${USE_LORA:-"true"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-"none"}
LORA_RESUME=${LORA_RESUME:-""}
MERGE_LORA=${MERGE_LORA:-"false"}

# LoRA 目标层
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"qkv proj fc1 fc2 sampling_offsets attention_weights value_proj output_proj v_proj l_proj values_v_proj values_l_proj out_v_proj out_l_proj linear1 linear2"}

# 需要解冻的层
LORA_UNFREEZE_LAYERS=${LORA_UNFREEZE_LAYERS:-"class_embed bbox_embed label_enc"}

# ========================
# HF-LoRA 参数（创新点）
# ========================
USE_HF_LORA=${USE_HF_LORA:-"false"}
HF_LORA_MODULES=${HF_LORA_MODULES:-"qkv fc1 fc2"}

# ========================
# 扩展评估参数
# ========================
# 一键启用所有扩展功能
EXTENDED_EVAL=${EXTENDED_EVAL:-"true"}

# 或者单独控制每个功能
SAVE_JSON=${SAVE_JSON:-"false"}        # 保存 COCO 格式 JSON 结果
PLOT_CURVES=${PLOT_CURVES:-"false"}    # 绘制 PR/P/R/F1 曲线
VISUALIZE=${VISUALIZE:-"false"}        # 可视化 GT vs 预测对比图
SAVE_METRICS=${SAVE_METRICS:-"false"}  # 保存详细指标到文件

# ========================
# 其他控制参数
# ========================
FIND_UNUSED_PARAMS=${FIND_UNUSED_PARAMS:-"true"}
NUM_WORKERS=${NUM_WORKERS:-8}

# ============================================================
# 打印配置信息
# ============================================================

# 确定评估模式名称
if [ "$USE_LORA" = "true" ]; then
    if [ "$USE_HF_LORA" = "true" ]; then
        if [ -n "$LORA_RESUME" ]; then
            EVAL_MODE="HF-LoRA (load from checkpoint)"
        else
            EVAL_MODE="HF-LoRA (inject + evaluate)"
        fi
    else
        if [ -n "$LORA_RESUME" ]; then
            EVAL_MODE="Standard LoRA (load from checkpoint)"
        else
            EVAL_MODE="Standard LoRA (inject + evaluate)"
        fi
    fi
else
    EVAL_MODE="Full Model (No LoRA)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Evaluation Configuration                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Mode: ${EVAL_MODE}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Basic:"
echo "║    GPU_NUM           = $GPU_NUM"
echo "║    CFG               = $CFG"
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
echo "║  Checkpoint:"
echo "║    LORA_RESUME       = $LORA_RESUME"
fi
echo "║    MERGE_LORA        = $MERGE_LORA"
echo "╠══════════════════════════════════════════════════════════╣"
fi
echo "║  Extended Evaluation:"
echo "║    EXTENDED_EVAL     = $EXTENDED_EVAL"
if [ "$EXTENDED_EVAL" != "true" ]; then
echo "║    SAVE_JSON         = $SAVE_JSON"
echo "║    PLOT_CURVES       = $PLOT_CURVES"
echo "║    VISUALIZE         = $VISUALIZE"
echo "║    SAVE_METRICS      = $SAVE_METRICS"
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

# main.py 基础参数（注意 --eval 标志）
CMD="$CMD main.py \
    --output_dir ${OUTPUT_DIR} \
    --eval \
    -c ${CFG} \
    --datasets ${DATASETS} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --num_workers ${NUM_WORKERS} \
    --options text_encoder_type=${TEXT_ENCODER_TYPE}"

# ------ LoRA 参数 ------
if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora"
    CMD="$CMD --lora_r ${LORA_R}"
    CMD="$CMD --lora_alpha ${LORA_ALPHA}"
    CMD="$CMD --lora_dropout ${LORA_DROPOUT}"
    CMD="$CMD --lora_bias ${LORA_BIAS}"
    CMD="$CMD --lora_target_modules ${LORA_TARGET_MODULES}"
    CMD="$CMD --lora_unfreeze_layers ${LORA_UNFREEZE_LAYERS}"

    # LoRA 检查点加载
    if [ -n "$LORA_RESUME" ]; then
        CMD="$CMD --lora_resume ${LORA_RESUME}"
    fi

    # 合并 LoRA 权重用于推理加速
    if [ "$MERGE_LORA" = "true" ]; then
        CMD="$CMD --merge_lora_after_train"
    fi

    # ------ HF-LoRA 参数 ------
    if [ "$USE_HF_LORA" = "true" ]; then
        CMD="$CMD --hf_lora_modules ${HF_LORA_MODULES}"
    fi
fi

# ------ 扩展评估参数 ------
if [ "$EXTENDED_EVAL" = "true" ]; then
    CMD="$CMD --extended_eval"
else
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

# ------ 其他控制参数 ------
if [ "$FIND_UNUSED_PARAMS" = "true" ]; then
    CMD="$CMD --find_unused_params"
fi

# ============================================================
# 执行
# ============================================================
echo "Executing command:"
echo "$CMD"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

eval $CMD

# ============================================================
# 评估完成后的提示
# ============================================================
EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "  Output files:"
    # 列出输出目录中的文件
    if [ -f "${OUTPUT_DIR}/eval.pth" ]; then
        echo "    ✓ eval.pth (COCO evaluation results)"
    fi
    if [ -f "${OUTPUT_DIR}/log.txt" ]; then
        echo "    ✓ log.txt (evaluation log)"
    fi
    if [ -d "${OUTPUT_DIR}/eval" ]; then
        echo "    ✓ eval/ (detailed evaluation data)"
    fi
    if [ -f "${OUTPUT_DIR}/predictions.json" ]; then
        echo "    ✓ predictions.json (COCO format predictions)"
    fi
    if [ -d "${OUTPUT_DIR}/curves" ]; then
        echo "    ✓ curves/ (PR/P/R/F1 curves)"
    fi
    if [ -d "${OUTPUT_DIR}/visualize" ]; then
        echo "    ✓ visualize/ (GT vs prediction comparison)"
    fi
    if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
        echo "    ✓ metrics.json (detailed metrics)"
    fi
else
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
    echo "  Check log: ${OUTPUT_DIR}/info.txt"
fi
echo "============================================================"

exit $EXIT_CODE

# bash test_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_coco_test.json ./eval_output/0304_exp1/sirst
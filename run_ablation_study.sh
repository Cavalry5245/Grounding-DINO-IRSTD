#!/bin/bash
# ============================================================
# run_ablation_study.sh - 红外小目标检测消融实验脚本
# ============================================================
# 用法：
#   bash run_ablation_study.sh [group_number]
# 
# 示例：
#   bash run_ablation_study.sh 0    # 运行 Group 0 (Baseline)
#   bash run_ablation_study.sh 1    # 运行 Group 1 (HF-LoRA 消融)
#   bash run_ablation_study.sh 2    # 运行 Group 2 (提示词库消融)
#   bash run_ablation_study.sh 3    # 运行 Group 3 (组合消融)
#   bash run_ablation_study.sh 4    # 运行 Group 4 (超参数消融)
#   bash run_ablation_study.sh all  # 运行所有实验
# ============================================================

set -e  # 遇到错误立即退出

# ========================
# 基础配置
# ========================
GPU_NUM=2
CFG="./config/cfg_odvg.py"
DATASETS="./config/datasets_mixed_odvg.json"
BASE_OUTPUT_DIR="./ablation_output"

# 模型路径
PRETRAIN_MODEL_PATH="/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth"
TEXT_ENCODER_TYPE="/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"

# 默认 LoRA 参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_BIAS="none"

# ========================
# 实验函数
# ========================

run_experiment() {
    local exp_name=$1
    local output_dir=$2
    shift 2
    local extra_args=$@
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Running: $exp_name"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    USE_LORA=true bash train_lora.sh \
        ${GPU_NUM} \
        ${CFG} \
        ${DATASETS} \
        ${output_dir} \
        LORA_R=${LORA_R} \
        LORA_ALPHA=${LORA_ALPHA} \
        LORA_DROPOUT=${LORA_DROPOUT} \
        LORA_BIAS=${LORA_BIAS} \
        ${extra_args}
}

# ========================
# Group 0: Baseline
# ========================
run_group_0() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Group 0: Baseline Experiments                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    # Exp-0.1: Full Fine-tuning
    run_experiment \
        "Exp-0.1: Full Fine-tuning" \
        "${BASE_OUTPUT_DIR}/group0/exp0_1_full_finetune" \
        USE_LORA=false \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=false
    
    # Exp-0.2: Standard LoRA
    run_experiment \
        "Exp-0.2: Standard LoRA" \
        "${BASE_OUTPUT_DIR}/group0/exp0_2_standard_lora" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=false
}

# ========================
# Group 1: HF-LoRA 消融
# ========================
run_group_1() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Group 1: HF-LoRA Ablation                        ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    # Exp-1.1: HF-LoRA (qkv only)
    run_experiment \
        "Exp-1.1: HF-LoRA (qkv only)" \
        "${BASE_OUTPUT_DIR}/group1/exp1_1_hf_lora_qkv" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv" \
        USE_PROMPT_BANK=false
    
    # Exp-1.2: HF-LoRA (fc1 only)
    run_experiment \
        "Exp-1.2: HF-LoRA (fc1 only)" \
        "${BASE_OUTPUT_DIR}/group1/exp1_2_hf_lora_fc1" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="fc1" \
        USE_PROMPT_BANK=false
    
    # Exp-1.3: HF-LoRA (fc2 only)
    run_experiment \
        "Exp-1.3: HF-LoRA (fc2 only)" \
        "${BASE_OUTPUT_DIR}/group1/exp1_3_hf_lora_fc2" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="fc2" \
        USE_PROMPT_BANK=false
    
    # Exp-1.4: HF-LoRA (qkv+fc1)
    run_experiment \
        "Exp-1.4: HF-LoRA (qkv+fc1)" \
        "${BASE_OUTPUT_DIR}/group1/exp1_4_hf_lora_qkv_fc1" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1" \
        USE_PROMPT_BANK=false
    
    # Exp-1.5: HF-LoRA (Full)
    run_experiment \
        "Exp-1.5: HF-LoRA (Full)" \
        "${BASE_OUTPUT_DIR}/group1/exp1_5_hf_lora_full" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=false
}

# ========================
# Group 2: 提示词库消融
# ========================
run_group_2() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Group 2: Prompt Bank Ablation                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    # Exp-2.1: Prompt Bank (Generic)
    run_experiment \
        "Exp-2.1: Prompt Bank (Generic)" \
        "${BASE_OUTPUT_DIR}/group2/exp2_1_prompt_generic" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic" \
        NUM_SAMPLE_PROMPTS=1
    
    # Exp-2.2: Prompt Bank (Generic+Appearance)
    run_experiment \
        "Exp-2.2: Prompt Bank (Generic+Appearance)" \
        "${BASE_OUTPUT_DIR}/group2/exp2_2_prompt_gen_app" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance" \
        NUM_SAMPLE_PROMPTS=2
    
    # Exp-2.3: Prompt Bank (Generic+Physical)
    run_experiment \
        "Exp-2.3: Prompt Bank (Generic+Physical)" \
        "${BASE_OUTPUT_DIR}/group2/exp2_3_prompt_gen_phy" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic physical" \
        NUM_SAMPLE_PROMPTS=2
    
    # Exp-2.4: Prompt Bank (All 3)
    run_experiment \
        "Exp-2.4: Prompt Bank (All 3)" \
        "${BASE_OUTPUT_DIR}/group2/exp2_4_prompt_3cat" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3
    
    # Exp-2.5: Prompt Bank (All 5)
    run_experiment \
        "Exp-2.5: Prompt Bank (All 5)" \
        "${BASE_OUTPUT_DIR}/group2/exp2_5_prompt_5cat" \
        USE_LORA=true \
        USE_HF_LORA=false \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical contextual size_aware" \
        NUM_SAMPLE_PROMPTS=5
}

# ========================
# Group 3: 组合消融
# ========================
run_group_3() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Group 3: Combination Ablation                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    # Exp-3.1: HF-LoRA + Prompt Bank (3类)
    run_experiment \
        "Exp-3.1: HF-LoRA + Prompt Bank (3类)" \
        "${BASE_OUTPUT_DIR}/group3/exp3_1_hf_prompt_3cat" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3
    
    # Exp-3.2: HF-LoRA + Prompt Bank (5类)
    run_experiment \
        "Exp-3.2: HF-LoRA + Prompt Bank (5类)" \
        "${BASE_OUTPUT_DIR}/group3/exp3_2_hf_prompt_5cat" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical contextual size_aware" \
        NUM_SAMPLE_PROMPTS=5
}

# ========================
# Group 4: 超参数消融
# ========================
run_group_4() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Group 4: Hyperparameter Ablation                 ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    
    # Exp-4.1: HF-LoRA r=8
    run_experiment \
        "Exp-4.1: HF-LoRA r=8" \
        "${BASE_OUTPUT_DIR}/group4/exp4_1_r8" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3 \
        LORA_R=8 \
        LORA_ALPHA=16
    
    # Exp-4.2: HF-LoRA r=32
    run_experiment \
        "Exp-4.2: HF-LoRA r=32" \
        "${BASE_OUTPUT_DIR}/group4/exp4_2_r32" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3 \
        LORA_R=32 \
        LORA_ALPHA=64
    
    # Exp-4.3: HF-LoRA alpha=16
    run_experiment \
        "Exp-4.3: HF-LoRA alpha=16" \
        "${BASE_OUTPUT_DIR}/group4/exp4_3_alpha16" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3 \
        LORA_R=16 \
        LORA_ALPHA=16
    
    # Exp-4.4: HF-LoRA alpha=64
    run_experiment \
        "Exp-4.4: HF-LoRA alpha=64" \
        "${BASE_OUTPUT_DIR}/group4/exp4_4_alpha64" \
        USE_LORA=true \
        USE_HF_LORA=true \
        HF_LORA_MODULES="qkv fc1 fc2" \
        USE_PROMPT_BANK=true \
        PROMPT_CATEGORIES="generic appearance physical" \
        NUM_SAMPLE_PROMPTS=3 \
        LORA_R=16 \
        LORA_ALPHA=64
}

# ========================
# 主程序
# ========================

GROUP_NUM=$1

if [ -z "$GROUP_NUM" ]; then
    echo "用法: bash run_ablation_study.sh [group_number|all]"
    echo ""
    echo "可选的组："
    echo "  0  - Baseline (Full Fine-tuning, Standard LoRA)"
    echo "  1  - HF-LoRA 消融 (5个实验)"
    echo "  2  - 提示词库消融 (5个实验)"
    echo "  3  - 组合消融 (2个实验)"
    echo "  4  - 超参数消融 (4个实验)"
    echo "  all - 运行所有实验"
    exit 1
fi

case $GROUP_NUM in
    0)
        run_group_0
        ;;
    1)
        run_group_1
        ;;
    2)
        run_group_2
        ;;
    3)
        run_group_3
        ;;
    4)
        run_group_4
        ;;
    all)
        run_group_0
        run_group_1
        run_group_2
        run_group_3
        run_group_4
        ;;
    *)
        echo "错误: 无效的组号 '$GROUP_NUM'"
        echo "请使用 0, 1, 2, 3, 4 或 all"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          消融实验完成！                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "结果保存在: ${BASE_OUTPUT_DIR}/"
echo ""

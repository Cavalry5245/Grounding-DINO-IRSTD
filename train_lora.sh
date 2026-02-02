#!/bin/bash
# Quick LoRA fine-tuning script

export TOKENIZERS_PARALLELISM=false
export USE_LORA="true"

GPU_NUM=${1:-4}
CFG=${2:-"/media/sisu/X/hc/projects/Open-GroundingDino/config/cfg_odvg.py"}
DATASETS=${3:-"/media/sisu/X/hc/projects/Open-GroundingDino/config/datasets_mixed_odvg.json"}
OUTPUT_DIR=${4:-"/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/lora_exp1"}

# LoRA defaults
export LORA_R=${LORA_R:-8}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export LORA_BIAS=${LORA_BIAS:-"none"}
export MERGE_LORA=${MERGE_LORA:-"true"}

echo "=============================================="
echo "LoRA Fine-tuning Configuration"
echo "=============================================="
echo "  LoRA rank (r): $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
echo "  Merge after training: $MERGE_LORA"
echo "=============================================="

bash train_dist.sh "$GPU_NUM" "$CFG" "$DATASETS" "$OUTPUT_DIR"

# CUDA_VISIBLE_DEVICES=2,3 bash train_lora.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output_lora/lora_nolabel/0129_exp1_K
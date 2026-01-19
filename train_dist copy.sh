#!/bin/bash
export TOKENIZERS_PARALLELISM=false
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased"}
echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
PORT = $PORT
MASTER_ADDR = $MASTER_ADDR
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
"

# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

torchrun --nproc_per_node="${GPU_NUM}" --master_port="${PORT}" main.py \
        --output_dir "${OUTPUT_DIR}" \
        -c "${CFG}" \
        --datasets "${DATASETS}"  \
        --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
        --options text_encoder_type="$TEXT_ENCODER_TYPE"

# bash train_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output/
# CUDA_VISIBLE_DEVICES=2,3 bash train_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output/sirst
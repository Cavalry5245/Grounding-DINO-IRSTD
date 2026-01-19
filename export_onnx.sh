#!/bin/bash
# GroundingDINO ONNX Export Script

# Default values
CONFIG=${1:-"config/cfg_odvg.py"}
CHECKPOINT=${2:-"weights/groundingdino_swint_ogc.pth"}
OUTPUT=${3:-"models/groundingdino.onnx"}
IMAGE_HEIGHT=${4:-800}
IMAGE_WIDTH=${5:-1200}

echo "=============================================="
echo "GroundingDINO ONNX Export"
echo "=============================================="
echo "Config:        $CONFIG"
echo "Checkpoint:    $CHECKPOINT"
echo "Output:        $OUTPUT"
echo "Image size:    ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "=============================================="

# Create output directory if needed
mkdir -p $(dirname "$OUTPUT")

# Run export
python tools/export_onnx.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --image-height "$IMAGE_HEIGHT" \
    --image-width "$IMAGE_WIDTH" \
    --opset 14

echo ""
echo "Export completed!"
echo "Model saved to: $OUTPUT"
#!/usr/bin/env python3
"""
GroundingDINO ONNX Inference Script
用于测试导出的 ONNX 模型
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.utils import get_phrases_from_posmap


def load_image(image_path, target_size=800, max_size=1333):
    """
    Load and preprocess image for inference
    """
    image_pil = Image.open(image_path).convert("RGB")
    original_size = image_pil.size

    # Simple resize to target size
    image_pil_resized = image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image_pil_resized)).permute(2, 0, 1).float() / 255.0

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_pil, image_tensor, original_size


def prepare_text_inputs(caption, tokenizer, max_text_len=256, device='cpu'):
    """
    Prepare text inputs for ONNX model
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    # Tokenize
    tokenized = tokenizer(caption, padding="max_length", max_length=max_text_len,
                          truncation=True, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    token_type_ids = tokenized.get("token_type_ids", torch.zeros_like(input_ids))

    # Create position ids
    position_ids = torch.arange(max_text_len, dtype=torch.long).unsqueeze(0)

    # Create text self attention masks (all ones for simplicity)
    text_self_attention_masks = torch.ones(max_text_len, max_text_len, dtype=torch.bool).unsqueeze(0)

    return {
        'input_ids': input_ids.numpy(),
        'attention_mask': attention_mask.numpy(),
        'token_type_ids': token_type_ids.numpy(),
        'position_ids': position_ids.numpy(),
        'text_self_attention_masks': text_self_attention_masks.numpy().astype(np.bool_)
    }


def run_onnx_inference(onnx_session, image_tensor, text_inputs):
    """
    Run inference with ONNX model
    """
    # Prepare inputs
    inputs = {
        'images': image_tensor.numpy(),
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'token_type_ids': text_inputs['token_type_ids'],
        'position_ids': text_inputs['position_ids'],
        'text_self_attention_masks': text_inputs['text_self_attention_masks']
    }

    # Run inference
    outputs = onnx_session.run(None, inputs)

    pred_logits = torch.from_numpy(outputs[0])
    pred_boxes = torch.from_numpy(outputs[1])

    return pred_logits, pred_boxes


def filter_predictions(pred_logits, pred_boxes, caption, tokenizer, box_threshold=0.3, text_threshold=0.25):
    """
    Filter predictions based on thresholds
    """
    pred_logits = pred_logits.sigmoid()[0]  # (num_queries, text_dim)
    pred_boxes = pred_boxes[0]  # (num_queries, 4)

    # Get max score for each query
    max_scores = pred_logits.max(dim=1)[0]

    # Filter by box threshold
    keep = max_scores > box_threshold
    pred_logits = pred_logits[keep]
    pred_boxes = pred_boxes[keep]

    if pred_logits.shape[0] == 0:
        return pred_boxes, []

    # Get phrases
    tokenized = tokenizer(caption, return_tensors="pt")
    pred_phrases = []
    for logit, box in zip(pred_logits, pred_boxes):
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        score = logit.max().item()
        pred_phrases.append(f"{phrase} ({score:.3f})")

    return pred_boxes, pred_phrases


def visualize_predictions(image_pil, boxes, phrases, original_size):
    """
    Visualize predictions on image
    """
    draw = ImageDraw.Draw(image_pil)

    H, W = original_size[1], original_size[0]

    for box, phrase in zip(boxes, phrases):
        # Convert from normalized to absolute coordinates
        box_abs = box * torch.tensor([W, H, W, H])
        x_center, y_center, w, h = box_abs

        # Convert from cx_cy_w_h to x1_y1_x2_y2
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Draw label
        draw.text((x1, y1 - 20), phrase, fill='red')

    return image_pil


def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO ONNX Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to ONNX model file")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("-t", "--text", type=str, required=True,
                        help="Text prompt for detection")
    parser.add_argument("-o", "--output", type=str, default="output_onnx.jpg",
                        help="Path to save output image")
    parser.add_argument("--box-threshold", type=float, default=0.3,
                        help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Text token threshold")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="Tokenizer to use for text encoding")
    parser.add_argument("--max-text-len", type=int, default=256,
                        help="Maximum text length")
    parser.add_argument("--target-size", type=int, default=800,
                        help="Target image size for preprocessing")

    args = parser.parse_args()

    print("=" * 60)
    print("GroundingDINO ONNX Inference")
    print("=" * 60)

    # Load ONNX model
    print(f"\n[1/4] Loading ONNX model from: {args.model}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    onnx_session = ort.InferenceSession(args.model, session_options, providers=providers)
    print(f"   Providers: {onnx_session.get_providers()}")

    # Load tokenizer
    print(f"[2/4] Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load and preprocess image
    print(f"[3/4] Loading image: {args.image}")
    image_pil, image_tensor, original_size = load_image(args.image, args.target_size)
    print(f"   Original size: {original_size}")
    print(f"   Input size: {image_tensor.shape}")

    # Prepare text inputs
    text_inputs = prepare_text_inputs(args.text, tokenizer, args.max_text_len)

    # Run inference
    print(f"[4/4] Running inference...")
    pred_logits, pred_boxes = run_onnx_inference(onnx_session, image_tensor, text_inputs)

    # Filter predictions
    pred_boxes, pred_phrases = filter_predictions(
        pred_logits, pred_boxes, args.text, tokenizer,
        args.box_threshold, args.text_threshold
    )

    print(f"\nFound {len(pred_boxes)} detections:")
    for box, phrase in zip(pred_boxes, pred_phrases):
        print(f"  {phrase}")

    # Visualize
    if len(pred_boxes) > 0:
        output_image = visualize_predictions(image_pil, pred_boxes, pred_phrases, original_size)
        output_image.save(args.output)
        print(f"\nOutput saved to: {args.output}")
    else:
        print("\nNo detections found to visualize.")

    print("=" * 60)


if __name__ == "__main__":
    main()
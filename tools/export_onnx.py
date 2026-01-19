#!/usr/bin/env python3
"""
GroundingDINO ONNX Export Tool
支持动态文本输入和图像尺寸
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import NestedTensor
from groundingdino.util import box_ops


class GroundingDINOONNXWrapper(nn.Module):
    """
    Wrapper for ONNX export that handles NestedTensor and text processing
    """
    def __init__(self, model, tokenizer, special_tokens, max_text_len=256):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_text_len = max_text_len

    def forward(self, images, input_ids, attention_mask, token_type_ids, position_ids, text_self_attention_masks):
        """
        Args:
            images: (B, 3, H, W) - normalized image tensor
            input_ids: (B, max_text_len) - tokenized text input
            attention_mask: (B, max_text_len) - attention mask
            token_type_ids: (B, max_text_len) - token type ids
            position_ids: (B, max_text_len) - position ids
            text_self_attention_masks: (B, max_text_len, max_text_len) - text self attention masks
        """
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]

        # Create NestedTensor
        masks = torch.zeros((batch_size, height, width), dtype=torch.bool, device=images.device)
        samples = NestedTensor(images, masks)

        # Build text dict
        text_dict = {
            "encoded_text": self.model.bert(
                input_ids=input_ids,
                attention_mask=text_self_attention_masks,
                token_type_ids=token_type_ids,
                position_ids=position_ids
            )["last_hidden_state"],
            "text_token_mask": attention_mask.bool(),
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

        # Truncate to max_text_len if needed
        if text_dict["encoded_text"].shape[1] > self.max_text_len:
            text_dict["encoded_text"] = text_dict["encoded_text"][:, :self.max_text_len, :]
            text_dict["text_token_mask"] = text_dict["text_token_mask"][:, :self.max_text_len]
            text_dict["position_ids"] = text_dict["position_ids"][:, :self.max_text_len]
            text_dict["text_self_attention_masks"] = text_dict["text_self_attention_masks"][:, :self.max_text_len, :self.max_text_len]

        # Map to hidden dim
        text_dict["encoded_text"] = self.model.feat_map(text_dict["encoded_text"])

        # Forward through backbone
        features, poss = self.model.backbone(samples)

        # Prepare input projections
        srcs = []
        masks_list = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks_list.append(mask)

        # Handle multi-scale features
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks_list.append(mask)
                poss.append(pos_l)

        # Transformer forward
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.model.transformer(
            srcs, masks_list, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # Compute outputs
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + box_ops.inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)
            ]
        )

        # Return final outputs
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1]
        }

        return out["pred_logits"], out["pred_boxes"]


def export_to_onnx(config_path, checkpoint_path, output_path,
                   image_size=(800, 1200), max_text_len=256,
                   opset_version=14, use_half=False):
    """
    Export GroundingDINO model to ONNX format

    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        image_size: (height, width) for dummy input
        max_text_len: Maximum text length
        opset_version: ONNX opset version
        use_half: Use half precision (FP16)
    """
    print("=" * 60)
    print("GroundingDINO ONNX Export")
    print("=" * 60)

    # Load config
    print(f"\n[1/5] Loading config from: {config_path}")
    args = SLConfig.fromfile(config_path)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    print(f"[2/5] Building model...")
    model, _, _ = build_model(args)

    # Load checkpoint
    print(f"[3/5] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = clean_state_dict(checkpoint['model'])
        else:
            state_dict = clean_state_dict(checkpoint)
    else:
        state_dict = clean_state_dict(checkpoint)

    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"   Missing keys: {len(load_result.missing_keys)}")
    print(f"   Unexpected keys: {len(load_result.unexpected_keys)}")

    # Set to eval mode
    model.eval()
    if use_half:
        model.half()
        print(f"   Using FP16 precision")

    # Create wrapper
    print(f"[4/5] Creating ONNX wrapper...")
    wrapper = GroundingDINOONNXWrapper(
        model,
        model.tokenizer,
        model.specical_tokens,
        max_text_len=max_text_len
    )

    # Create dummy inputs
    batch_size = 1
    height, width = image_size
    device = next(model.parameters()).device

    dummy_images = torch.randn(batch_size, 3, height, width, device=device)
    if use_half:
        dummy_images = dummy_images.half()

    # Create dummy text inputs
    dummy_input_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(batch_size, max_text_len, dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    dummy_position_ids = torch.arange(max_text_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    dummy_text_self_attention_masks = torch.ones(batch_size, max_text_len, max_text_len, dtype=torch.bool, device=device)

    # Prepare dynamic axes
    dynamic_axes = {
        'images': {0: 'batch_size', 2: 'height', 3: 'width'},
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'token_type_ids': {0: 'batch_size'},
        'position_ids': {0: 'batch_size'},
        'text_self_attention_masks': {0: 'batch_size', 1: 'text_len', 2: 'text_len'},
        'pred_logits': {0: 'batch_size', 1: 'num_queries'},
        'pred_boxes': {0: 'batch_size', 1: 'num_queries'}
    }

    # Export to ONNX
    print(f"[5/5] Exporting to ONNX...")
    print(f"   Output path: {output_path}")
    print(f"   Opset version: {opset_version}")
    print(f"   Image size: {image_size}")
    print(f"   Max text length: {max_text_len}")

    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_input_ids, dummy_attention_mask,
         dummy_token_type_ids, dummy_position_ids, dummy_text_self_attention_masks),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images', 'input_ids', 'attention_mask',
                    'token_type_ids', 'position_ids', 'text_self_attention_masks'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f"\n" + "=" * 60)
    print(f"ONNX export completed successfully!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("\n[ONNX Model Verification] PASSED")
    except ImportError:
        print("\n[Warning] onnx package not installed. Skipping verification.")
    except Exception as e:
        print(f"\n[Error] ONNX model verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export GroundingDINO model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to model config file")
    parser.add_argument("-p", "--checkpoint", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to save ONNX model")

    # Optional arguments
    parser.add_argument("--image-height", type=int, default=800,
                        help="Dummy input image height")
    parser.add_argument("--image-width", type=int, default=1200,
                        help="Dummy input image width")
    parser.add_argument("--max-text-len", type=int, default=256,
                        help="Maximum text length for tokenization")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--half", action="store_true",
                        help="Export with FP16 precision")

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Export
    export_to_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=(args.image_height, args.image_width),
        max_text_len=args.max_text_len,
        opset_version=args.opset,
        use_half=args.half
    )


if __name__ == "__main__":
    main()
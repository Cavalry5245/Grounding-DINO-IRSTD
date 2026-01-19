#!/usr/bin/env python
"""
Merge LoRA weights into base model using peft.
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peft import PeftModel
from util.slconfig import SLConfig


def get_args():
    parser = argparse.ArgumentParser('Merge LoRA weights')
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to base model checkpoint')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to LoRA checkpoint directory (peft format)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save merged model')
    return parser.parse_args()


def main():
    args = get_args()
    
    print(f"Loading config from {args.config_file}")
    cfg = SLConfig.fromfile(args.config_file)
    
    # Build model
    from models.registry import MODULE_BUILD_FUNCS
    
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(model_args, k, v)
    
    print("Building base model...")
    build_func = MODULE_BUILD_FUNCS.get(model_args.modelname)
    model, _, _ = build_func(model_args)
    
    # Load base weights
    print(f"Loading base model weights from {args.base_model_path}")
    checkpoint = torch.load(args.base_model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=False)
    
    # Load and merge LoRA
    print(f"Loading LoRA from {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # Save
    print(f"Saving merged model to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({'model': model.state_dict()}, args.output_path)
    
    print("Done!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == '__main__':
    main()
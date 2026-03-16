from gdino_eval_core import evaluate_gdino_on_coco
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 你常改的参数都集中放这里
# ============================================================
CFG = {
    # ==================== dataset ====================
    "images_dir": "data/IRSTD-1k/images/test",
    "coco_gt": "data/IRSTD-1k/annotations/test.json",

    # ==================== model ====================
    "config": "config/cfg_odvg.py",
    "text_prompt": "Infrared small target",

    # ==================== 权重加载模式 ====================
    # 模式 A：使用合并后的权重（不需要 LoRA）
    #   "weights": "output/merged_model.pth",
    #   "use_lora": False,
    #
    # 模式 B：使用未合并的 LoRA 权重（预训练 + LoRA 检查点）
    #   "weights": "weights/groundingdino_swint_ogc.pth",  # 预训练基础模型
    #   "use_lora": True,
    #   "lora_checkpoint": "output/hf_lora/lora_checkpoint_best",
    # ================================================

    # --- 选择你要用的模式，注释掉另一个 ---

    # # # 模式 A：合并后的权重
    # "weights": "/media/sisu/X/hc/projects/Open-GroundingDino/training_output/no_lora_exp1_D/checkpoint_best.pth",
    # "use_lora": False,

    # 模式 B：未合并的 LoRA 权重
    "weights": "/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth",
    "use_lora": True,
    "lora_checkpoint": "/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/exp1_hf_prompt_combined/lora_checkpoint_best",

    # ==================== LoRA 配置 ====================
    # 如果 lora_checkpoint 目录中有 lora_config.json，以下参数会被自动覆盖
    # 如果没有，则使用以下默认值
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    # "lora_target_modules": [
    #     "qkv", "proj", "fc1", "fc2",
    #     "sampling_offsets", "attention_weights",
    #     "value_proj", "output_proj",
    #     "v_proj", "l_proj",
    #     "values_v_proj", "values_l_proj",
    #     "out_v_proj", "out_l_proj",
    #     "linear1", "linear2"
    # ],

    # HF-LoRA 配置（如果训练时用了 HF-LoRA，这里也要开启）
    # 设为空列表 [] 表示全部使用标准 LoRA
    "hf_lora_modules": ["qkv", "fc1", "fc2"],
    # "hf_lora_modules": [],  # 标准 LoRA 时用这行

    # ==================== thresholds ====================
    "box_threshold": 0.001,
    "text_threshold": 0.85,
    "nms_threshold": 0.5,

    # ==================== optional filter ====================
    "min_area": 0.0,
    "max_area": 1e18,

    # ==================== metrics behavior ====================
    "v5_metric": False,
    "force_single_class": False,

    # ==================== output ====================
    "output_dir": "eval_output/test/IRSTD-1k3",
    "save_pred_json": True,
    "save_pr_curve_data": True,
}


def _json_safe(obj):
    """确保 Path / numpy 类型可序列化"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics(output_dir: str, cfg: dict, metrics: dict):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cfg": cfg,
        "metrics": metrics
    }

    # JSON（机器可读）
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8"
    )

    # TXT（人可读）
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # 打印加载模式
    if CFG.get("use_lora", False):
        hf_modules = CFG.get("hf_lora_modules", [])
        lora_type = "HF-LoRA" if hf_modules else "Standard LoRA"
        print("=" * 60)
        print(f"Evaluation Mode: {lora_type} (未合并权重)")
        print(f"  Base model:      {CFG['weights']}")
        print(f"  LoRA checkpoint: {CFG['lora_checkpoint']}")
        print(f"  LoRA r={CFG.get('lora_r', 8)}, alpha={CFG.get('lora_alpha', 16)}")
        if hf_modules:
            print(f"  HF-LoRA modules: {hf_modules}")
        print("=" * 60)
    else:
        print("=" * 60)
        print(f"Evaluation Mode: Merged model (合并后权重)")
        print(f"  Weights: {CFG['weights']}")
        print("=" * 60)

    metrics = evaluate_gdino_on_coco(CFG)
    save_metrics(CFG["output_dir"], CFG, metrics)

    n_img = metrics.get("_num_images_coco", None)
    n_miss = metrics.get("_missing_images", None)

    # 计数项
    GT = metrics.get("GT", None)
    TP = metrics.get("TP", None)
    FP = metrics.get("FP_box", None)
    FN = metrics.get("FN", None)
    best_thres = metrics.get("best_thres", None)

    header = []
    if n_img is not None:
        header.append(f"Images(COCO): {n_img}")
    if n_miss is not None:
        header.append(f"Missing: {n_miss}")
    if GT is not None:
        header.append(f"GT: {GT}")
    print(" | ".join(header))

    if best_thres is not None and TP is not None:
        print(f"Best-F1 threshold: {best_thres:.4f} | TP: {TP} | FP_box: {FP} | FN: {FN}")

    print(
        f"P: {metrics['P']:.4f} | R: {metrics['R']:.4f} | F1: {metrics.get('F1', float('nan')):.4f} | "
        f"PD: {metrics.get('PD', float('nan')):.4f} | FA: {metrics.get('FA', float('nan')):.4f}"
    )
    print(
        f"mAP@0.5: {metrics.get('mAP@0.5', float('nan')):.4f} | "
        f"mAP@0.5:0.95: {metrics.get('mAP@0.5:0.95', float('nan')):.4f}"
    )

    print(f"Output: {CFG['output_dir']}")

#  CUDA_VISIBLE_DEVICES=1 python -u "/media/sisu/X/hc/projects/Open-GroundingDino/eval/run_eval.py"
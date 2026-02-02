from gdino_eval_core import evaluate_gdino_on_coco
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 你常改的参数都集中放这里
CFG = {
    # dataset
    "images_dir": "data/sirst/images/test",
    "coco_gt": "data/sirst/annotations/test.json",

    # model
    "config": "config/cfg_odvg.py",
    "weights": "/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/lora_0121_exp2/merged_model.pth",
    "text_prompt": "Infrared small target",

    # thresholds (对齐你确认的口径)
    "box_threshold": 0.001,
    "text_threshold": 0.85,
    "nms_threshold": 0.5,

    # optional filter (如需完全对齐 YOLO baseline，建议设得很宽或关闭)
    "min_area": 0.0,
    "max_area": 1e18,

    # metrics behavior
    "v5_metric": False,          # 要和对方 repo 一致（默认 False）
    "force_single_class": False, # 若你的 COCO 有多类但你只做单类prompt，可设 True

    # output
    "output_dir": "eval_output/test/SIRST_withHR",
    "save_pred_json": True,
    "save_pr_curve_data": True,
}

def _json_safe(obj):
    # 确保 Path / numpy 类型可序列化
    if isinstance(obj, Path):
        return str(obj)
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
    metrics = evaluate_gdino_on_coco(CFG)
    save_metrics(CFG["output_dir"], CFG, metrics)

    n_img = metrics.get("_num_images_coco", None)
    n_miss = metrics.get("_missing_images", None)

    # 计数项（如果你按方案加入了）
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

    # 主要指标（你要求的：F1、AP、PD、FA + 原有 mAP）
    print(
        f"P: {metrics['P']:.4f} | R: {metrics['R']:.4f} | F1: {metrics.get('F1', float('nan')):.4f} | "
        f"PD: {metrics.get('PD', float('nan')):.4f} | FA: {metrics.get('FA', float('nan')):.4f}"
    )
    print(
        # f"AP50: {metrics.get('AP50', float('nan')):.4f} | AP50-95: {metrics.get('AP50_95', float('nan')):.4f} | "
        f"mAP@0.5: {metrics.get('mAP@0.5', float('nan')):.4f} | mAP@0.5:0.95: {metrics.get('mAP@0.5:0.95', float('nan')):.4f}"
    )

    print(f"Output: {CFG['output_dir']}")
# CUDA_VISIBLE_DEVICES=3 python -u "/media/sisu/X/hc/projects/Open-GroundingDino/eval/run_eval.py"
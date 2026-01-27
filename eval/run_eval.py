from gdino_eval_core import evaluate_gdino_on_coco

# 你常改的参数都集中放这里
CFG = {
    # dataset
    "images_dir": "data/NUDT-SIRST/images/test",
    "coco_gt": "data/NUDT-SIRST/annotations/test.json",

    # model
    "config": "config/cfg_odvg.py",
    "weights": "/media/sisu/X/hc/projects/Open-GroundingDino/training_output_lora/lora_0125_exp2_D/merged_model.pth",
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
    "output_dir": "eval_output/0125_exp2/NUDT-SIRST",
    "save_pred_json": True,
}

if __name__ == "__main__":
    metrics = evaluate_gdino_on_coco(CFG)
    print(f"Images(COCO): {metrics['_num_images_coco']} | Missing: {metrics['_missing_images']}")
    print(f"P: {metrics['P']:.4f} | R: {metrics['R']:.4f} | "
          f"mAP@0.5: {metrics['mAP@0.5']:.4f} | mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"Output: {CFG['output_dir']}")
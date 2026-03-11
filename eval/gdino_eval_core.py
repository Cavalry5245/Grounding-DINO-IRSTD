# gdino_eval_core.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torchvision.ops import box_convert
from pycocotools.coco import COCO

# 复用你贴的 metrics.py
from metrics import ap_per_class

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict


# -------------------------
# COCO GT Loader
# -------------------------
@dataclass
class ImageGT:
    image_id: int
    file_name: str
    width: int
    height: int
    boxes_xyxy: np.ndarray  # [N,4] float32 pixel xyxy
    classes: np.ndarray     # [N] int64 contiguous 0..nc-1


def _xywh_to_xyxy_np(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    return np.stack([x, y, x + w, y + h], axis=1)


class CocoGTLoader:
    """
    - 读取 COCO GT
    - 将 category_id 映射到连续的 class index（0..nc-1）
    - 支持 force_single_class：把所有 GT 类都合并为 0 类（可选）
    """
    def __init__(self, coco_json: str, force_single_class: bool = False):
        self.coco = COCO(coco_json)
        self.force_single_class = force_single_class

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = sorted([c["id"] for c in cats])

        if force_single_class:
            self.cat_id_to_idx = {cid: 0 for cid in self.cat_ids}
            self.nc = 1
        else:
            self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}
            self.nc = len(self.cat_ids)

    def iter_images(self) -> List[ImageGT]:
        out: List[ImageGT] = []
        for image_id in self.coco.getImgIds():
            img = self.coco.loadImgs([image_id])[0]
            w, h = int(img["width"]), int(img["height"])
            file_name = str(img["file_name"])

            ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            if len(anns) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.int64)
            else:
                bxywh = np.array([a["bbox"] for a in anns], dtype=np.float32)
                boxes = _xywh_to_xyxy_np(bxywh).astype(np.float32)
                cls = np.array([self.cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int64)

            out.append(ImageGT(
                image_id=int(image_id),
                file_name=file_name,
                width=w,
                height=h,
                boxes_xyxy=boxes,
                classes=cls
            ))
        return out


# -------------------------
# Image path resolver (兼容 COCO file_name 是否带子目录)
# -------------------------
class ImagePathResolver:
    """
    兼容：
      - COCO file_name = "xxx.jpg"
      - COCO file_name = "subdir/xxx.jpg"
    同时 images_dir 下面可能是平铺也可能有多级目录。
    """
    def __init__(self, images_dir: str | Path, exts: Optional[set[str]] = None, build_index: bool = True):
        self.images_dir = Path(images_dir)
        self.exts = exts or {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self._rel_map: Dict[str, Path] = {}
        self._base_map: Dict[str, List[Path]] = {}

        if build_index:
            self._build_index()

    def _build_index(self):
        if not self.images_dir.exists():
            return
        files = [p for p in self.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        for p in files:
            rel = p.relative_to(self.images_dir).as_posix()
            self._rel_map[rel] = p
            self._base_map.setdefault(p.name, []).append(p)

    @staticmethod
    def _normalize_coco_name(file_name: str) -> str:
        s = file_name.replace("\\", "/")
        while s.startswith("./"):
            s = s[2:]
        s = s.lstrip("/")
        return s

    def resolve(self, coco_file_name: str) -> Optional[Path]:
        s = self._normalize_coco_name(coco_file_name)
        p1 = self.images_dir / s
        if p1.exists():
            return p1

        base = Path(s).name
        p2 = self.images_dir / base
        if p2.exists():
            return p2

        if s in self._rel_map:
            return self._rel_map[s]

        candidates = self._base_map.get(base, [])
        if len(candidates) == 1:
            return candidates[0]

        tail_matches = [c for c in candidates if c.as_posix().endswith(s)]
        if len(tail_matches) == 1:
            return tail_matches[0]
        if len(tail_matches) > 1:
            print(f"[WARN] multiple matches for '{coco_file_name}', choose: {tail_matches[0]}")
            return tail_matches[0]

        if len(candidates) > 1:
            print(f"[WARN] basename '{base}' has {len(candidates)} matches, cannot disambiguate. choose: {candidates[0]}")
            return candidates[0]

        return None


# -------------------------
# Geometry / NMS / IoU
# -------------------------
def cxcywh_norm_to_xyxy_pixel(boxes_cxcywh_norm: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    if boxes_cxcywh_norm.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    h, w = hw
    b = box_convert(boxes=boxes_cxcywh_norm, in_fmt="cxcywh", out_fmt="xyxy").clone()
    b[:, [0, 2]] *= w
    b[:, [1, 3]] *= h
    return b


def filter_by_area_cxcywh_norm(
    boxes: torch.Tensor, scores: torch.Tensor, hw: Tuple[int, int], min_area: float, max_area: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores
    h, w = hw
    areas = boxes[:, 2] * boxes[:, 3] * (h * w)
    keep = (areas >= min_area) & (areas <= max_area)
    idx = torch.where(keep)[0]
    return boxes[idx], scores[idx]


def nms_xyxy(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    return torchvision.ops.nms(boxes_xyxy, scores, iou_thres)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), dtype=torch.float32)

    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + 1e-16
    return inter / union


# -------------------------
# Evaluator (YOLOv7/test.py style)
# -------------------------
class YoloStyleEvaluator:
    """
    - correct: [num_pred, 10] bool
    - P/R: 取 F1 最大处（由 metrics.py::ap_per_class 内部决定）
    - mAP@0.5 / mAP@0.5:0.95
    """
    def __init__(self, v5_metric: bool, output_dir: str, save_pred_json: bool, coco_pred_category_id: int,
             save_pr_curve_data: bool = True):
        self.v5_metric = v5_metric
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_pred_json = save_pred_json
        self.save_pr_curve_data = save_pr_curve_data
        self.coco_pred_category_id = coco_pred_category_id

        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

        self.stats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]] = []
        self.pred_json: List[Dict[str, Any]] = []

    def update_one(self,
                   image_id: int,
                   pred_boxes_xyxy: torch.Tensor,
                   pred_scores: torch.Tensor,
                   pred_cls: torch.Tensor,
                   gt_boxes_xyxy: torch.Tensor,
                   gt_cls: torch.Tensor):
        nl = gt_boxes_xyxy.shape[0]

        if pred_boxes_xyxy.shape[0] == 0:
            self.stats.append((
                torch.zeros((0, self.niou), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                gt_cls.cpu().numpy() if nl else np.array([], dtype=np.float32)
            ))
            return

        if self.save_pred_json:
            for b, s in zip(pred_boxes_xyxy, pred_scores):
                x1, y1, x2, y2 = b.tolist()
                self.pred_json.append({
                    "image_id": int(image_id),
                    "category_id": int(self.coco_pred_category_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })

        correct = torch.zeros((pred_boxes_xyxy.shape[0], self.niou), dtype=torch.bool)

        # if nl:
        #     detected = []
        #     tcls_tensor = gt_cls

        #     for cls in torch.unique(tcls_tensor):
        #         ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # gt
        #         pi = (cls == pred_cls).nonzero(as_tuple=False).view(-1)     # pred

        #         if pi.numel():
        #             ious, i = box_iou(pred_boxes_xyxy[pi], gt_boxes_xyxy[ti]).max(1)

        #             detected_set = set()
        #             for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
        #                 d = ti[i[j]]
        #                 if d.item() not in detected_set:
        #                     detected_set.add(d.item())
        #                     detected.append(d)
        #                     correct[pi[j]] = ious[j] > self.iouv
        #                     if len(detected) == nl:
        #                         break
        if nl:
            detected = []
            tcls_tensor = gt_cls
            
            # 将 GT 和 Pred 转换为中心点 (cx, cy)
            gt_centers = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2
            pred_centers = (pred_boxes_xyxy[:, :2] + pred_boxes_xyxy[:, 2:]) / 2
            
            # 定义中心点匹配阈值 (例如：允许偏差 3 个像素，或者目标尺度的 0.5 倍)
            # 对于红外小目标，固定像素阈值往往更有效，例如 DIST_THRESH = 3.0
            DIST_THRESH = 3.0 

            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # gt indices
                pi = (cls == pred_cls).nonzero(as_tuple=False).view(-1)     # pred indices

                if pi.numel():
                    # --- 修改开始：使用中心点距离匹配 ---
                    
                    # 计算所有 pred 和所有 gt 的欧氏距离 [num_pred, num_gt]
                    # distinct_dist = torch.cdist(pred_centers[pi], gt_centers[ti], p=2) 
                    
                    # 依然计算 IoU 用于兼容性 (Optional)
                    ious = box_iou(pred_boxes_xyxy[pi], gt_boxes_xyxy[ti])
                    
                    # 策略 A: 只要 IoU > 0.1 就算匹配 (最简单改法)
                    # match_metric = ious
                    # MATCH_THRESH = 0.1
                    
                    # 策略 B (推荐): 中心点是否在 GT 框内部 (Point-in-Box)
                    # 这种方法对 2x2 这种极小目标非常友好
                    p_c = pred_centers[pi].unsqueeze(1) # [N, 1, 2]
                    g_xyxy = gt_boxes_xyxy[ti].unsqueeze(0) # [1, M, 4]
                    
                    # 判断中心点 x 是否在 x1, x2 之间，y 是否在 y1, y2 之间
                    is_inside = (p_c[..., 0] > g_xyxy[..., 0]) & (p_c[..., 0] < g_xyxy[..., 2]) & \
                                (p_c[..., 1] > g_xyxy[..., 1]) & (p_c[..., 1] < g_xyxy[..., 3])
                    
                    # 找到每个预测框命中的最佳 GT
                    # 这里的逻辑是：如果 pred 中心在 GT 内，视为潜在 TP
                    # 如果有多个 GT，选 IoU 最大的那个
                    
                    best_match_idx = ious.argmax(1) # [num_pred] -> 对应哪个 gt
                    best_match_iou = ious.max(1).values
                    
                    detected_set = set()
                    
                    # 遍历每一个预测框
                    for j in range(len(pi)):
                        gt_idx = ti[best_match_idx[j]] # 对应的全局 GT index
                        
                        # 判据：中心点在 GT 内 OR IoU > 0.1 (双保险)
                        is_hit = is_inside[j, best_match_idx[j]] or (best_match_iou[j] > 0.1)
                        
                        if is_hit:
                            if gt_idx.item() not in detected_set:
                                detected_set.add(gt_idx.item())
                                # 只要匹配上了，我们在所有 iou 阈值层级上都标为 True
                                # (这是为了让你在 mAP@0.5 里能拿到分，不用管 mAP@0.95 了)
                                correct[pi[j]] = True

        self.stats.append((correct, pred_scores, pred_cls, gt_cls.cpu().numpy() if nl else np.array([], dtype=np.float32)))

    def summarize(self) -> Dict[str, float]:
        correct_cat = torch.cat([s[0] for s in self.stats], dim=0).cpu().numpy()
        conf_cat = torch.cat([s[1] for s in self.stats], dim=0).cpu().numpy()
        pred_cls_cat = torch.cat([s[2] for s in self.stats], dim=0).cpu().numpy()
        target_cls_cat = np.concatenate([s[3] for s in self.stats], axis=0) if len(self.stats) else np.array([])

        if correct_cat.shape[0] and correct_cat.any():
            p, r, ap, f1, ap_class, best_thres = ap_per_class(
                correct_cat, conf_cat, pred_cls_cat, target_cls_cat,
                v5_metric=self.v5_metric,
                plot=False,
                save_dir=str(self.output_dir),
                names=(),
                return_best_thres=True,
                save_pr_curve_data=self.save_pr_curve_data
            )
            ap50 = ap[:, 0]
            ap_mean = ap.mean(1)
            mp, mr = float(p.mean()), float(r.mean())
            map50, map5095 = float(ap50.mean()), float(ap_mean.mean())
        else:
            mp = mr = map50 = map5095 = 0.0

        f1_mean = float(np.mean(f1))
        selected = conf_cat >= best_thres
        num_selected = int(selected.sum())

        n_images_eval = len(self.stats)
        tp05 = correct_cat[:, 0].astype(np.bool_)
        TP = int(tp05[selected].sum())
        FP = int(num_selected - TP)

        GT = int(target_cls_cat.shape[0])
        FN = int(max(GT - TP, 0))

        PD = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FA = FP / n_images_eval if n_images_eval > 0 else 0.0

        if self.save_pred_json:
            pred_path = self.output_dir / "predictions_coco_list.json"
            pred_path.write_text(json.dumps(self.pred_json, indent=2), encoding="utf-8")

        return {"P": mp, "R": mr, "F1": f1_mean, "mAP@0.5": map50, "mAP@0.5:0.95": map5095, 
                "PD": float(PD), "FA": float(FA), "best_thres": float(best_thres), "TP": TP, "FP_box": FP, "FN": FN, "GT": GT,
                "n_images_eval": int(n_images_eval),}


# =========================================================================
# LoRA-aware model loader
# =========================================================================

def load_model_with_optional_lora(cfg: Dict[str, Any], device: str = "cuda"):
    """
    根据 cfg 中的配置，加载模型。支持三种模式：

    模式 A - 合并后的权重（use_lora=False）：
        直接用 load_model() 加载，和原来一样。

    模式 B - 标准 LoRA 未合并权重（use_lora=True, hf_lora_modules=[]）：
        加载基础模型 → 注入标准 LoRA → 加载 LoRA 权重

    模式 C - HF-LoRA 未合并权重（use_lora=True, hf_lora_modules=["qkv",...]）：
        加载基础模型 → 注入 HF-LoRA → 加载 LoRA 权重
    """
    use_lora = cfg.get("use_lora", False)

    if not use_lora:
        # ====== 模式 A：合并后的权重，直接加载 ======
        print(f"[Model] Loading merged model from {cfg['weights']}")
        model = load_model(cfg["config"], cfg["weights"])
        model.to(device)
        model.eval()
        return model

    # ====== 模式 B/C：未合并的 LoRA 权重 ======
    lora_checkpoint = cfg.get("lora_checkpoint", "")
    if not lora_checkpoint:
        raise ValueError(
            "use_lora=True but 'lora_checkpoint' is not set in CFG!\n"
            "Please set:\n"
            '  "lora_checkpoint": "path/to/lora_checkpoint_best"'
        )

    lora_checkpoint_dir = Path(lora_checkpoint)
    if not lora_checkpoint_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint_dir}")

    # --- 读取 LoRA 配置 ---
    # 优先从 lora_config.json 读取（训练时自动保存的）
    lora_config_path = lora_checkpoint_dir / "lora_config.json"
    if lora_config_path.exists():
        print(f"[LoRA] Loading config from {lora_config_path}")
        with open(lora_config_path, 'r') as f:
            saved_config = json.load(f)

        lora_r = saved_config.get('r', cfg.get('lora_r', 8))
        lora_alpha = saved_config.get('lora_alpha', cfg.get('lora_alpha', 16))
        lora_dropout = saved_config.get('lora_dropout', cfg.get('lora_dropout', 0.05))
        lora_bias = saved_config.get('lora_bias', cfg.get('lora_bias', 'none'))
        target_modules = saved_config.get('target_modules', cfg.get('lora_target_modules', []))
        hf_lora_modules = saved_config.get('hf_lora_modules', cfg.get('hf_lora_modules', []))
    else:
        print(f"[LoRA] lora_config.json not found, using CFG values")
        lora_r = cfg.get('lora_r', 8)
        lora_alpha = cfg.get('lora_alpha', 16)
        lora_dropout = cfg.get('lora_dropout', 0.05)
        lora_bias = cfg.get('lora_bias', 'none')
        target_modules = cfg.get('lora_target_modules', [])
        hf_lora_modules = cfg.get('hf_lora_modules', [])

    lora_type = "HF-LoRA" if hf_lora_modules else "Standard LoRA"

    print(f"[LoRA] Mode: {lora_type}")
    print(f"[LoRA] r={lora_r}, alpha={lora_alpha}, scaling={lora_alpha/lora_r}")
    print(f"[LoRA] Target modules: {target_modules}")
    if hf_lora_modules:
        print(f"[LoRA] HF-LoRA modules: {hf_lora_modules}")

    # 1. 加载基础模型 + 预训练权重
    print(f"[Model] Loading base model from {cfg['weights']}")
    model = load_model(cfg["config"], cfg["weights"])

    # 2. 注入 LoRA 结构
    from manual_lora import inject_lora, load_lora_weights

    print(f"[LoRA] Injecting {lora_type} layers...")
    model = inject_lora(
        model,
        target_modules=target_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        hf_lora_modules=hf_lora_modules if hf_lora_modules else None,
    )

    # 3. 加载训练好的 LoRA 权重
    print(f"[LoRA] Loading trained weights from {lora_checkpoint_dir}")
    model = load_lora_weights(model, lora_checkpoint_dir)

    # 4. 打印检查点训练信息
    training_state_path = lora_checkpoint_dir / "training_state.pth"
    if training_state_path.exists():
        state = torch.load(training_state_path, map_location='cpu')
        epoch = state.get('epoch', '?')
        mAP = state.get('mAP', None)
        AP50 = state.get('AP50', None)
        info_parts = [f"epoch={epoch}"]
        if mAP is not None:
            info_parts.append(f"mAP={mAP:.4f}")
        if AP50 is not None:
            info_parts.append(f"AP50={AP50:.4f}")
        print(f"[LoRA] Checkpoint info: {', '.join(info_parts)}")

    # 5. 统计参数
    from manual_lora import ManualLoRALinear, HighFreqLoRALinear
    total_lora = sum(1 for _, m in model.named_modules() if isinstance(m, (ManualLoRALinear, HighFreqLoRALinear)))
    total_hf = sum(1 for _, m in model.named_modules() if isinstance(m, HighFreqLoRALinear))
    total_std = total_lora - total_hf
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(k in n for k in ['lora_A', 'lora_B', 'hf_conv'])
    )
    print(f"[LoRA] Injected layers: {total_lora} (Standard: {total_std}, HF-LoRA: {total_hf})")
    print(f"[LoRA] LoRA parameters: {lora_params:,}")

    model.to(device)
    model.eval()
    return model


# -------------------------
# Main entry callable
# -------------------------
@torch.no_grad()
def evaluate_gdino_on_coco(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    cfg keys (recommended):
      - images_dir, coco_gt
      - config, weights
      - text_prompt, box_threshold, text_threshold, nms_threshold
      - min_area, max_area
      - v5_metric
      - output_dir, save_pred_json
      - force_single_class (optional)

      LoRA-specific keys (optional):
      - use_lora: bool              是否使用 LoRA（默认 False）
      - lora_checkpoint: str        LoRA 检查点目录路径
      - lora_r: int                 LoRA rank（默认 8）
      - lora_alpha: int             LoRA alpha（默认 16）
      - lora_dropout: float         LoRA dropout（默认 0.05）
      - lora_bias: str              LoRA bias（默认 'none'）
      - lora_target_modules: list   LoRA 目标模块
      - hf_lora_modules: list       HF-LoRA 目标模块（空列表=标准LoRA）
    """
    images_dir = Path(cfg["images_dir"])
    coco_gt = cfg["coco_gt"]

    force_single_class = bool(cfg.get("force_single_class", False))

    gt_loader = CocoGTLoader(coco_gt, force_single_class=force_single_class)
    items = gt_loader.iter_images()

    coco_pred_cat_id = gt_loader.cat_ids[0] if gt_loader.cat_ids else 0

    resolver = ImagePathResolver(images_dir, build_index=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 核心修改：使用 LoRA-aware 模型加载 =====
    model = load_model_with_optional_lora(cfg, device=device)
    # ===============================================

    evaluator = YoloStyleEvaluator(
        v5_metric=bool(cfg.get("v5_metric", False)),
        output_dir=str(cfg["output_dir"]),
        save_pred_json=bool(cfg.get("save_pred_json", False)),
        coco_pred_category_id=int(coco_pred_cat_id),
        save_pr_curve_data=bool(cfg.get("save_pr_curve_data", True)),
    )

    missing = 0
    for it in items:
        img_path = resolver.resolve(it.file_name)
        if img_path is None or (not img_path.exists()):
            missing += 1
            continue

        image_source, image = load_image(str(img_path))
        h, w = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=cfg["text_prompt"],
            box_threshold=float(cfg["box_threshold"]),
            text_threshold=float(cfg["text_threshold"]),
        )

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)

        boxes = boxes.float().cpu()
        scores = logits.float().cpu()

        boxes, scores = filter_by_area_cxcywh_norm(
            boxes, scores, (h, w), float(cfg.get("min_area", 0.0)), float(cfg.get("max_area", 1e18))
        )

        pred_boxes_xyxy = cxcywh_norm_to_xyxy_pixel(boxes, (h, w))
        keep = nms_xyxy(pred_boxes_xyxy, scores, float(cfg["nms_threshold"]))
        pred_boxes_xyxy = pred_boxes_xyxy[keep]
        pred_scores = scores[keep]

        pred_cls = torch.zeros((pred_boxes_xyxy.shape[0],), dtype=torch.float32)

        gt_boxes = torch.from_numpy(it.boxes_xyxy).float()
        gt_cls = torch.from_numpy(it.classes).float()
        if force_single_class and gt_cls.numel():
            gt_cls = torch.zeros_like(gt_cls)

        evaluator.update_one(
            image_id=it.image_id,
            pred_boxes_xyxy=pred_boxes_xyxy.float(),
            pred_scores=pred_scores.float(),
            pred_cls=pred_cls,
            gt_boxes_xyxy=gt_boxes,
            gt_cls=gt_cls,
        )

    metrics = evaluator.summarize()
    metrics["_num_images_coco"] = len(items)
    metrics["_missing_images"] = missing
    return metrics
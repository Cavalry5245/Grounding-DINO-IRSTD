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
        # 统一分隔符、去掉前导 ./ 或 /
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

        # 从索引精确匹配相对路径
        if s in self._rel_map:
            return self._rel_map[s]

        # basename 唯一匹配
        candidates = self._base_map.get(base, [])
        if len(candidates) == 1:
            return candidates[0]

        # basename 不唯一：尝试“尾部路径匹配”
        # 例如 COCO 给的是 subdir/a.jpg，但 images_dir 下真实是 images/subdir/a.jpg
        tail_matches = [c for c in candidates if c.as_posix().endswith(s)]
        if len(tail_matches) == 1:
            return tail_matches[0]
        if len(tail_matches) > 1:
            # 仍然不唯一，返回第一个并提示
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

        if nl:
            detected = []
            tcls_tensor = gt_cls

            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # gt
                pi = (cls == pred_cls).nonzero(as_tuple=False).view(-1)     # pred

                if pi.numel():
                    ious, i = box_iou(pred_boxes_xyxy[pi], gt_boxes_xyxy[ti]).max(1)

                    detected_set = set()
                    for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > self.iouv
                            if len(detected) == nl:
                                break

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
        # count-based PD/FA at IoU=0.5 and confidence >= best_thres
        selected = conf_cat >= best_thres
        num_selected = int(selected.sum())

        n_images_eval = len(self.stats)  # 实际参与评估的图像数（不含 missing） 
        tp05 = correct_cat[:, 0].astype(np.bool_)  # IoU=0.5 correctness
        TP = int(tp05[selected].sum())
        FP = int(num_selected - TP)

        GT = int(target_cls_cat.shape[0])
        FN = int(max(GT - TP, 0))

        PD = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # == TP/GT when GT>0
        # FA = FP / (TP + FP) if (TP + FP) > 0 else 0.0  # your definition
        FA = FP / n_images_eval if n_images_eval > 0 else 0.0

        if self.save_pred_json:
            pred_path = self.output_dir / "predictions_coco_list.json"
            pred_path.write_text(json.dumps(self.pred_json, indent=2), encoding="utf-8")

        return {"P": mp, "R": mr, "F1": f1_mean, "mAP@0.5": map50, "mAP@0.5:0.95": map5095, 
                "PD": float(PD), "FA": float(FA), "best_thres": float(best_thres), "TP": TP, "FP_box": FP, "FN": FN, "GT": GT,
                "n_images_eval": int(n_images_eval),}


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
    """
    images_dir = Path(cfg["images_dir"])
    coco_gt = cfg["coco_gt"]

    force_single_class = bool(cfg.get("force_single_class", False))

    gt_loader = CocoGTLoader(coco_gt, force_single_class=force_single_class)
    items = gt_loader.iter_images()

    # pred json 的 category_id：如果 GT 是单类且 category_id=0，就会是 0；否则取第一个 cat_id
    coco_pred_cat_id = gt_loader.cat_ids[0] if gt_loader.cat_ids else 0

    resolver = ImagePathResolver(images_dir, build_index=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(cfg["config"], cfg["weights"])
    model.to(device)

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

        # 你的任务是单类检测（prompt = 一个类），所以 pred_cls 全 0
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
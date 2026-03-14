import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO


# ==========================
# 配置（你只需要改这里）
# ==========================
CFG = {
    "images_dir": "data/NUDT-SIRST/images/test",
    "coco_gt": "data/NUDT-SIRST/annotations/test.json",

    # 评估输出目录（用于读取 metrics.json 的 best_thres）
    "eval_output_dir": "eval_output/test/NUDT-SIRST2",
    "pred_json": "eval_output/test/NUDT-SIRST2/predictions_coco_list.json",

    # 阈值：用 "best" 则自动读取 metrics.json 里的 best_thres；或写成浮点数例如 0.3
    "conf": "best",

    # IoU阈值：用于把预测标成 TP/FP（可视化用）
    "iou_thres": 0.5,

    # 输出目录
    "save_dir": "eval_output/test/NUDT-SIRST2/viz",

    # 可视化数量控制
    "max_images": 200,          # 最多保存多少张（None 表示全部）
    "only_images_with_fp": False,  # True：只保存含FP的图（看误检）
    "only_images_with_fn": False,  # True：只保存含FN的图（看漏检）

    # 是否同时画 GT
    "draw_gt": False,

    # 字体/线宽
    "box_thickness": 1,
    "font_scale": 0.5,
}


# ==========================
# 工具函数
# ==========================
def load_best_thres(eval_output_dir: str) -> Optional[float]:
    p = Path(eval_output_dir) / "metrics.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    # 兼容你保存结构：{"metrics": {...}} 或直接 {...}
    if "metrics" in j and isinstance(j["metrics"], dict):
        return j["metrics"].get("best_thres", None)
    return j.get("best_thres", None)


class ImagePathResolver:
    """兼容 COCO file_name 带/不带子目录"""
    def __init__(self, images_dir: str, exts=None, build_index=True):
        self.images_dir = Path(images_dir)
        self.exts = exts or {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.rel_map: Dict[str, Path] = {}
        self.base_map: Dict[str, List[Path]] = {}
        if build_index:
            self._build_index()

    def _build_index(self):
        if not self.images_dir.exists():
            return
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.exts:
                rel = p.relative_to(self.images_dir).as_posix()
                self.rel_map[rel] = p
                self.base_map.setdefault(p.name, []).append(p)

    @staticmethod
    def _norm(s: str) -> str:
        s = s.replace("\\", "/")
        while s.startswith("./"):
            s = s[2:]
        return s.lstrip("/")

    def resolve(self, coco_file_name: str) -> Optional[Path]:
        s = self._norm(coco_file_name)

        p1 = self.images_dir / s
        if p1.exists():
            return p1

        base = Path(s).name
        p2 = self.images_dir / base
        if p2.exists():
            return p2

        if s in self.rel_map:
            return self.rel_map[s]

        cands = self.base_map.get(base, [])
        if len(cands) == 1:
            return cands[0]

        tail = [c for c in cands if c.as_posix().endswith(s)]
        if len(tail) == 1:
            return tail[0]
        if len(tail) > 1:
            print(f"[WARN] multiple matches for {coco_file_name}, choose {tail[0]}")
            return tail[0]

        if len(cands) > 1:
            print(f"[WARN] basename collision for {base}, choose {cands[0]}")
            return cands[0]

        return None


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def clamp_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), w - 1.0))
    y1 = max(0.0, min(float(y1), h - 1.0))
    x2 = max(0.0, min(float(x2), w - 1.0))
    y2 = max(0.0, min(float(y2), h - 1.0))
    # 避免反向框
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N,4], b: [M,4] in xyxy
    returns IoU [N,M]
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-16
    return (inter / union).astype(np.float32)


def greedy_match_tp_fp(pred_xyxy: np.ndarray, pred_scores: np.ndarray,
                       gt_xyxy: np.ndarray, iou_thres: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      pred_is_tp: [N] bool
      gt_matched: [M] bool
    逻辑：按 score 降序，每个 pred 找 IoU 最大的未匹配 GT，>=iou_thres 即 TP
    """
    n = len(pred_xyxy)
    m = len(gt_xyxy)
    pred_is_tp = np.zeros((n,), dtype=bool)
    gt_matched = np.zeros((m,), dtype=bool)
    if n == 0 or m == 0:
        return pred_is_tp, gt_matched

    order = np.argsort(-pred_scores)
    ious = box_iou_np(pred_xyxy[order], gt_xyxy)  # [N,M]

    for k, pi in enumerate(order):
        best_j = int(np.argmax(ious[k]))
        best_iou = float(ious[k, best_j])
        if best_iou >= iou_thres and (not gt_matched[best_j]):
            pred_is_tp[pi] = True
            gt_matched[best_j] = True

    return pred_is_tp, gt_matched


def draw_box(img, box, color, thickness, text=None, font_scale=0.5):
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        y = max(0, y1 - 5)
        cv2.putText(img, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


# ==========================
# 主流程
# ==========================
def main():
    images_dir = CFG["images_dir"]
    coco_gt_path = CFG["coco_gt"]
    pred_json_path = CFG["pred_json"]
    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # threshold
    conf = CFG["conf"]
    if conf == "best":
        best = load_best_thres(CFG["eval_output_dir"])
        if best is None:
            raise RuntimeError("CFG['conf']='best' but best_thres not found in metrics.json")
        conf_thres = float(best)
    else:
        conf_thres = float(conf)

    print(f"Using conf_thres = {conf_thres:.4f} | IoU_thres = {CFG['iou_thres']}")

    # load coco gt
    coco = COCO(coco_gt_path)
    imgs = coco.loadImgs(coco.getImgIds())
    imgid_to_info = {int(im["id"]): im for im in imgs}

    # load predictions list
    preds = json.loads(Path(pred_json_path).read_text(encoding="utf-8"))
    # group preds by image_id
    pred_by_img: Dict[int, List[dict]] = {}
    for p in preds:
        iid = int(p["image_id"])
        pred_by_img.setdefault(iid, []).append(p)

    resolver = ImagePathResolver(images_dir, build_index=True)

    saved = 0
    missing = 0

    # iterate over coco images (so we can also visualize images with 0 preds)
    for image_id, info in imgid_to_info.items():
        file_name = info["file_name"]
        img_path = resolver.resolve(file_name)
        if img_path is None or (not img_path.exists()):
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]

        # GT boxes
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_xyxy = []
        for a in anns:
            b = xywh_to_xyxy(np.array(a["bbox"], dtype=np.float32))
            b = clamp_xyxy(b, w, h)
            gt_xyxy.append(b)
        gt_xyxy = np.array(gt_xyxy, dtype=np.float32).reshape(-1, 4)

        # Pred boxes
        plist = pred_by_img.get(image_id, [])
        pred_xyxy = []
        pred_scores = []
        for p in plist:
            s = float(p.get("score", 0.0))
            if s < conf_thres:
                continue
            bx = np.array(p["bbox"], dtype=np.float32)  # xywh
            b = xywh_to_xyxy(bx)
            b = clamp_xyxy(b, w, h)
            pred_xyxy.append(b)
            pred_scores.append(s)
        pred_xyxy = np.array(pred_xyxy, dtype=np.float32).reshape(-1, 4)
        pred_scores = np.array(pred_scores, dtype=np.float32).reshape(-1)

        # TP/FP (visualization)
        pred_is_tp, gt_matched = greedy_match_tp_fp(pred_xyxy, pred_scores, gt_xyxy, CFG["iou_thres"])

        num_fp = int((~pred_is_tp).sum()) if len(pred_is_tp) else 0
        num_fn = int((~gt_matched).sum()) if len(gt_matched) else len(gt_xyxy)

        if CFG["only_images_with_fp"] and num_fp == 0:
            continue
        if CFG["only_images_with_fn"] and num_fn == 0:
            continue

        vis = img.copy()

        # draw GT
        if CFG["draw_gt"] and len(gt_xyxy):
            for j, g in enumerate(gt_xyxy):
                # matched GT: green, unmatched GT (FN): cyan
                if len(gt_matched) and gt_matched[j]:
                    color = (0, 200, 0)
                    txt = "GT(TP)"
                else:
                    color = (255, 255, 0)
                    txt = "GT(FN)"
                draw_box(vis, g, color=color, thickness=CFG["box_thickness"], text=txt, font_scale=CFG["font_scale"])

        # draw preds
        for b, s, is_tp in zip(pred_xyxy, pred_scores, pred_is_tp):
            if is_tp:
                color = (0, 0, 255)     # TP 
                # tag = "TP"
            else:
                color = (0, 255, 255)     # FP 
                # tag = "FP"
            # draw_box(vis, b, color=color, thickness=CFG["box_thickness"],
            #          text=f"{tag} {s:.3f}", font_scale=CFG["font_scale"])
            draw_box(vis, b, color=color, thickness=CFG["box_thickness"],
                     font_scale=CFG["font_scale"])

        # header text
        # header = f"id={image_id}  preds={len(pred_xyxy)}  fp={num_fp}  fn={num_fn}  conf>={conf_thres:.3f}"
        # cv2.putText(vis, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

        # save (保持原 file_name 的子目录结构更好查)
        rel = file_name.replace("\\", "/")
        out_path = save_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

        saved += 1
        if CFG["max_images"] is not None and saved >= int(CFG["max_images"]):
            break

    print(f"Saved {saved} visualizations to: {save_dir}")
    print(f"Missing images: {missing}")


if __name__ == "__main__":
    main()
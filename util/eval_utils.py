"""
扩展评估工具模块
提供PR曲线计算、指标保存、JSON导出等功能
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class ExtendedMetrics:
    """
    扩展指标计算类
    收集预测结果并计算详细的P/R/F1指标
    """
    
    def __init__(self, nc=80, conf_thres=0.001, iou_thres=0.5, save_dir=''):
        """
        Args:
            nc: 类别数量
            conf_thres: 置信度阈值
            iou_thres: IoU阈值（用于匹配）
            save_dir: 保存目录
        """
        self.nc = nc
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储统计信息: (correct, conf, pred_cls, target_cls)
        self.stats = []
        
        # IoU阈值向量 (与COCO一致)
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.niou = len(self.iouv)
        
    def update(self, predictions, targets):
        """
        更新统计信息
        
        Args:
            predictions: list of dict, 每个dict包含 {'boxes': [N,4], 'scores': [N], 'labels': [N]}
                        boxes格式为xyxy，像素坐标
            targets: list of dict, 每个dict包含 {'boxes': [M,4], 'labels': [M]}
                    boxes格式为xyxy，像素坐标
        """
        for pred, tgt in zip(predictions, targets):
            # 转换为numpy
            pred_boxes = self._to_numpy(pred.get('boxes', []))
            pred_scores = self._to_numpy(pred.get('scores', []))
            pred_labels = self._to_numpy(pred.get('labels', []))

            if len(pred_scores) > 0:
                # 必须降序排列，否则 _match_predictions 中的贪婪匹配逻辑会出错
                sort_idx = np.argsort(-pred_scores) 
                pred_boxes = pred_boxes[sort_idx]
                pred_scores = pred_scores[sort_idx]
                pred_labels = pred_labels[sort_idx]
            
            tgt_boxes = self._to_numpy(tgt.get('boxes', []))
            tgt_labels = self._to_numpy(tgt.get('labels', []))
            
            # 获取该图片的真实类别列表
            tcls = tgt_labels.tolist() if len(tgt_labels) > 0 else []
            nl = len(tgt_labels)
            
            # 无预测的情况
            if len(pred_boxes) == 0:
                if nl:
                    self.stats.append((
                        np.zeros((0, self.niou), dtype=bool),
                        np.array([]),
                        np.array([]),
                        tcls
                    ))
                continue
            
            # 初始化correct矩阵
            correct = np.zeros((len(pred_boxes), self.niou), dtype=bool)
            
            if nl:
                # 计算IoU并匹配
                ious = self._box_iou(pred_boxes, tgt_boxes)
                correct = self._match_predictions(pred_labels, tgt_labels, ious, correct)
            
            self.stats.append((
                correct,
                pred_scores,
                pred_labels,
                tcls
            ))
    
    def _to_numpy(self, x):
        """转换为numpy数组"""
        if torch.is_tensor(x):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, list):
            return np.array(x)
        else:
            return np.array([])
    
    def _box_iou(self, boxes1, boxes2):
        """
        计算两组框的IoU矩阵
        boxes格式: xyxy
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
            
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 计算交集
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # 计算IoU
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-7)
        
        return iou
    
    def _match_predictions(self, pred_labels, tgt_labels, ious, correct):
        """
        匹配预测和真实框（贪婪匹配）
        """
        for cls in np.unique(tgt_labels):
            ti = np.where(tgt_labels == cls)[0]  # 该类别的GT索引
            pi = np.where(pred_labels == cls)[0]  # 该类别的预测索引
            
            if len(pi) == 0 or len(ti) == 0:
                continue
                
            # 获取该类别的IoU
            cls_ious = ious[pi][:, ti]
            
            # 按预测置信度的顺序（假设已按置信度排序）进行贪婪匹配
            matched_gt = set()
            for i, p_idx in enumerate(pi):
                if len(matched_gt) >= len(ti):
                    break
                    
                # 找最大IoU的GT
                valid_mask = np.array([j not in matched_gt for j in range(len(ti))])
                if not valid_mask.any():
                    continue
                    
                masked_ious = cls_ious[i].copy()
                masked_ious[~valid_mask] = 0
                
                max_iou_idx = np.argmax(masked_ious)
                max_iou = masked_ious[max_iou_idx]
                
                if max_iou >= self.iouv[0]:
                    matched_gt.add(max_iou_idx)
                    correct[p_idx] = max_iou >= self.iouv
                    
        return correct
    
    def compute_metrics(self):
        """
        计算所有指标
        
        Returns:
            dict: 包含所有计算的指标
        """
        if len(self.stats) == 0:
            return None
            
        # 合并所有统计信息
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        
        if len(stats) == 0 or len(stats[0]) == 0:
            return None
            
        tp, conf, pred_cls, target_cls = stats
        
        # 按置信度排序
        sort_idx = np.argsort(-conf)
        tp, conf, pred_cls = tp[sort_idx], conf[sort_idx], pred_cls[sort_idx]
        
        # 计算每个类别的指标
        unique_classes = np.unique(np.concatenate([pred_cls, target_cls]))
        nc = len(unique_classes)
        
        # 存储结果
        ap = np.zeros((nc, self.niou))
        precision_curves = []
        recall_curves = []
        f1_curves = []
        conf_curves = []
        
        p_at_best_f1 = np.zeros(nc)
        r_at_best_f1 = np.zeros(nc)
        best_conf = np.zeros(nc)
        
        for ci, cls in enumerate(unique_classes):
            i = pred_cls == cls
            n_gt = (target_cls == cls).sum()
            n_pred = i.sum()
            
            if n_pred == 0 or n_gt == 0:
                precision_curves.append(np.array([]))
                recall_curves.append(np.array([]))
                f1_curves.append(np.array([]))
                conf_curves.append(np.array([]))
                continue
            
            # 累积TP和FP
            tpc = tp[i].cumsum(0)
            fpc = (1 - tp[i]).cumsum(0)
            
            # Recall
            recall = tpc / (n_gt + 1e-16)
            
            # Precision  
            precision = tpc / (tpc + fpc + 1e-16)
            
            # 计算AP (对每个IoU阈值)
            for j in range(self.niou):
                ap[ci, j] = self._compute_ap(recall[:, j], precision[:, j])
            
            # 保存曲线数据 (使用IoU=0.5的数据)
            prec = precision[:, 0]
            rec = recall[:, 0]
            f1 = 2 * prec * rec / (prec + rec + 1e-16)
            
            precision_curves.append(prec)
            recall_curves.append(rec)
            f1_curves.append(f1)
            conf_curves.append(conf[i])
            
            # 最佳F1时的P和R
            if len(f1) > 0:
                best_f1_idx = np.argmax(f1)
                p_at_best_f1[ci] = prec[best_f1_idx]
                r_at_best_f1[ci] = rec[best_f1_idx]
                best_conf[ci] = conf[i][best_f1_idx]
        
        # 汇总结果
        valid_mask = p_at_best_f1 > 0
        
        results = {
            'ap': ap,
            'ap50': ap[:, 0].mean() if ap[:, 0].any() else 0,
            'ap75': ap[:, 5].mean() if ap.shape[1] > 5 and ap[:, 5].any() else 0,
            'map': ap.mean() if ap.any() else 0,
            'precision': p_at_best_f1[valid_mask].mean() if valid_mask.any() else 0,
            'recall': r_at_best_f1[valid_mask].mean() if valid_mask.any() else 0,
            'f1': 0,
            'precision_curves': precision_curves,
            'recall_curves': recall_curves,
            'f1_curves': f1_curves,
            'conf_curves': conf_curves,
            'unique_classes': unique_classes,
            'p_per_class': p_at_best_f1,
            'r_per_class': r_at_best_f1,
            'best_conf_per_class': best_conf,
            'ap_per_class': ap,
            'n_gt_per_class': np.array([(target_cls == c).sum() for c in unique_classes]),
            'n_pred_per_class': np.array([(pred_cls == c).sum() for c in unique_classes])
        }
        
        # 计算F1
        if results['precision'] > 0 and results['recall'] > 0:
            results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'])
        
        return results
    
    def _compute_ap(self, recall, precision):
        """
        计算AP (PR曲线下面积)
        """
        if len(recall) == 0:
            return 0
            
        # 添加首尾点
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        
        # 使precision单调递减
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        
        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
            
        return ap
    
    def save_results(self, names=None):
        """
        保存结果到文件
        
        Args:
            names: 类别名称字典 {id: name}
        """
        results = self.compute_metrics()
        if results is None:
            print("No results to save")
            return None
        
        # 保存原始结果数据，方便后续画对比图
        data_save_path = self.save_dir / 'results.pt'
        torch.save(results, data_save_path)
        print(f"Raw results saved to {data_save_path}")
        
        # 保存文本结果
        self._save_text_results(results, names)
        
        return results
    
    def _save_text_results(self, results, names=None):
        """
        保存文本格式结果
        """
        save_path = self.save_dir / 'extended_metrics.txt'
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Extended Evaluation Results\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体指标
            f.write("Overall Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Precision:     {results['precision']:.4f}\n")
            f.write(f"  Recall:        {results['recall']:.4f}\n")
            f.write(f"  F1-Score:      {results['f1']:.4f}\n")
            f.write(f"  mAP@0.5:       {results['ap50']:.4f}\n")
            f.write(f"  mAP@0.75:      {results['ap75']:.4f}\n")
            f.write(f"  mAP@0.5:0.95:  {results['map']:.4f}\n")
            f.write("\n")
            
            # 每个类别的指标
            f.write("Per-Class Metrics:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Class':<25} {'GT':>8} {'Pred':>8} {'P':>10} {'R':>10} {'AP50':>10} {'AP':>10} {'BestConf':>10}\n")
            f.write("-" * 100 + "\n")
            
            for i, cls in enumerate(results['unique_classes']):
                cls_name = names.get(int(cls), str(int(cls))) if names else str(int(cls))
                n_gt = int(results['n_gt_per_class'][i])
                n_pred = int(results['n_pred_per_class'][i])
                p = results['p_per_class'][i]
                r = results['r_per_class'][i]
                ap50 = results['ap_per_class'][i, 0]
                ap = results['ap_per_class'][i].mean()
                best_conf = results['best_conf_per_class'][i]
                
                f.write(f"{cls_name:<25} {n_gt:>8} {n_pred:>8} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f} {best_conf:>10.3f}\n")
            
            f.write("=" * 100 + "\n")
        
        print(f"Extended metrics saved to {save_path}")


class JSONResultsSaver:
    """
    COCO格式JSON结果保存器
    """
    
    def __init__(self, save_dir=''):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def update(self, predictions, image_ids, coco80_to_coco91=None):
        """
        更新预测结果
        
        Args:
            predictions: list of dict, 每个包含 boxes, scores, labels
            image_ids: list of image ids
            coco80_to_coco91: 80类到91类的映射字典 (可选)
        """
        for pred, img_id in zip(predictions, image_ids):
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            labels = pred.get('labels', [])
            
            # 转换为numpy/list
            if torch.is_tensor(boxes):
                boxes = boxes.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            if len(boxes) == 0:
                continue
                
            # 转换box格式: xyxy -> xywh (COCO格式)
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            
            for i in range(len(boxes)):
                cat_id = int(labels[i])
                if coco80_to_coco91 is not None:
                    cat_id = coco80_to_coco91.get(cat_id, cat_id)
                
                self.results.append({
                    'image_id': int(img_id) if isinstance(img_id, (int, np.integer)) else img_id,
                    'category_id': cat_id,
                    'bbox': [round(float(x), 3) for x in boxes_xywh[i]],
                    'score': round(float(scores[i]), 5)
                })
    
    def save(self, filename='predictions.json'):
        """
        保存结果到JSON文件
        """
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Predictions saved to {save_path} ({len(self.results)} detections)")
        return save_path
    
    def reset(self):
        """重置结果"""
        self.results = []
"""
可视化工具模块
提供PR曲线、F1曲线、预测对比图等可视化功能
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import seaborn as sns
import pandas as pd
import torch

# 尝试导入cv2，如果失败则使用PIL
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image, ImageDraw, ImageFont


class EvalPlotter:
    """
    评估可视化绘图器
    """
    
    def __init__(self, save_dir='', names=None):
        """
        Args:
            save_dir: 保存目录
            names: 类别名称字典 {id: name}
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.names = names or {}
        
        # 颜色设置
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    def plot_pr_curve(self, results):
            """
            绘制全局平均PR曲线 (仿照mAP@50风格)
            将所有类别的曲线插值到标准的Recall轴上，然后求平均。
            """
            if results is None:
                return
                
            precision_curves = results.get('precision_curves', [])
            recall_curves = results.get('recall_curves', [])
            
            # 获取 mAP@0.5 用于图例显示
            map50 = results.get('ap50', 0.0)
            
            if len(precision_curves) == 0:
                return
                
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 1. 定义标准的 Recall X轴 (0 到 1，共 101 个点，对应 COCO 标准)
            x_grid = np.linspace(0, 1, 101)
            
            # 用于存放所有类别插值后的 Precision
            interp_precisions = []
            
            # 2. 遍历每个类别进行插值
            for prec, rec in zip(precision_curves, recall_curves):
                if len(prec) == 0 or len(rec) == 0:
                    continue
                    
                # np.interp 要求 x (rec) 必须是单调递增的
                # eval_utils 计算出的 recall 应该是递增的，但为了保险起见做个排序
                if rec.shape[0] > 1 and rec[0] > rec[-1]:
                    rec = rec[::-1]
                    prec = prec[::-1]
                elif rec.shape[0] > 1 and not np.all(rec[1:] >= rec[:-1]):
                    # 如果是乱序的（极少情况），进行排序
                    sort_idx = np.argsort(rec)
                    rec = rec[sort_idx]
                    prec = prec[sort_idx]

                # 将当前类别的 P-R 映射到标准的 x_grid 上
                # right=0 表示 Recall 大于实际最大值的部分 Precision 视为 0
                p_interp = np.interp(x_grid, rec, prec, left=1.0, right=0.0)
                interp_precisions.append(p_interp)

            # 3. 计算并绘图
            if interp_precisions:
                # 转换为矩阵 (n_classes, 101)
                interp_precisions = np.array(interp_precisions)
                
                # 计算平均值 (Mean Precision)
                mean_precision = interp_precisions.mean(axis=0)
                
                # 绘制所有类别的灰色细线作为背景 (可选，这样看起来更专业)
                for p in interp_precisions:
                    ax.plot(x_grid, p, color='gray', linewidth=0.5, alpha=0.2)
                
                # 绘制加粗的蓝色平均线 (这是核心)
                ax.plot(x_grid, mean_precision, color='#1f77b4', linewidth=1, 
                        label=f'mAP@0.5 = {map50:.3f}')
                
                # 填充曲线下方面积
                # ax.fill_between(x_grid, mean_precision, alpha=0.1, color='#1f77b4')

            # 4. 设置图表样式
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_title('Precision-Recall Curve', fontsize=14)
            ax.legend(loc='lower left', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 保存
            save_path = self.save_dir / 'PR_curve.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Global PR curve saved to {save_path}")
    
    def plot_p_curve(self, results):
        """
        绘制Precision vs Confidence曲线
        """
        self._plot_metric_vs_conf(results, 'precision_curves', 'Precision', 'P_curve.png')
    
    def plot_r_curve(self, results):
        """
        绘制Recall vs Confidence曲线
        """
        self._plot_metric_vs_conf(results, 'recall_curves', 'Recall', 'R_curve.png')
    
    def plot_f1_curve(self, results):
        """
        绘制F1 vs Confidence曲线
        """
        self._plot_metric_vs_conf(results, 'f1_curves', 'F1', 'F1_curve.png')
    
    def _plot_metric_vs_conf(self, results, key, metric_name, filename):
        """
        绘制指标随置信度/样本数变化的曲线
        """
        if results is None:
            return
            
        curves = results.get(key, [])
        class_ids = results.get('unique_classes', [])
        
        if len(curves) == 0:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制每个类别的曲线
        for i, (values, cls_id) in enumerate(zip(curves, class_ids)):
            if len(values) == 0:
                continue
            x = np.linspace(0, 1, len(values))
            color = self.colors[i % len(self.colors)]
            cls_name = self.names.get(int(cls_id), str(int(cls_id)))
            ax.plot(x, values, color=color, linewidth=1, alpha=0.5, label=cls_name)
        
        # 计算并绘制平均曲线
        valid_curves = [c for c in curves if len(c) > 0]
        if len(valid_curves) > 0:
            max_len = max(len(c) for c in valid_curves)
            avg_values = np.zeros(max_len)
            count = 0
            for c in valid_curves:
                interp_c = np.interp(np.linspace(0, 1, max_len), 
                                    np.linspace(0, 1, len(c)), c)
                avg_values += interp_c
                count += 1
            if count > 0:
                avg_values /= count
                ax.plot(np.linspace(0, 1, max_len), avg_values, 
                       color='blue', linewidth=3, label=f'Mean {metric_name}')
        
        ax.set_xlabel('Normalized Sample Index', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_title(f'{metric_name} Curve', fontsize=14)
        
        # 只显示平均曲线的图例
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend([handles[-1]], [labels[-1]], loc='lower left')
            
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"{metric_name} curve saved to {save_path}")
    
    def plot_all_curves(self, results):
        """
        绘制所有曲线
        """
        if results is None:
            print("No results to plot")
            return
        
        self.plot_pr_curve(results)
        self.plot_p_curve(results)
        self.plot_r_curve(results)
        self.plot_f1_curve(results)


class PredictionVisualizer:
    """
    极简版可视化器：只做原图和预测结果的对比
    """
    
    def __init__(self, save_dir='', names=None, max_batches=5):
        self.save_dir = Path(save_dir) / 'visualizations'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.names = names or {}
        self.batch_count = 0
        self.max_batches = max_batches
        
        # 定义一些高亮颜色 (RGB格式)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (255, 128, 0)
        ]

    def visualize_batch(self, images, targets, predictions, image_ids=None):
        if self.batch_count >= self.max_batches:
            return

        # 兼容 NestedTensor
        if hasattr(images, 'tensors'):
            images = images.tensors
            
        batch_size = min(len(images), 4) # 每次最多画4张
        
        for i in range(batch_size):
            # --- 1. 图像转换 (不做复杂的去归一化) ---
            img_tensor = images[i]
            img_np = img_tensor.cpu().numpy()
            
            # [C, H, W] -> [H, W, C]
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            
            # 简单处理：如果是0-1的浮点数，转为0-255
            # 注意：如果你的输入是经过 mean/std 标准化的(数值范围-2到2)，这里出来的图可能是黑的或过曝的
            # 但既然要求“不做处理”，这里只做最安全的类型转换
            if img_np.max() <= 1.05:
                img_np = img_np * 255
            
            # 转为 uint8 并确保连续内存 (防止cv2报错)
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)
            
            # RGB -> BGR (OpenCV默认使用BGR)
            if HAS_CV2:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_np # 如果没有cv2，就凑合用RGB
            
            h, w = img_bgr.shape[:2]
            
            # 复制两份画布
            canvas_gt = img_bgr.copy()
            canvas_pred = img_bgr.copy()
            
            # --- 2. 绘制 GT (真实框) ---
            tgt = targets[i]
            gt_boxes = self._to_numpy(tgt.get('boxes', []))
            gt_labels = self._to_numpy(tgt.get('labels', []))
            
            # 坐标转换：如果GT是归一化的(<=1.05)，认为是cxcywh，转为xyxy
            if len(gt_boxes) > 0 and gt_boxes.max() <= 1.05:
                cx, cy, bw, bh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_boxes = np.stack([x1, y1, x2, y2], axis=1)
                
            self._draw_boxes(canvas_gt, gt_boxes, gt_labels, title="Ground Truth")
            
            # --- 3. 绘制 Prediction (预测框) ---
            pred = predictions[i]
            pred_boxes = self._to_numpy(pred.get('boxes', []))
            pred_labels = self._to_numpy(pred.get('labels', []))
            pred_scores = self._to_numpy(pred.get('scores', []))
            
            # 简单过滤：只画置信度 > 0.3 的
            if len(pred_scores) > 0:
                mask = pred_scores > 0.3
                self._draw_boxes(canvas_pred, pred_boxes[mask], pred_labels[mask], 
                               scores=pred_scores[mask], title="Prediction")
            
            # --- 4. 拼接并保存 ---
            # 左右拼接
            combined = np.hstack([canvas_gt, canvas_pred])
            
            save_path = self.save_dir / f'batch{self.batch_count}_img{i}.jpg'
            if HAS_CV2:
                cv2.imwrite(str(save_path), combined)
            else:
                # 兼容无cv2环境
                from PIL import Image
                Image.fromarray(combined).save(save_path)
                
        self.batch_count += 1
        print(f"Visualization saved to {self.save_dir}")

    def _draw_boxes(self, img, boxes, labels, scores=None, title=""):
        """使用OpenCV绘制"""
        if not HAS_CV2: return
        
        # 字体和线宽自适应
        h, w = img.shape[:2]
        lw = max(2, int(w * 0.003))  # 线宽
        font_scale = lw / 3
        
        # 写标题
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(labels[j])
            color = self.colors[cls_id % len(self.colors)][::-1] # RGB转BGR
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
            
            # 标签文字
            name = self.names.get(cls_id, str(cls_id))
            if scores is not None:
                text = f"{name} {scores[j]:.2f}"
            else:
                text = name
                
            # 文字背景
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    def _to_numpy(self, x):
        if hasattr(x, 'cpu'): return x.cpu().numpy()
        if isinstance(x, list): return np.array(x)
        return x
        
    def reset(self):
        self.batch_count = 0
def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        if field == 'mAP':
            ax.legend([Path(p).name for p in logs])
            ax.set_title(field)
        else:
            ax.legend([f'train', f'test'])
            ax.set_title(field)

    return fig, axs
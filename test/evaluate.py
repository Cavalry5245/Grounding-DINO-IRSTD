import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. NUAA-SIRST 数据集 Mask 的文件夹路径
# 注意：文件名需要和预测结果里的文件名对应
GT_FOLDER = 'dataset/SIRST/masks'

# 2. Grounding DINO 的预测结果文件 (假设你存成了JSON格式)
# 格式参考下方的 load_predictions 函数注释
PRED_JSON = 'test_output/Swin_T/results_0.0/SIRST/instances_results.json'

# 3. 阈值设置
CONF_THRESHOLD = 0.1756  # 置信度阈值，低于这个分数的框丢弃
IOU_THRESHOLD = 0.1    # 小目标IoU很难高，0.1或0.0通常就够了，或者用Center-Hit

# ===========================================

def compute_iou(box1, box2):
    """计算两个矩形框的IoU: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0: return 0
    return intersection / union

def is_center_hit(pred_box, gt_box):
    """
    红外小目标常用指标：如果预测框包含了GT的中心点，就算检测成功。
    pred_box: [x1, y1, x2, y2]
    gt_box: [x1, y1, x2, y2]
    """
    gt_cx = (gt_box[0] + gt_box[2]) / 2
    gt_cy = (gt_box[1] + gt_box[3]) / 2
    
    if (pred_box[0] <= gt_cx <= pred_box[2]) and \
       (pred_box[1] <= gt_cy <= pred_box[3]):
        return True
    return False

def get_gt_boxes_from_mask(mask_path):
    """从Mask图像中提取GT Bounding Boxes"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # 二值化
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    gt_boxes = []
    # labels=0 是背景，所以从1开始遍历
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # 格式: [x1, y1, x2, y2]
        gt_boxes.append([x, y, x + w, y + h])
        
    return gt_boxes

def load_predictions(json_path):
    """
    读取预测结果 (适配 COCO 格式)。
    COCO 格式结构:
    {
        "images": [{"id": 1, "file_name": "Misc_1.png"}, ...],
        "annotations": [
            {"image_id": 1, "bbox": [x, y, width, height], "score": 0.9}, 
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建 image_id 到 file_name 的映射
    id_to_filename = {}
    for img_info in data['images']:
        id_to_filename[img_info['id']] = img_info['file_name']
    
    # 转换为字典方便查找: {'文件名': {'boxes': [], 'scores': []}}
    preds_dict = {}
    
    # 初始化所有图像
    for img_info in data['images']:
        filename = img_info['file_name']
        preds_dict[filename] = {
            'boxes': [],
            'scores': []
        }
    
    # 处理 annotations
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id in id_to_filename:
            filename = id_to_filename[image_id]
            # COCO bbox 格式是 [x, y, width, height]，需要转换为 [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]  # 转换为 [x1, y1, x2, y2]
            score = ann.get('score', 1.0)  # 如果没有 score 字段，默认为 1.0
            
            preds_dict[filename]['boxes'].append(box)
            preds_dict[filename]['scores'].append(score)
    
    return preds_dict

def evaluate():
    preds = load_predictions(PRED_JSON)
    
    # 统计变量
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    # 获取GT文件夹下所有图片
    gt_files = [f for f in os.listdir(GT_FOLDER) if f.endswith(('.png', '.jpg', '.bmp'))]
    
    print(f"开始评估，共 {len(gt_files)} 张图片...")

    for filename in tqdm(gt_files):
        # 1. 获取 Ground Truth Boxes
        gt_path = os.path.join(GT_FOLDER, filename)
        gt_boxes = get_gt_boxes_from_mask(gt_path)
        total_gt += len(gt_boxes)
        
        # 2. 获取 Prediction Boxes
        # 注意：这里可能需要处理文件名匹配问题（比如Mask文件名带_mask后缀）
        # 假设 prediction 里的文件名和 mask 文件名一致
        if filename in preds:
            pred_info = preds[filename]
            p_boxes = pred_info['boxes']
            p_scores = pred_info['scores']
        else:
            p_boxes = []
            p_scores = []
            
        # 过滤低置信度
        valid_preds = []
        for box, score in zip(p_boxes, p_scores):
            if score >= CONF_THRESHOLD:
                valid_preds.append(box)
        
        # 3. 匹配统计 (使用 Center-Hit 或 IoU)
        # 简单的贪婪匹配策略
        matched_gt_indices = set()
        
        # 先算 TP (有多少GT被检出了)
        # 遍历每一个预测框
        current_image_tp = 0
        current_image_fp = 0
        
        # 标记哪些预测框是匹配成功的
        pred_matched = [False] * len(valid_preds)
        
        # 这是一个简化的匹配逻辑：如果预测框命中了任意一个未匹配的GT，就算TP
        # 严谨逻辑应该用匈牙利算法，但对于稀疏的小目标，简单遍历通常足够
        for i, p_box in enumerate(valid_preds):
            is_match = False
            for j, g_box in enumerate(gt_boxes):
                # 使用 Center-Hit 判定 (推荐)
                if is_center_hit(p_box, g_box): 
                # 或者使用 IoU 判定
                # if compute_iou(p_box, g_box) > IOU_THRESHOLD:
                    if j not in matched_gt_indices:
                        matched_gt_indices.add(j)
                        is_match = True
                        # 一个预测框只能匹配一个GT，匹配到就跳出
                        break 
            
            if is_match:
                pred_matched[i] = True
            
        current_image_tp = sum(pred_matched)
        current_image_fp = len(valid_preds) - current_image_tp
        
        total_tp += current_image_tp
        total_fp += current_image_fp

    # ================= 结果计算 =================
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_gt + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print("\n" + "="*30)
    print("📊 评估结果 (Evaluation Results)")
    print("="*30)
    print(f"Total GT Targets : {total_gt}")
    print(f"Total Pred Boxes : {total_tp + total_fp} (Score Thresh: {CONF_THRESHOLD})")
    print(f"True Positives   : {total_tp}")
    print(f"False Positives  : {total_fp}")
    print("-" * 30)
    print(f"Precision        : {precision:.4f}")
    print(f"Recall (Pd)      : {recall:.4f}")
    print(f"F1-Score         : {f1_score:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. NUAA-SIRST 数据集 Mask 的文件夹路径
GT_FOLDER = 'dataset/NUAA-SIRST/masks'

# 2. Grounding DINO 的预测结果文件
PRED_JSON = 'demo_outputs/results_checkpoint0014/instances_results.json'

# 3. 阈值设置
CONF_THRESHOLD = 0.35  # 置信度阈值，低于这个分数的框丢弃

# ===========================================

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

def evaluate_pd_fa():
    """
    评估Pd (检测概率) 和 Fa (虚警率)
    Pd = 检测到的真实目标数 / 总真实目标数 (即召回率)
    Fa = 错误检测数 / 总预测目标数
    """
    preds = load_predictions(PRED_JSON)
    
    # 统计变量
    detected_targets = 0  # 成功检测到的真实目标数
    false_alarms = 0      # 虚警数
    total_gt_targets = 0  # 总真实目标数
    total_predictions = 0 # 总预测目标数
    
    # 获取GT文件夹下所有图片
    gt_files = [f for f in os.listdir(GT_FOLDER) if f.endswith(('.png', '.jpg', '.bmp'))]
    total_images = len(gt_files)
    
    print(f"开始评估Pd和Fa，共 {total_images} 张图片...")

    for filename in tqdm(gt_files):
        # 1. 获取 Ground Truth Boxes
        gt_path = os.path.join(GT_FOLDER, filename)
        gt_boxes = get_gt_boxes_from_mask(gt_path)
        total_gt_targets += len(gt_boxes)
        
        # 2. 获取 Prediction Boxes
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
        
        # 累加总预测数
        total_predictions += len(valid_preds)
        
        # 3. 匹配统计 (使用 Center-Hit)
        matched_gt_indices = set()
        
        # 标记哪些预测框是匹配成功的
        pred_matched = [False] * len(valid_preds)
        
        # 匹配预测框和真实框
        for i, p_box in enumerate(valid_preds):
            for j, g_box in enumerate(gt_boxes):
                if is_center_hit(p_box, g_box): 
                    if j not in matched_gt_indices:
                        matched_gt_indices.add(j)
                        pred_matched[i] = True
                        break 
        
        # 统计检测到的真实目标数
        detected_targets += len(matched_gt_indices)
        
        # 统计虚警数 (未匹配的预测框数量)
        false_alarms += len(valid_preds) - sum(pred_matched)

    # ================= 结果计算 =================
    # Pd (检测概率) = 检测到的真实目标数 / 总真实目标数
    pd = detected_targets / (total_gt_targets + 1e-6)
    
    # Fa (虚警率) = 错误检测数 / 总预测目标数
    fa = false_alarms / (total_predictions + 1e-6)
    
    print("\n" + "="*30)
    print("📊 Pd & Fa 评估结果")
    print("="*30)
    print(f"总图像数         : {total_images}")
    print(f"总真实目标数     : {total_gt_targets}")
    print(f"总预测目标数     : {total_predictions}")
    print(f"检测到的目标数   : {detected_targets}")
    print(f"虚警数           : {false_alarms}")
    print("-" * 30)
    print(f"Pd (检测概率)    : {pd:.4f}")
    print(f"Fa (虚警率)      : {fa:.4f}")
    print("="*30)
    
    return pd, fa

if __name__ == "__main__":
    evaluate_pd_fa()
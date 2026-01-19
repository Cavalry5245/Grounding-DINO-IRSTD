import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors

# ================= 配置区域 =================
# 1. 数据路径
GT_FOLDER = 'dataset/SIRST/masks'       # Mask 文件夹路径

# 2. 不同方法的预测结果 (方法名称: JSON文件路径)
METHODS = {
    'original': 'test_output/Swin_T/results_0.0/SIRST/instances_results.json',
}

# 3. 输出路径
OUTPUT_PLOT_PATH = 'figs/pr_curve/pr_curve_SIRST_orignal.png'

# 4. 评估标准
CENTER_HIT = True   # True: 使用中心点命中 (推荐用于红外小目标)
IOU_THRESH = 0.1    # 如果 CENTER_HIT=False, 则使用此 IoU 阈值

# 5. 图表标题
CHART_TITLE = 'Precision-Recall Curve (SIRST)'
# ===========================================

def get_gt_boxes_from_mask(mask_path):
    """从Mask读取GT框"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    gt_boxes = []
    for i in range(1, num_labels):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        gt_boxes.append([x, y, x + w, y + h])
    return gt_boxes

def is_match(pred_box, gt_box):
    """判定是否匹配 (支持 Center-Hit 或 IoU)"""
    if CENTER_HIT:
        # 中心点命中逻辑
        gt_cx = (gt_box[0] + gt_box[2]) / 2
        gt_cy = (gt_box[1] + gt_box[3]) / 2
        return (pred_box[0] <= gt_cx <= pred_box[2]) and (pred_box[1] <= gt_cy <= pred_box[3])
    else:
        # IoU 逻辑
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1], gt_box[1])
        x2 = min(pred_box[2], gt_box[2])
        y2 = min(pred_box[3], gt_box[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        union = area1 + area2 - intersection
        return (intersection / union) > IOU_THRESH if union > 0 else False

def compute_ap(recalls, precisions):
    """
    计算AP (Average Precision) - 使用全点插值法 (VOC2010+ 标准)
    """
    # 添加哨兵值
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # 确保precision是单调递减的 (从右向左取最大值)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 找到recall变化的点
    recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # 计算AP (曲线下面积)
    ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) * 
                precisions[recall_change_indices + 1])
    
    return ap
def calculate_pr_for_method(method_name, pred_json_path, gt_dict, total_gt_count):
    """为单个方法计算PR曲线数据"""
    print(f"加载 {pred_json_path} 的预测结果...")
    
    # 加载预测结果，展平为列表 (适配 COCO 格式)
    with open(pred_json_path, 'r') as f:
        data = json.load(f)
    
    # 创建 image_id 到 file_name 的映射
    id_to_filename = {}
    for img_info in data['images']:
        id_to_filename[img_info['id']] = img_info['file_name']
    
    # 展平所有预测框
    all_preds = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id in id_to_filename:
            filename = id_to_filename[image_id]
            # COCO bbox 格式是 [x, y, width, height]，需要转换为 [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]  # 转换为 [x1, y1, x2, y2]
            score = ann.get('score', 1.0)  # 如果没有 score 字段，默认为 1.0
            
            all_preds.append({
                'image_id': filename,
                'box': box,
                'score': score
            })
            
    # 按置信度从高到低排序
    all_preds.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"总共有 {len(all_preds)} 个预测框，正在逐个评估...")

    # 逐个计算 TP/FP
    tp_list = [] # 1 for TP, 0 for FP
    
    # 重置所有GT的匹配状态
    for img_key in gt_dict:
        gt_dict[img_key]['matched'] = [False] * len(gt_dict[img_key]['boxes'])
    
    # 遍历每一个预测框 (从最高分开始)
    for pred in all_preds:
        img_id = pred['image_id']
        p_box = pred['box']
        
        is_tp = 0 # 默认为 False Positive
        
        # 如果这张图里有GT
        if img_id in gt_dict:
            gt_info = gt_dict[img_id]
            
            # 尝试匹配该图中尚未被匹配的 GT
            best_match_idx = -1
            
            # 简单贪婪匹配：找到第一个能匹配上的未被占用的GT
            for idx, g_box in enumerate(gt_info['boxes']):
                if not gt_info['matched'][idx]: # 必须是还没被高分框匹配过的
                    if is_match(p_box, g_box):
                        best_match_idx = idx
                        break
            
            if best_match_idx != -1:
                is_tp = 1
                gt_info['matched'][best_match_idx] = True # 标记为已匹配
        
        tp_list.append(is_tp)

    # 计算累积的 Precision 和 Recall
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - x for x in tp_list])
    
    recalls = tp_cumsum / total_gt_count
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 计算 AP (Average Precision) - 曲线下面积
    # ap = np.trapz(precisions, recalls)
    ap = compute_ap(recalls, precisions)
    print(f"方法 {method_name} 的 AP: {ap:.4f}")

    # 计算 F1-score (F1-score 是 precision 和 recall 的调和平均数)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 添加小量避免除零
    
    # 找到 F1-score 最大值及其索引
    best_f1_idx = np.argmax(f1_scores)
    best_f1_score = f1_scores[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    best_precision = precisions[best_f1_idx]
    best_score_threshold = all_preds[best_f1_idx]['score'] if best_f1_idx < len(all_preds) else 0
    
    print(f"方法 {method_name} 的最佳 F1-score: {best_f1_score:.4f}")
    print(f"   对应阈值: {best_score_threshold:.4f}")
    print(f"   Recall: {best_recall:.4f}")
    print(f"   Precision: {best_precision:.4f}")

    return recalls, precisions, ap, best_f1_score, best_recall, best_precision, best_score_threshold

def load_ground_truth():
    """加载所有GT数据"""
    print("🚀 正在加载 Ground Truth 数据...")
    
    # 加载所有 GT，并按文件名索引
    gt_dict = {} 
    total_gt_count = 0
    gt_files = [f for f in os.listdir(GT_FOLDER) if f.endswith(('.png', '.jpg', '.bmp'))]
    
    print("加载 Ground Truth...")
    for filename in tqdm(gt_files):
        boxes = get_gt_boxes_from_mask(os.path.join(GT_FOLDER, filename))
        if len(boxes) > 0:
            # 用文件名作为Key (去除潜在的路径差异)
            key = filename 
            gt_dict[key] = {
                'boxes': boxes,
                'matched': [False] * len(boxes) # 记录每个GT是否已经被匹配过
            }
            total_gt_count += len(boxes)
            
    total_image_count = len(gt_files)
    print(f"总共有 {total_gt_count} 个 Ground Truth 目标，{total_image_count} 张图像。")
    
    return gt_dict, total_gt_count, total_image_count

def calculate_pr_curve():
    """计算并绘制多个方法的PR曲线对比图"""
    # 加载GT数据
    gt_dict, total_gt_count, total_image_count = load_ground_truth()
    
    # 为每个方法计算PR曲线
    pr_data = {}
    
    for method_name, pred_json_path in METHODS.items():
        if os.path.exists(pred_json_path):
            recalls, precisions, ap, best_f1_score, best_recall, best_precision, best_score_threshold = calculate_pr_for_method(
                method_name, pred_json_path, gt_dict.copy(), total_gt_count
            )
            pr_data[method_name] = {
                'recalls': recalls,
                'precisions': precisions,
                'ap': ap,
                'best_f1_score': best_f1_score,
                'best_recall': best_recall,
                'best_precision': best_precision,
                'best_score_threshold': best_score_threshold
            }
        else:
            print(f"警告: 找不到预测文件 {pred_json_path}，跳过方法 {method_name}")
    
    if not pr_data:
        print("错误: 没有找到任何有效的预测结果文件")
        return
    
    # 绘制对比PR曲线
    print("正在绘制PR曲线对比图...")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH) if os.path.dirname(OUTPUT_PLOT_PATH) else '.', exist_ok=True)
    
    # 设置图形
    plt.figure(figsize=(12, 8))
    
    # 获取一系列颜色用于区分不同方法
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    # 绘制每种方法的PR曲线
    for i, (method_name, data) in enumerate(pr_data.items()):
        color = colors[i % len(colors)]
        recalls = data['recalls']
        precisions = data['precisions']
        ap = data['ap']
        
        plt.plot(recalls, precisions, color=color, lw=2, label=f'{method_name} (AP = {ap:.4f})')
    
    # 美化图表
    plt.title(CHART_TITLE, fontsize=16)
    plt.xlabel('Recall (Sensivity)', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", fontsize=12)
    
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ PR曲线对比图已保存至: {OUTPUT_PLOT_PATH}")
    plt.show()
    
    # 打印最佳F1-score结果
    print("\n" + "="*50)
    print("📊 最佳 F1-score 结果")
    print("="*50)
    for method_name, data in pr_data.items():
        print(f"{method_name}:")
        print(f"  最佳 F1-score: {data['best_f1_score']:.4f}")
        print(f"  对应阈值: {data['best_score_threshold']:.4f}")
        print(f"  Recall: {data['best_recall']:.4f}")
        print(f"  Precision: {data['best_precision']:.4f}")
        print("-" * 30)
    print("="*50)

if __name__ == "__main__":
    calculate_pr_curve()
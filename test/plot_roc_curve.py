import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors

# ================= 配置区域 =================
# 1. 数据路径
GT_FOLDER = 'dataset/NUDT-SIRST/masks'       # Mask 文件夹路径

# 2. 不同方法的预测结果 (方法名称: JSON文件路径)
METHODS = {
    'original': 'demo_outputs/Swin_T/results_original/NUDT-SIRST/instances_results.json',
    'sirst_finetune': 'demo_outputs/Swin_T/results_sirst_finetune/NUDT-SIRST/instances_results.json',
    'sirst_lora_finetune': 'demo_outputs/Swin_T/results_sirst_lora_finetune/NUDT-SIRST/instances_results.json',
    # 可以添加更多方法进行对比，例如：
    # 'Method A': 'path/to/method_a_results.json',
    # 'Method B': 'path/to/method_b_results.json',
}

# 3. 输出路径
OUTPUT_ROC_PATH = 'roc_curve/roc_curve_comparison3.png'  # ROC曲线保存路径

# 4. 图表标题
CHART_TITLE = 'ROC Curve Comparison (NUDT-SIRST)'

# 5. 评估标准
CENTER_HIT = True   # True: 使用中心点命中 (推荐用于红外小目标)
IOU_THRESH = 0.1    # 如果 CENTER_HIT=False, 则使用此 IoU 阈值
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

def calculate_roc_for_method(pred_json_path, gt_dict, total_gt_count):
    """为单个方法计算ROC曲线数据"""
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

    # 逐个计算不同阈值下的 Pd 和 Fa
    pd_list = []  # Probability of Detection (Pd) 列表
    fa_list = []  # False Alarm (Fa) 列表
    
    # 初始化统计变量
    detected_targets = 0  # 已检测到的真实目标数
    false_alarms = 0      # 虚警数
    total_predictions = 0 # 总预测数
    
    # 重置所有GT的匹配状态
    for img_key in gt_dict:
        gt_dict[img_key]['matched'] = [False] * len(gt_dict[img_key]['boxes'])
    
    # 添加初始点 (0, 0)
    pd_list.append(0.0)
    fa_list.append(0.0)
    
    # 遍历每一个预测框 (从最高分开始)
    for i, pred in enumerate(all_preds):
        img_id = pred['image_id']
        p_box = pred['box']
        score = pred['score']
        
        # 增加总预测数
        total_predictions += 1
        
        # 更新匹配状态
        is_false_alarm = True  # 默认为虚警
        
        # 如果这张图里有GT
        if img_id in gt_dict:
            gt_info = gt_dict[img_id]
            
            # 尝试匹配该图中尚未被匹配的 GT
            best_match_idx = -1
            
            # 简单贪婪匹配：找到第一个能匹配上的未被占用的GT
            for idx, g_box in enumerate(gt_info['boxes']):
                if not gt_info['matched'][idx]:  # 必须是还没被高分框匹配过的
                    if is_center_hit(p_box, g_box):  # 使用与eval_pd_fa.py一致的匹配方法
                        best_match_idx = idx
                        break
            
            if best_match_idx != -1:
                is_false_alarm = False
                gt_info['matched'][best_match_idx] = True  # 标记为已匹配
                detected_targets += 1
        
        # 如果是虚警，则增加虚警计数
        if is_false_alarm:
            false_alarms += 1
            
        # 计算当前的Pd和Fa
        pd = detected_targets / (total_gt_count + 1e-6)
        fa = false_alarms / (total_predictions + 1e-6)
        
        # 每隔一定数量的预测框记录一次数据点，避免数据点过多
        if i % max(1, len(all_preds) // 1000) == 0 or i == len(all_preds) - 1:
            pd_list.append(pd)
            fa_list.append(fa)
    
    # 添加终点，形成直线延伸效果
    final_pd = detected_targets / (total_gt_count + 1e-6)
    final_fa = false_alarms / (total_predictions + 1e-6)
    
    # 添加额外的点以形成水平线延伸到x=1
    if final_fa < 1.0:
        pd_list.append(final_pd)
        fa_list.append(1.0)
    
    # 计算 AUC (Area Under Curve)
    auc = np.trapz(pd_list, fa_list)
    
    print(f"方法 {pred_json_path} 的 AUC: {abs(auc):.4f}")
    
    return fa_list, pd_list, abs(auc)

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

def calculate_roc_curve():
    """计算并绘制多个方法的ROC曲线对比图"""
    # 加载GT数据
    gt_dict, total_gt_count, total_image_count = load_ground_truth()
    
    # 为每个方法计算ROC曲线
    roc_data = {}
    for method_name, pred_json_path in METHODS.items():
        if os.path.exists(pred_json_path):
            fa_list, pd_list, auc = calculate_roc_for_method(pred_json_path, gt_dict.copy(), total_gt_count)
            roc_data[method_name] = {
                'fa_list': fa_list,
                'pd_list': pd_list,
                'auc': auc
            }
        else:
            print(f"警告: 找不到预测文件 {pred_json_path}，跳过方法 {method_name}")
    
    if not roc_data:
        print("错误: 没有找到任何有效的预测结果文件")
        return
    
    # 绘制对比ROC曲线
    print("正在绘制ROC曲线对比图...")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_ROC_PATH) if os.path.dirname(OUTPUT_ROC_PATH) else '.', exist_ok=True)
    
    # 设置图形
    plt.figure(figsize=(12, 8))
    
    # 获取一系列颜色用于区分不同方法
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    # 绘制每种方法的ROC曲线
    for i, (method_name, data) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        fa_list = data['fa_list']
        pd_list = data['pd_list']
        auc = data['auc']
        
        plt.plot(fa_list, pd_list, color=color, lw=2, label=f'{method_name} (AUC = {auc:.4f})')
    
    # 美化图表
    plt.title(CHART_TITLE, fontsize=16)
    plt.xlabel('False Alarm Rate (Fa)', fontsize=14)
    plt.ylabel('Probability of Detection (Pd)', fontsize=14)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=12)
    
    # 添加对角线参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Detector')

    plt.savefig(OUTPUT_ROC_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ ROC曲线对比图已保存至: {OUTPUT_ROC_PATH}")
    plt.show()

if __name__ == "__main__":
    calculate_roc_curve()
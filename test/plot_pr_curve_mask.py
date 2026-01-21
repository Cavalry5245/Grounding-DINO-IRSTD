import os
import cv2
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors

# ================= 配置区域 =================
# 1. 数据路径
GT_FOLDER = 'dataset/SIRST/masks'       # Mask 文件夹路径

# 2. 不同方法的预测结果 (方法名称: JSON文件路径)
METHODS = {
    'original': 'test_output/Swin_T/results_0121exp1/SIRST/instances_results.json',
}

# 3. 输出路径
OUTPUT_PLOT_PATH = 'figs/pr_curve/0121exp1/pr_curve_SIRST_0121exp1.png'
# 4. 评估标准
CENTER_HIT = True   # True: 使用中心点命中 (推荐用于红外小目标)
IOU_THRESH = 0.1    # 如果 CENTER_HIT=False, 则使用此 IoU 阈值

# 5. 图表标题
CHART_TITLE = 'Precision-Recall Curve (SIRST)'
# ===========================================


def get_gt_boxes_from_mask(mask_path):
    """从Mask读取GT框"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    gt_boxes = []
    for i in range(1, num_labels):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
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


def normalize_filename(filename):
    """统一文件名格式，去除路径，只保留文件名"""
    return os.path.basename(filename)


def compute_ap(recalls, precisions):
    # """
    # 计算AP (Average Precision) - 使用全点插值法 (VOC2010+ 标准)
    # """
    # # 添加哨兵值
    # recalls = np.concatenate([[0], recalls, [1]])
    # precisions = np.concatenate([[0], precisions, [0]])
    
    # # 确保precision是单调递减的 (从右向左取最大值)
    # for i in range(len(precisions) - 2, -1, -1):
    #     precisions[i] = max(precisions[i], precisions[i + 1])
    
    # # 找到recall变化的点
    # recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # # 计算AP (曲线下面积)
    # ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) * 
    #             precisions[recall_change_indices + 1])
    
    """
    计算 AP：既然 recalls 和 precisions 已经是插值过的单调序列，
    直接计算曲线下面积即可。
    """
    # 确保 recalls 是升序
    # 计算相邻 recall 之间的差异，乘以对应的 precision
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]

    return ap


def load_predictions(pred_json_path):
    """
    加载预测结果，支持多种JSON格式
    返回: list of dict, 每个dict包含 {'image_id': str, 'box': [x1,y1,x2,y2], 'score': float}
    """
    with open(pred_json_path, 'r') as f:
        data = json.load(f)
    
    all_preds = []
    
    # 格式1: COCO格式 {'images': [...], 'annotations': [...]}
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        # 创建 image_id 到 file_name 的映射
        id_to_filename = {}
        for img_info in data['images']:
            id_to_filename[img_info['id']] = normalize_filename(img_info['file_name'])
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id in id_to_filename:
                filename = id_to_filename[image_id]
                # COCO bbox 格式是 [x, y, width, height]
                x, y, w, h = ann['bbox']
                box = [x, y, x + w, y + h]
                score = ann.get('score', 1.0)
                
                all_preds.append({
                    'image_id': filename,
                    'box': box,
                    'score': score
                })
    
    # 格式2: COCO detection结果格式 - 列表形式 [{'image_id': int, 'bbox': [...], 'score': float}, ...]
    elif isinstance(data, list):
        for item in data:
            if 'bbox' in item and 'image_id' in item:
                x, y, w, h = item['bbox']
                box = [x, y, x + w, y + h]
                score = item.get('score', 1.0)
                # image_id 可能是整数，需要额外处理
                image_id = item['image_id']
                if isinstance(image_id, str):
                    image_id = normalize_filename(image_id)
                
                all_preds.append({
                    'image_id': image_id,
                    'box': box,
                    'score': score
                })
    
    # 格式3: 自定义格式 {'image_name': {'boxes': [...], 'scores': [...]}}
    elif isinstance(data, dict) and 'images' not in data:
        for img_name, pred_info in data.items():
            filename = normalize_filename(img_name)
            boxes = pred_info.get('boxes', [])
            scores = pred_info.get('scores', [1.0] * len(boxes))
            
            for box, score in zip(boxes, scores):
                all_preds.append({
                    'image_id': filename,
                    'box': box,
                    'score': score
                })
    
    return all_preds


def calculate_pr_for_method(method_name, pred_json_path, gt_dict_original, total_gt_count):
    """为单个方法计算PR曲线数据"""
    print(f"\n📊 处理方法: {method_name}")
    print(f"   加载预测文件: {pred_json_path}")
    
    # 深拷贝GT字典，避免影响其他方法的评估
    gt_dict = copy.deepcopy(gt_dict_original)
    
    # 加载预测结果
    all_preds = load_predictions(pred_json_path)
    
    if len(all_preds) == 0:
        print(f"   ⚠️ 警告: 没有找到任何预测结果")
        return None
    
    # 按置信度从高到低排序
    all_preds.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"   总共有 {len(all_preds)} 个预测框")

    # 逐个计算 TP/FP
    tp_list = []  # 1 for TP, 0 for FP
    scores_list = []  # 记录每个预测的分数
    
    # 遍历每一个预测框 (从最高分开始)
    for pred in all_preds:
        img_id = pred['image_id']
        p_box = pred['box']
        score = pred['score']
        
        is_tp = 0  # 默认为 False Positive
        
        # 如果这张图里有GT
        if img_id in gt_dict:
            gt_info = gt_dict[img_id]
            
            # 尝试匹配该图中尚未被匹配的 GT
            best_match_idx = -1
            
            # 简单贪婪匹配：找到第一个能匹配上的未被占用的GT
            for idx, g_box in enumerate(gt_info['boxes']):
                if not gt_info['matched'][idx]:  # 必须是还没被高分框匹配过的
                    if is_match(p_box, g_box):
                        best_match_idx = idx
                        break
            
            if best_match_idx != -1:
                is_tp = 1
                gt_info['matched'][best_match_idx] = True  # 标记为已匹配
        
        tp_list.append(is_tp)
        scores_list.append(score)

    # 计算累积的 Precision 和 Recall
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - x for x in tp_list])

    # 基础计算
    raw_recalls = tp_cumsum / total_gt_count
    raw_precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # recalls = tp_cumsum / total_gt_count
    # precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = np.concatenate(([0.0], raw_recalls))
    precisions = np.concatenate(([1.0], raw_precisions))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    
    # 计算 AP (Average Precision) - 曲线下面积
    # ap = np.trapz(precisions, recalls)
    ap = compute_ap(recalls, precisions)
    print(f"方法 {method_name} 的 AP: {ap:.4f}")

    # 计算 F1-score (F1-score 是 precision 和 recall 的调和平均数)
    # f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 添加小量避免除零
    f1_scores = 2 * (raw_precisions * raw_recalls) / (raw_precisions + raw_recalls + 1e-8)
    
    # 找到 F1-score 最大值及其索引
    best_f1_idx = np.argmax(f1_scores)
    best_f1_score = f1_scores[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    best_precision = precisions[best_f1_idx]
    best_score_threshold = scores_list[best_f1_idx]
    
    # 计算最终的统计信息
    total_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
    total_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
    total_fn = total_gt_count - total_tp
    
    print(f"   ✅ AP: {ap:.4f}")
    print(f"   ✅ 最佳 F1-score: {best_f1_score:.4f} (阈值={best_score_threshold:.4f})")
    print(f"   📈 TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

    return {
        'recalls': recalls,
        'precisions': precisions,
        'ap': ap,
        'best_f1_score': best_f1_score,
        'best_recall': best_recall,
        'best_precision': best_precision,
        'best_score_threshold': best_score_threshold,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'scores': scores_list
    }


def load_ground_truth():
    """加载所有GT数据"""
    print("🚀 正在加载 Ground Truth 数据...")
    
    if not os.path.exists(GT_FOLDER):
        raise FileNotFoundError(f"GT文件夹不存在: {GT_FOLDER}")
    
    gt_dict = {}
    total_gt_count = 0
    gt_files = [f for f in os.listdir(GT_FOLDER) if f.endswith(('.png', '.jpg', '.bmp'))]
    
    if len(gt_files) == 0:
        raise ValueError(f"GT文件夹中没有找到图像文件: {GT_FOLDER}")
    
    print(f"   找到 {len(gt_files)} 个mask文件")
    
    for filename in tqdm(gt_files, desc="   加载GT"):
        boxes = get_gt_boxes_from_mask(os.path.join(GT_FOLDER, filename))
        # 统一使用标准化的文件名作为key
        key = normalize_filename(filename)
        gt_dict[key] = {
            'boxes': boxes,
            'matched': [False] * len(boxes)
        }
        total_gt_count += len(boxes)
    
    # 统计有目标的图像数量
    images_with_targets = sum(1 for v in gt_dict.values() if len(v['boxes']) > 0)
    
    print(f"   ✅ 总共 {total_gt_count} 个GT目标")
    print(f"   ✅ {images_with_targets}/{len(gt_files)} 张图像包含目标")
    
    return gt_dict, total_gt_count, len(gt_files)


def calculate_pr_curve():
    """计算并绘制多个方法的PR曲线对比图"""
    # 加载GT数据
    gt_dict, total_gt_count, total_image_count = load_ground_truth()
    
    if total_gt_count == 0:
        print("❌ 错误: 没有找到任何GT目标")
        return
    
    # 为每个方法计算PR曲线
    pr_data = {}
    
    for method_name, pred_json_path in METHODS.items():
        if os.path.exists(pred_json_path):
            result = calculate_pr_for_method(
                method_name, pred_json_path, gt_dict, total_gt_count
            )
            if result is not None:
                pr_data[method_name] = result
        else:
            print(f"⚠️ 警告: 找不到预测文件 {pred_json_path}，跳过方法 {method_name}")
    
    if not pr_data:
        print("❌ 错误: 没有找到任何有效的预测结果文件")
        return
    
    # 绘制对比PR曲线
    print("\n🎨 正在绘制PR曲线对比图...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_PLOT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用更好的配色方案
    colors = plt.cm.tab10.colors
    
    # 按AP排序方法（从高到低）
    sorted_methods = sorted(pr_data.items(), key=lambda x: x[1]['ap'], reverse=True)
    
    # 绘制每种方法的PR曲线
    for i, (method_name, data) in enumerate(sorted_methods):
        color = colors[i % len(colors)]
        recalls = data['recalls']
        precisions = data['precisions']
        ap = data['ap']
        f1 = data['best_f1_score']
        
        # 绘制曲线
        ax.plot(recalls, precisions, color=color, lw=2, 
                label=f'{method_name} (AP={ap:.4f}, F1={f1:.4f})')
        
    
    # 美化图表
    ax.set_title(CHART_TITLE, fontsize=16)
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    
    # 添加对角线参考
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ PR曲线对比图已保存至: {OUTPUT_PLOT_PATH}")
    plt.show()
    
    # 打印详细结果表格
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)
    print(f"{'方法名称':<25} {'AP':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'阈值':<10}")
    print("-" * 80)
    
    for method_name, data in sorted_methods:
        print(f"{method_name:<25} {data['ap']:<10.4f} {data['best_f1_score']:<10.4f} "
              f"{data['best_precision']:<12.4f} {data['best_recall']:<10.4f} "
              f"{data['best_score_threshold']:<10.4f}")
    
    print("-" * 80)
    print(f"评估标准: {'中心点命中' if CENTER_HIT else f'IoU > {IOU_THRESH}'}")
    print(f"GT总数: {total_gt_count} | 图像数: {total_image_count}")
    print("=" * 80)
    
    # 保存结果到JSON
    results_json_path = OUTPUT_PLOT_PATH.replace('.png', '_results.json')
    results_to_save = {
        'config': {
            'center_hit': CENTER_HIT,
            'iou_thresh': IOU_THRESH,
            'total_gt_count': total_gt_count,
            'total_image_count': total_image_count
        },
        'methods': {}
    }
    
    for method_name, data in pr_data.items():
        results_to_save['methods'][method_name] = {
            'ap': float(data['ap']),
            'best_f1_score': float(data['best_f1_score']),
            'best_precision': float(data['best_precision']),
            'best_recall': float(data['best_recall']),
            'best_score_threshold': float(data['best_score_threshold']),
            'total_tp': int(data['total_tp']),
            'total_fp': int(data['total_fp']),
            'total_fn': int(data['total_fn'])
        }
    
    with open(results_json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"✅ 评估结果已保存至: {results_json_path}")


if __name__ == "__main__":
    calculate_pr_curve()
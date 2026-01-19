# coco_validation.py
import json
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle
import argparse


def validate_coco_structure(coco_json_path):
    """验证COCO JSON文件的基本结构"""
    print("=== 验证COCO文件结构 ===")

    # 检查文件是否存在
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"COCO JSON文件不存在: {coco_json_path}")

    # 加载JSON文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 检查必需字段
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            raise ValueError(f"缺少必需字段: {key}")
        print(f"✓ 包含字段: {key} (数量: {len(coco_data[key])})")

    # 验证images字段
    for img in coco_data['images'][:3]:  # 检查前3个样本
        required_img_fields = ['id', 'file_name', 'width', 'height']
        for field in required_img_fields:
            if field not in img:
                raise ValueError(f"images中缺少字段: {field}")

    # 验证annotations字段
    for ann in coco_data['annotations'][:3]:  # 检查前3个样本
        required_ann_fields = ['image_id', 'category_id', 'bbox', 'area', 'id']
        for field in required_ann_fields:
            if field not in ann:
                raise ValueError(f"annotations中缺少字段: {field}")

    # 验证categories字段
    for cat in coco_data['categories']:
        required_cat_fields = ['id', 'name']
        for field in required_cat_fields:
            if field not in cat:
                raise ValueError(f"categories中缺少字段: {field}")

    print("✓ COCO文件结构验证通过\n")
    return coco_data


def validate_with_coco_api(coco_json_path):
    """使用COCO API验证数据集"""
    print("=== 使用COCO API验证 ===")

    # 加载COCO数据集
    coco = COCO(coco_json_path)

    # 获取基本信息
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cat_ids = coco.getCatIds()

    print(f"✓ 图像数量: {len(img_ids)}")
    print(f"✓ 标注数量: {len(ann_ids)}")
    print(f"✓ 类别数量: {len(cat_ids)}")

    # 显示类别信息
    cats = coco.loadCats(cat_ids)
    print("✓ 类别列表:")
    for cat in cats:
        print(f"  - {cat['name']} (ID: {cat['id']})")

    # 统计每个类别的实例数量
    print("\n✓ 各类别实例数量:")
    for cat_id in cat_ids:
        anns = coco.getAnnIds(catIds=[cat_id])
        cat_info = coco.loadCats([cat_id])[0]
        print(f"  - {cat_info['name']}: {len(anns)} 个实例")

    print("✓ COCO API验证通过\n")
    return coco


def validate_image_files(coco, image_dir):
    """验证图像文件是否存在"""
    print("=== 验证图像文件 ===")

    img_ids = coco.getImgIds()
    missing_files = []

    for img_id in img_ids[:10]:  # 只检查前10个文件以节省时间
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_files.append(img_info['file_name'])
            print(f"✗ 缺失文件: {img_info['file_name']}")
        else:
            print(f"✓ 文件存在: {img_info['file_name']}")

    if missing_files:
        print(f"\n警告: 发现 {len(missing_files)} 个缺失文件")
    else:
        print("✓ 所有检查的图像文件均存在")

    print()


def visualize_annotations(coco, image_dir, num_samples=3, output_dir="./visualization"):
    """可视化部分标注结果"""
    print("=== 可视化标注结果 ===")

    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建可视化输出目录: {output_dir}")

    # 按文件名字典序排序
    imgs = coco.loadImgs(coco.getImgIds())
    sorted_imgs = sorted(imgs, key=lambda x: x['file_name'])

    # 处理可视化所有样本的情况
    if num_samples == float('inf') or num_samples <= 0 or num_samples >= len(sorted_imgs):
        sample_imgs = sorted_imgs
        print(f"将可视化所有 {len(sorted_imgs)} 个样本")
    else:
        sample_imgs = sorted_imgs[:num_samples]
        print(f"将可视化 {len(sample_imgs)} 个样本")

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    for i, img_info in enumerate(sample_imgs):
        img_path = os.path.join(image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"跳过 {img_info['file_name']} (文件不存在)")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取该图像的所有标注
        ann_ids = coco.getAnnIds(imgIds=[img_info['id']])
        anns = coco.loadAnns(ann_ids)

        # 绘制图像和边界框
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        for ann in anns:
            bbox = ann['bbox']
            category_name = cat_id_to_name[ann['category_id']]

            # 绘制边界框
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # 添加类别标签
            ax.text(bbox[0], bbox[1] - 5, category_name,
                    color='red', fontsize=10, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))

        ax.set_title(f"图像: {img_info['file_name']}")
        ax.axis('off')

        # 保存可视化结果，使用原始文件名作为标识
        output_filename = f"visualization_{img_info['file_name']}"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 可视化结果已保存至: {output_path}")

    print()


def print_dataset_statistics(coco):
    """打印数据集统计信息"""
    print("=== 数据集统计信息 ===")

    # 基本统计
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cat_ids = coco.getCatIds()

    print(f"总图像数: {len(img_ids)}")
    print(f"总标注数: {len(ann_ids)}")
    print(f"总类别数: {len(cat_ids)}")

    # 图像尺寸统计
    imgs = coco.loadImgs(img_ids)
    widths = [img['width'] for img in imgs]
    heights = [img['height'] for img in imgs]

    print(f"图像宽度范围: {min(widths)} - {max(widths)}")
    print(f"图像高度范围: {min(heights)} - {max(heights)}")

    # 类别分布
    print("\n类别分布:")
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
    for cat_id in cat_ids:
        anns = coco.getAnnIds(catIds=[cat_id])
        cat_name = cat_id_to_name[cat_id]
        print(f"  {cat_name}: {len(anns)} 个实例")

    # 标注统计
    anns = coco.loadAnns(ann_ids)
    areas = [ann['area'] for ann in anns]
    print(f"\n标注面积统计:")
    print(f"  最小面积: {min(areas):.2f}")
    print(f"  最大面积: {max(areas):.2f}")
    print(f"  平均面积: {np.mean(areas):.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description='验证COCO格式数据集')
    parser.add_argument('--json-path', type=str, default="demo_outputs/Swin_T/results_sirst_finetune/NUDT-SIRST/instances_results.json", required=False, help='COCO JSON文件路径')
    parser.add_argument('--image-dir', type=str, default="dataset/NUDT-SIRST/images", required=False, help='图像文件目录')
    parser.add_argument('--visualize', action='store_true', help='是否进行可视化')
    parser.add_argument('--samples', type=int, default=float('inf'), help='可视化样本数量')
    parser.add_argument('--output-dir', type=str, default="demo_outputs/Swin_T/results_sirst_finetune/NUDT-SIRST/visualization", help='可视化结果保存目录')

    args = parser.parse_args()

    try:
        # 1. 验证COCO结构
        coco_data = validate_coco_structure(args.json_path)

        # 2. 使用COCO API验证
        coco = validate_with_coco_api(args.json_path)

        # 3. 验证图像文件
        validate_image_files(coco, args.image_dir)

        # 4. 打印统计信息
        print_dataset_statistics(coco)

        # 5. 可视化(可选)
        if args.visualize:
            visualize_annotations(coco, args.image_dir, args.samples, args.output_dir)

        print("=== 验证完成 ===")
        print("✓ 数据集验证通过!")

    except Exception as e:
        print(f"验证失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()

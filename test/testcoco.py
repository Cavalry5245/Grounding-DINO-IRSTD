# 推理结果，包含标注图和coco标准
import cv2
import os
from pathlib import Path
import torch
import torchvision
import json
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import supervision as sv
from torchvision.ops import box_convert


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Grounding DINO 推理脚本')
    parser.add_argument('--input-folder', type=str, default="/media/sisu/X/hc/projects/datasets/IRDST/real/images/3",
                        help='输入文件夹路径')
    parser.add_argument('--output-folder', type=str, default="test_output/Swin_T/results_0114exp2/IRDST",
                        help='输出文件夹路径')
    parser.add_argument('--config', type=str, default="config/cfg_odvg.py",
                        help='模型配置文件路径')
    parser.add_argument('--weights', type=str, default="training_output_lora/lora_1230exp1/merged_model.pth",
                        help='模型权重文件路径')
    parser.add_argument('--text-prompt', type=str, default="Infrared small target",
                        help='文本提示')
    parser.add_argument('--box-threshold', type=float, default=0.001,
                        help='边界框阈值')
    parser.add_argument('--text-threshold', type=float, default=0.85,
                        help='文本阈值')
    parser.add_argument('--nms-threshold', type=float, default=0.3,
                        help='NMS阈值')
    parser.add_argument('--min-area', type=int, default=0,
                        help='最小面积阈值')
    parser.add_argument('--max-area', type=int, default=1000,
                        help='最大面积阈值')
    parser.add_argument('--save-images', action='store_true',
                        help='保存可视化图片 (默认: False)')
    parser.add_argument('--no-save-images', dest='save_images', action='store_false',
                        help='不保存可视化图片')
    parser.set_defaults(save_images=False)
    
    return parser.parse_args()


# 导入当前项目的模型加载功能
from groundingdino.util.inference import load_model, load_image, predict, annotate


def filter_boxes_by_area(boxes, logits, phrases, min_area, max_area, image_shape):
    """根据面积筛选检测框"""
    if len(boxes) == 0:
        return boxes, logits, phrases

    h, w = image_shape[:2]
    # 计算检测框面积（boxes格式为cxcywh，已归一化）
    areas = boxes[:, 2] * boxes[:, 3] * h * w  # width * height
    area_mask = (areas >= min_area) & (areas <= max_area)
    indices = torch.where(area_mask)[0]

    filtered_boxes = boxes[indices]
    filtered_logits = logits[indices]
    filtered_phrases = [phrases[i] for i in indices]

    return filtered_boxes, filtered_logits, filtered_phrases


def apply_nms(boxes, logits, phrases, nms_threshold, image_shape):
    """应用非极大值抑制"""
    if len(boxes) == 0:
        return boxes, logits, phrases

    h, w = image_shape[:2]
    # 将归一化的cxcywh转换为xyxy格式，再转换为像素坐标
    boxes_xyxy = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy')
    boxes_xyxy[:, [0, 2]] *= w  # x坐标
    boxes_xyxy[:, [1, 3]] *= h  # y坐标

    nms_indices = torchvision.ops.nms(boxes_xyxy, logits, nms_threshold)

    filtered_boxes = boxes[nms_indices]
    filtered_logits = logits[nms_indices]
    filtered_phrases = [phrases[i] for i in nms_indices]

    return filtered_boxes, filtered_logits, filtered_phrases


def convert_to_coco_bbox(box, image_shape):
    """
    将 Grounding DINO 的 cxcywh 归一化格式转换为 COCO 的 xywh 像素格式
    box: tensor([cx, cy, w, h]) in [0,1]
    image_shape: (H, W, ...)
    Returns: [x, y, width, height] in pixels (float -> int)
    """
    h, w = image_shape[:2]
    cx, cy, bw, bh = box

    # 转为 float 再计算，避免 Tensor 运算后仍是 Tensor
    cx, cy, bw, bh = cx.item(), cy.item(), bw.item(), bh.item()

    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    width = bw * w
    height = bh * h
    return [round(x), round(y), round(width), round(height)]


def process_image(image_path, output_path, image_id, model, text_prompt, box_threshold, text_threshold, 
                  nms_threshold, min_area, max_area, save_images):
    """处理单张图片，并记录 COCO 格式结果"""
    global annotation_id
    try:
        # 加载图像
        image_source, image = load_image(str(image_path))
        h, w = image_source.shape[:2]

        # 运行模型预测
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # 筛选 + NMS
        boxes, logits, phrases = filter_boxes_by_area(
            boxes, logits, phrases, min_area, max_area, image_source.shape
        )
        boxes, logits, phrases = apply_nms(boxes, logits, phrases, nms_threshold, image_source.shape)

        # 添加图像信息到 COCO
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": w,
            "height": h,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # 添加每个检测框为 annotation
        for box, logit in zip(boxes, logits):
            bbox = convert_to_coco_bbox(box, image_source.shape)
            area = bbox[2] * bbox[3]
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "segmentation": [],  # 可选：可后续用 mask 扩展
                "iscrowd": 0,
                "score": float(logit)  # COCO 官方不包含 score，但很多工具支持
            })
            annotation_id += 1

        # 只有在用户指定保存图片时才进行可视化保存
        if save_images:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"已处理: {image_path.name} -> {output_path}")
        else:
            print(f"已处理: {image_path.name}")

    except Exception as e:
        print(f"处理 {image_path.name} 时出错: {e}")


def main():
    args = parse_args()
    
    # 配置参数
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    nms_threshold = args.nms_threshold  # NMS阈值
    min_area = args.min_area   # 最小面积阈值
    max_area = args.max_area  # 最大面积阈值
    input_folder = args.input_folder  # 输入文件夹路径
    output_folder = args.output_folder  # 输出文件夹路径
    save_images = args.save_images  # 是否保存可视化图片

    # 加载模型
    model = load_model(args.config, args.weights)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # COCO 相关变量
    global coco_output, annotation_id
    coco_output = {
        "info": {
            "description": "Grounding DINO detection results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "anomaly", "supercategory": "object"}]
    }

    annotation_id = 1  # 全局 annotation ID 计数器

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # 无论是否保存图片，都需要创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"错误: 输入文件夹 {input_folder} 不存在")
        return

    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"在 {input_folder} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")
    print(f"保存图片: {'是' if save_images else '否'}")

    for idx, image_file in enumerate(image_files, start=1):
        output_file = output_path / f"{image_file.name}"
        process_image(image_file, output_file, image_id=idx, model=model, 
                     text_prompt=text_prompt, box_threshold=box_threshold, 
                     text_threshold=text_threshold, nms_threshold=nms_threshold, 
                     min_area=min_area, max_area=max_area, save_images=save_images)

    # 保存 COCO JSON
    coco_json_path = output_path / "instances_results.json"
    with open(coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"COCO 格式结果已保存至: {coco_json_path}")
    print("所有图片处理完成!")


if __name__ == "__main__":
    main()
import json
import os
import argparse
from tqdm import tqdm

def split_val_coco(annotation_path, val_images_dir, output_json):
    """
    从 COCO 标准标注文件中提取验证集图片的标注
    
    Args:
        annotation_path: 包含所有图片标注的 COCO JSON 文件路径
        val_images_dir: 验证集图片所在文件夹路径（包含图片文件名）
        output_json: 输出的验证集标注文件路径
    """
    # 读取整个 COCO 标注文件
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    # 获取验证集图片文件名列表（只取文件名，不含路径）
    val_image_files = [f for f in os.listdir(val_images_dir) 
                      if os.path.isfile(os.path.join(val_images_dir, f))]
    
    print(f"找到 {len(val_image_files)} 张验证集图片")
    
    # 创建验证集图片 ID 集合
    val_image_ids = set()
    
    # 从 COCO 图片信息中匹配验证集图片
    for img in coco_data['images']:
        if img['file_name'] in val_image_files:
            val_image_ids.add(img['id'])
    
    print(f"匹配到 {len(val_image_ids)} 张验证集图片")
    
    # 提取验证集对应的标注
    val_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in val_image_ids
    ]
    
    print(f"提取到 {len(val_annotations)} 个验证集标注")
    
    # 创建新的 COCO 格式验证集数据
    val_coco_data = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "images": [img for img in coco_data["images"] if img["id"] in val_image_ids],
        "annotations": val_annotations,
        "categories": coco_data["categories"]
    }
    
    # 保存到输出文件
    with open(output_json, 'w') as f:
        json.dump(val_coco_data, f, indent=4)
    
    print(f"✅ 验证集标注已保存至: {output_json}")
    print(f"✅ 验证集图片数量: {len(val_coco_data['images'])}")
    print(f"✅ 验证集标注数量: {len(val_coco_data['annotations'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从COCO标注中提取验证集')
    parser.add_argument('--annotation', type=str, default='data/sirst/annotations/annotations.json',
                        help='包含所有图片标注的COCO JSON文件路径')
    parser.add_argument('--val_dir', type=str, default='data/sirst/train',
                        help='验证集图片文件夹路径')
    parser.add_argument('--output', type=str, default='data/sirst/annotations/train.json',
                        help='输出验证集标注文件路径 ')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.annotation):
        raise FileNotFoundError(f"标注文件未找到: {args.annotation}")
    
    if not os.path.exists(args.val_dir):
        raise FileNotFoundError(f"验证集图片文件夹未找到: {args.val_dir}")
    
    # 执行分割
    split_val_coco(args.annotation, args.val_dir, args.output)
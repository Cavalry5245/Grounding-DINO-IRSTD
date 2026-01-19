# GroundingDINO ONNX Export Guide

本指南介绍如何将 GroundingDINO 模型导出为 ONNX 格式，以及如何使用导出的模型进行推理。

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [导出模型](#导出模型)
- [ONNX 推理](#onnx-推理)
- [常见问题](#常见问题)

---

## 功能特性

- ✅ 支持动态文本输入（不限于固定提示）
- ✅ 支持动态图像尺寸
- ✅ 支持 FP16 精度导出
- ✅ 自动验证导出的 ONNX 模型
- ✅ 提供完整的推理脚本

---

## 环境要求

### 必需依赖

```bash
pip install onnx onnxruntime onnxruntime-gpu  # 根据你的环境选择
```

### PyTorch 版本

建议使用 PyTorch >= 1.12 以获得更好的 ONNX 支持。

---

## 导出模型

### 基本用法

```bash
# 使用默认参数导出
./export_onnx.sh

# 指定配置和检查点
./export_onnx.sh config/cfg_odvg.py weights/groundingdino_swint_ogc.pth models/groundingdino.onnx

# 自定义图像尺寸
./export_onnx.sh config/cfg_odvg.py weights/groundingdino_swint_ogc.pth models/groundingdino.onnx 800 1200
```

### 高级选项

直接使用 Python 脚本以获得更多控制：

```bash
python tools/export_onnx.py \
    --config config/cfg_odvg.py \
    --checkpoint weights/groundingdino_swint_ogc.pth \
    --output models/groundingdino.onnx \
    --image-height 800 \
    --image-width 1200 \
    --max-text-len 256 \
    --opset 14 \
    --half  # 使用 FP16 精度
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config, -c` | 模型配置文件路径 | 必需 |
| `--checkpoint, -p` | 模型检查点路径 | 必需 |
| `--output, -o` | 输出 ONNX 文件路径 | 必需 |
| `--image-height` | 虚拟输入图像高度 | 800 |
| `--image-width` | 虚拟输入图像宽度 | 1200 |
| `--max-text-len` | 最大文本长度 | 256 |
| `--opset` | ONNX opset 版本 | 14 |
| `--half` | 使用 FP16 精度导出 | False |

---

## ONNX 推理

### 基本用法

```bash
python tools/inference_onnx.py \
    --model models/groundingdino.onnx \
    --image path/to/image.jpg \
    --text "infrared small target" \
    --output output_onnx.jpg
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m` | ONNX 模型路径 | 必需 |
| `--image, -i` | 输入图像路径 | 必需 |
| `--text, -t` | 文本提示 | 必需 |
| `--output, -o` | 输出图像路径 | output_onnx.jpg |
| `--box-threshold` | 检测框置信度阈值 | 0.3 |
| `--text-threshold` | 文本 token 阈值 | 0.25 |
| `--tokenizer` | Tokenizer 路径 | bert-base-uncased |
| `--max-text-len` | 最大文本长度 | 256 |
| `--target-size` | 目标图像尺寸 | 800 |

---

## 输入/输出格式

### 输入

导出的 ONNX 模型接受以下输入：

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `images` | (B, 3, H, W) | float32 | 归一化的图像张量 |
| `input_ids` | (B, max_text_len) | int64 | 文本 token IDs |
| `attention_mask` | (B, max_text_len) | int64 | 注意力掩码 |
| `token_type_ids` | (B, max_text_len) | int64 | Token 类型 IDs |
| `position_ids` | (B, max_text_len) | int64 | 位置 IDs |
| `text_self_attention_masks` | (B, max_text_len, max_text_len) | bool | 文本自注意力掩码 |

### 输出

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `pred_logits` | (B, num_queries, text_dim) | float32 | 预测 logits |
| `pred_boxes` | (B, num_queries, 4) | float32 | 预测边界框 (cx, cy, w, h) |

---

## 常见问题

### Q1: 导出时出现 "CUDA out of memory" 错误

**解决方案**：减小虚拟输入的图像尺寸，或使用 CPU 导出：

```bash
# 使用 CPU 导出
CUDA_VISIBLE_DEVICES="" python tools/export_onnx.py ...
```

### Q2: ONNX 推理速度很慢

**解决方案**：
1. 确保安装了 GPU 版本的 onnxruntime
2. 使用 FP16 精度导出（`--half`）
3. 考虑使用 TensorRT 优化

### Q3: 文本编码需要与训练时一致

**解决方案**：确保使用与训练相同的 tokenizer 路径。对于你的项目，应该使用：

```bash
--tokenizer /media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased
```

### Q4: 如何验证导出的模型是否正确？

导出脚本会自动验证模型。你也可以手动验证：

```python
import onnx
model = onnx.load("groundingdino.onnx")
onnx.checker.check_model(model)
print("Model is valid!")
```

### Q5: ONNX 模型文件很大怎么办？

可以尝试以下优化：
1. 使用 FP16 精度（`--half`）
2. 使用 ONNX Simplifier 进行模型简化
3. 使用 TensorRT 进行进一步优化

---

## 示例工作流

```bash
# 1. 导出模型
./export_onnx.sh config/cfg_odvg.py weights/groundingdino_swint_ogc.pth models/groundingdino.onnx

# 2. 运行推理
python tools/inference_onnx.py \
    --model models/groundingdino.onnx \
    --image dataset/test.jpg \
    --text "infrared small target" \
    --tokenizer /media/sisu/X/hc/projects/Open-GroundingDino/weights/bert-base-uncased \
    --output results/detection.jpg
```

---

## 相关文件

- [tools/export_onnx.py](../tools/export_onnx.py) - ONNX 导出工具
- [tools/inference_onnx.py](../tools/inference_onnx.py) - ONNX 推理工具
- [export_onnx.sh](../export_onnx.sh) - 导出便捷脚本
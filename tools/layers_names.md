根据您提供的模型参数列表，GroundingDINO模型中的Attention层名称包括：

## Transformer相关Attention层：

### 1. Encoder自注意力层
- `transformer.encoder.layers.{0-5}.self_attn` - 每层包含：
  - [sampling_offsets](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/ops/modules/ms_deform_attn.py#L0-L0) - 采样偏移
  - [attention_weights](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/ops/modules/ms_deform_attn.py#L0-L0) - 注意力权重
  - [value_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/ops/modules/ms_deform_attn.py#L0-L0) - 值投影
  - [output_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/ops/modules/ms_deform_attn.py#L0-L0) - 输出投影

### 2. 文本编码器层
- `transformer.encoder.text_layers.{0-5}.self_attn.out_proj` - 输出投影层

### 3. 融合层
- `transformer.encoder.fusion_layers.{0-5}.attn` - 每层包含：
  - [v_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 视觉投影
  - [l_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 语言投影
  - [values_v_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 视觉值投影
  - [values_l_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 语言值投影
  - [out_v_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 视觉输出投影
  - [out_l_proj](file:///media/sisu/X/hc/projects/Open-GroundingDino/models/GroundingDINO/fuse_modules.py#L0-L0) - 语言输出投影

### 4. Decoder注意力层
- `transformer.decoder.layers.{0-5}.cross_attn` - 交叉注意力（每层包含sampling_offsets, attention_weights, value_proj, output_proj）
- `transformer.decoder.layers.{0-5}.ca_text.out_proj` - 文本交叉注意力输出投影
- `transformer.decoder.layers.{0-5}.self_attn.out_proj` - 自注意力输出投影

## BERT文本编码器Attention层：

### 12层BERT编码层
- `bert.encoder.layer.{0-11}.attention.self` - 每层包含：
  - `query` - 查询投影
  - `key` - 键投影
  - [value](file:///media/sisu/X/hc/projects/Open-GroundingDino/util/misc.py#L83-L84) - 值投影
- `bert.encoder.layer.{0-11}.attention.output.dense` - 注意力输出投影

## Backbone视觉编码器Attention层：

### Swin Transformer块
- `backbone.0.layers.{0-3}.blocks.{0-5}.attn.qkv` - QKV投影
- `backbone.0.layers.{0-3}.blocks.{0-5}.attn.proj` - 注意力输出投影

这些是GroundingDINO模型中所有的Attention相关层名称，它们负责模型中不同模块间的特征交互和信息融合。
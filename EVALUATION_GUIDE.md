# 批量评估消融实验结果

## 📁 创建的文件

1. **[run_all_evaluations.sh](file:///media/sisu/X/hc/projects/Open-GroundingDino/run_all_evaluations.sh)** - Bash版本
2. **[run_all_evaluations.py](file:///media/sisu/X/hc/projects/Open-GroundingDino/run_all_evaluations.py)** - Python版本

---

## 🚀 快速开始

### 方法1: Bash脚本（推荐）

```bash
# 评估 Group 0 (Baseline)
bash run_all_evaluations.sh 0

# 评估 Group 1 (HF-LoRA 消融)
bash run_all_evaluations.sh 1

# 评估 Group 2 (提示词库消融)
bash run_all_evaluations.sh 2

# 评估 Group 3 (组合消融)
bash run_all_evaluations.sh 3

# 评估 Group 4 (超参数消融)
bash run_all_evaluations.sh 4

# 评估所有实验
bash run_all_evaluations.sh all
```

### 方法2: Python脚本（更灵活）

```bash
# 评估指定组
python run_all_evaluations.py --group 0
python run_all_evaluations.py --group all

# 评估单个实验
python run_all_evaluations.py --exp exp0_1_full_finetune

# 指定输出目录
python run_all_evaluations.py --group all --output_dir ./eval_output/ablation
```

---

## 📊 评估结果

所有评估结果将保存在 `./eval_output/ablation/` 目录下，按组分类：

```
eval_output/ablation/
├── group0/
│   ├── exp0_1_full_finetune/
│   │   ├── metrics.json
│   │   ├── metrics.txt
│   │   └── predictions.json
│   └── exp0_2_standard_lora/
│       ├── metrics.json
│       ├── metrics.txt
│       └── predictions.json
├── group1/
│   ├── exp1_1_hf_lora_qkv/
│   ├── exp1_2_hf_lora_fc1/
│   ├── exp1_3_hf_lora_fc2/
│   ├── exp1_4_hf_lora_qkv_fc1/
│   └── exp1_5_hf_lora_full/
├── group2/
│   ├── exp2_1_prompt_generic/
│   ├── exp2_2_prompt_gen_app/
│   ├── exp2_3_prompt_gen_phy/
│   ├── exp2_4_prompt_3cat/
│   └── exp2_5_prompt_5cat/
├── group3/
│   ├── exp3_1_hf_prompt_3cat/
│   └── exp3_2_hf_prompt_5cat/
└── group4/
    ├── exp4_1_r8/
    ├── exp4_2_r32/
    ├── exp4_3_alpha16/
    └── exp4_4_alpha64/
```

---

## 📋 评估指标

每个实验的 `metrics.json` 包含以下指标：

| 指标 | 说明 |
|------|------|
| `P` | Precision (精确率) |
| `R` | Recall (召回率) |
| `F1` | F1 Score |
| `mAP@0.5` | IoU=0.5时的平均精度 |
| `mAP@0.5:0.95` | IoU从0.5到0.95的平均精度 |
| `PD` | Probability of Detection (检测概率) |
| `FA` | False Alarm (虚警率) |
| `TP` | True Positives |
| `FP_box` | False Positives (box) |
| `FN` | False Negatives |

---

## 🔧 配置说明

### 数据集配置
- **测试图像**: `data/IRSTD-1k/images/test`
- **标注文件**: `data/IRSTD-1k/annotations/test.json`
- **文本提示**: "Infrared small target"

### 模型配置
- **配置文件**: `config/cfg_odvg.py`
- **预训练权重**: `weights/groundingdino_swint_ogc.pth`

### 评估阈值
- **box_threshold**: 0.001
- **text_threshold**: 0.85
- **nms_threshold**: 0.5

---

## ⚠️ 注意事项

1. **训练完成**: 确保所有消融实验训练完成，并且存在 `lora_checkpoint_best` 目录
2. **GPU资源**: 评估需要GPU，建议一次运行一个组
3. **磁盘空间**: 评估结果会占用一定磁盘空间，确保有足够空间
4. **超时设置**: Python脚本默认1小时超时，可根据需要调整

---

## 📈 收集结果

评估完成后，使用结果收集脚本汇总所有结果：

```bash
python ablation_results_collector.py --output_dir ./eval_output/ablation --save_csv --save_md
```

这将生成：
- `eval_output/ablation/ablation_results.csv` - CSV格式结果
- `eval_output/ablation/ablation_results.md` - Markdown格式结果

---

## 🎯 实验清单

### Group 0: Baseline (2个实验)
- [ ] Exp-0.1: Full Fine-tuning
- [ ] Exp-0.2: Standard LoRA

### Group 1: HF-LoRA 消融 (5个实验)
- [ ] Exp-1.1: HF-LoRA (qkv only)
- [ ] Exp-1.2: HF-LoRA (fc1 only)
- [ ] Exp-1.3: HF-LoRA (fc2 only)
- [ ] Exp-1.4: HF-LoRA (qkv+fc1)
- [ ] Exp-1.5: HF-LoRA (Full)

### Group 2: 提示词库消融 (5个实验)
- [ ] Exp-2.1: Prompt Bank (Generic)
- [ ] Exp-2.2: Prompt Bank (Gen+App)
- [ ] Exp-2.3: Prompt Bank (Gen+Phy)
- [ ] Exp-2.4: Prompt Bank (3 cat)
- [ ] Exp-2.5: Prompt Bank (5 cat)

### Group 3: 组合消融 (2个实验)
- [ ] Exp-3.1: HF-LoRA + Prompt (3 cat)
- [ ] Exp-3.2: HF-LoRA + Prompt (5 cat)

### Group 4: 超参数消融 (4个实验)
- [ ] Exp-4.1: HF-LoRA r=8
- [ ] Exp-4.2: HF-LoRA r=32
- [ ] Exp-4.3: HF-LoRA alpha=16
- [ ] Exp-4.4: HF-LoRA alpha=64

---

## 💡 使用建议

1. **先运行Group 0**: 建立基线性能
2. **按需运行**: 根据论文需要选择运行哪些组
3. **并行评估**: 如果有多个GPU，可以同时运行多个组
4. **结果验证**: 评估完成后检查 `metrics.json` 确保结果正确

---

## 🐛 故障排除

### 问题：LoRA checkpoint 不存在
```
⚠️  Warning: LoRA checkpoint not found: ./ablation_output/group0/exp0_2_standard_lora/lora_checkpoint_best
   Skipping this experiment...
```

**解决**: 确保训练完成，检查 `lora_checkpoint_best` 目录是否存在

### 问题：评估超时
```
⏱️  Timeout: Exp-0.1 Full Fine-tuning took too long
```

**解决**: 在 `run_all_evaluations.py` 中增加 `timeout` 参数

### 问题：GPU内存不足
```
RuntimeError: CUDA out of memory
```

**解决**: 减小 `batch_size` 或使用更小的模型

---

## 📞 联系方式

如有问题，请检查：
1. 训练是否完成
2. 路径是否正确
3. 数据集是否存在
4. GPU是否可用

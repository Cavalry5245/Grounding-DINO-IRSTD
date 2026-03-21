# 红外小目标检测消融实验计划

## 实验概述

本文提出两个主要创新点：
1. **HF-LoRA (High-Frequency Enhanced LoRA)**: 针对红外小目标高频特征增强的LoRA微调方法
2. **多粒度提示词库**: 从5个维度描述红外小目标的提示词增强方法

## 实验分组

### Group 0: Baseline (基线实验)

| 实验编号 | 实验名称 | HF-LoRA | 提示词库 | 说明 | 状态 |
|----------|----------|---------|----------|------|------|
| Exp-0.1 | Full Fine-tuning | ❌ | ❌ | 全量微调，不使用LoRA | ⏳ 待运行 |
| Exp-0.2 | Standard LoRA | ❌ | ❌ | 标准LoRA微调 | ⏳ 待运行 |

**目的**: 建立性能基线，对比全量微调和LoRA微调的效果差异

---

### Group 1: HF-LoRA 消融实验

| 实验编号 | 实验名称 | HF-LoRA | 提示词库 | 说明 | 状态 |
|----------|----------|---------|----------|------|------|
| Exp-1.1 | HF-LoRA (qkv only) | ✅ (qkv) | ❌ | 仅在qkv层使用HF-LoRA | ⏳ 待运行 |
| Exp-1.2 | HF-LoRA (fc1 only) | ✅ (fc1) | ❌ | 仅在fc1层使用HF-LoRA | ⏳ 待运行 |
| Exp-1.3 | HF-LoRA (fc2 only) | ✅ (fc2) | ❌ | 仅在fc2层使用HF-LoRA | ⏳ 待运行 |
| Exp-1.4 | HF-LoRA (qkv+fc1) | ✅ (qkv+fc1) | ❌ | 在qkv和fc1层使用HF-LoRA | ⏳ 待运行 |
| Exp-1.5 | HF-LoRA (Full) | ✅ (qkv+fc1+fc2) | ❌ | 完整HF-LoRA配置 | ⏳ 待运行 |

**目的**: 验证HF-LoRA在不同层的作用，找出最优的层组合

**预期结果**: 
- Exp-1.5 (Full) 应该优于 Exp-0.2 (Standard LoRA)
- qkv层可能对高频特征最敏感

---

### Group 2: 提示词库消融实验

| 实验编号 | 实验名称 | HF-LoRA | 提示词库 | 说明 | 状态 |
|----------|----------|---------|----------|------|------|
| Exp-2.1 | Prompt Bank (Generic) | ❌ | ✅ (generic) | 仅使用通用描述 | ⏳ 待运行 |
| Exp-2.2 | Prompt Bank (Gen+App) | ❌ | ✅ (generic+appearance) | 通用+外观描述 | ⏳ 待运行 |
| Exp-2.3 | Prompt Bank (Gen+Phy) | ❌ | ✅ (generic+physical) | 通用+物理描述 | ⏳ 待运行 |
| Exp-2.4 | Prompt Bank (3 cat) | ❌ | ✅ (3类) | 通用+外观+物理（默认） | ⏳ 待运行 |
| Exp-2.5 | Prompt Bank (5 cat) | ❌ | ✅ (5类) | 全部5个类别 | ⏳ 待运行 |

**目的**: 验证多粒度提示词库的效果，找出最优的提示词组合

**预期结果**:
- Exp-2.4 (3类) 应该优于 Exp-2.1-2.3
- Exp-2.5 (5类) 可能进一步提升，但也可能引入噪声

---

### Group 3: 组合消融实验

| 实验编号 | 实验名称 | HF-LoRA | 提示词库 | 说明 | 状态 |
|----------|----------|---------|----------|------|------|
| Exp-3.1 | HF-LoRA + Prompt (3 cat) | ✅ | ✅ (3类) | 完整方法（默认配置） | ⏳ 待运行 |
| Exp-3.2 | HF-LoRA + Prompt (5 cat) | ✅ | ✅ (5类) | 最大配置 | ⏳ 待运行 |

**目的**: 验证两个创新点的组合效果

**预期结果**:
- Exp-3.1 应该是最佳配置
- Exp-3.2 可能进一步提升，但计算开销更大

---

### Group 4: 超参数消融实验

| 实验编号 | 实验名称 | LoRA-r | LoRA-alpha | 说明 | 状态 |
|----------|----------|--------|------------|------|------|
| Exp-4.1 | HF-LoRA r=8 | 8 | 16 | 低rank配置 | ⏳ 待运行 |
| Exp-4.2 | HF-LoRA r=32 | 32 | 64 | 高rank配置 | ⏳ 待运行 |
| Exp-4.3 | HF-LoRA alpha=16 | 16 | 16 | 低alpha配置 | ⏳ 待运行 |
| Exp-4.4 | HF-LoRA alpha=64 | 16 | 64 | 高alpha配置 | ⏳ 待运行 |

**目的**: 验证LoRA超参数对性能的影响

**预期结果**:
- r=16, alpha=32 (默认) 应该是平衡性能和效率的最佳选择
- 更高的r和alpha可能提升性能，但增加计算开销

---

## 运行命令

### 运行单个实验组

```bash
# Group 0: Baseline
bash run_ablation_study.sh 0

# Group 1: HF-LoRA 消融
bash run_ablation_study.sh 1

# Group 2: 提示词库消融
bash run_ablation_study.sh 2

# Group 3: 组合消融
bash run_ablation_study.sh 3

# Group 4: 超参数消融
bash run_ablation_study.sh 4
```

### 运行所有实验

```bash
bash run_ablation_study.sh all
```

---

## 结果收集

### 查看结果汇总

```bash
python ablation_results_collector.py --output_dir ./ablation_output
```

### 保存结果到CSV

```bash
python ablation_results_collector.py --output_dir ./ablation_output --save_csv
```

### 保存结果到Markdown

```bash
python ablation_results_collector.py --output_dir ./ablation_output --save_md
```

---

## 评估指标

| 指标 | 说明 | 越高越好 |
|------|------|----------|
| mAP | 平均精度均值 | ✅ |
| mAP@0.5 | IoU=0.5时的mAP | ✅ |
| mAP@0.5:0.95 | IoU从0.5到0.95的mAP平均值 | ✅ |
| P (Precision) | 精确率 | ✅ |
| R (Recall) | 召回率 | ✅ |
| F1 | F1分数 | ✅ |
| PD (Probability of Detection) | 检测概率 | ✅ |
| FA (False Alarm) | 虚警率 | ❌ |

---

## 论文中的实验设计建议

### 1. 主实验表格

| Method | HF-LoRA | Prompt Bank | mAP | mAP@0.5 | F1 | PD | FA |
|--------|---------|-------------|-----|---------|-----|-----|-----|
| Full Fine-tuning | ❌ | ❌ | - | - | - | - | - |
| Standard LoRA | ❌ | ❌ | - | - | - | - | - |
| HF-LoRA (Full) | ✅ | ❌ | - | - | - | - | - |
| Prompt Bank (3 cat) | ❌ | ✅ | - | - | - | - | - |
| **Ours (HF-LoRA + Prompt Bank)** | ✅ | ✅ | - | - | - | - | - |

### 2. 消融实验表格

#### HF-LoRA 消融

| HF-LoRA Modules | mAP | mAP@0.5 | F1 |
|-----------------|-----|---------|-----|
| qkv only | - | - | - |
| fc1 only | - | - | - |
| fc2 only | - | - | - |
| qkv+fc1 | - | - | - |
| qkv+fc1+fc2 (Full) | - | - | - |

#### 提示词库消融

| Prompt Categories | mAP | mAP@0.5 | F1 |
|------------------|-----|---------|-----|
| Generic | - | - | - |
| Generic+Appearance | - | - | - |
| Generic+Physical | - | - | - |
| Generic+Appearance+Physical | - | - | - |
| All 5 categories | - | - | - |

#### 超参数消融

| LoRA-r | LoRA-alpha | mAP | mAP@0.5 | F1 |
|--------|------------|-----|---------|-----|
| 8 | 16 | - | - | - |
| 16 | 16 | - | - | - |
| 16 | 32 (default) | - | - | - |
| 16 | 64 | - | - | - |
| 32 | 64 | - | - | - |

---

## 实验进度跟踪

- [ ] Exp-0.1: Full Fine-tuning
- [ ] Exp-0.2: Standard LoRA
- [ ] Exp-1.1: HF-LoRA (qkv only)
- [ ] Exp-1.2: HF-LoRA (fc1 only)
- [ ] Exp-1.3: HF-LoRA (fc2 only)
- [ ] Exp-1.4: HF-LoRA (qkv+fc1)
- [ ] Exp-1.5: HF-LoRA (Full)
- [ ] Exp-2.1: Prompt Bank (Generic)
- [ ] Exp-2.2: Prompt Bank (Gen+App)
- [ ] Exp-2.3: Prompt Bank (Gen+Phy)
- [ ] Exp-2.4: Prompt Bank (3 cat)
- [ ] Exp-2.5: Prompt Bank (5 cat)
- [ ] Exp-3.1: HF-LoRA + Prompt (3 cat)
- [ ] Exp-3.2: HF-LoRA + Prompt (5 cat)
- [ ] Exp-4.1: HF-LoRA r=8
- [ ] Exp-4.2: HF-LoRA r=32
- [ ] Exp-4.3: HF-LoRA alpha=16
- [ ] Exp-4.4: HF-LoRA alpha=64

---

## 注意事项

1. **随机种子**: 所有实验使用相同的随机种子 (seed=42) 确保可重复性
2. **训练轮数**: 建议每个实验训练15-20个epoch
3. **GPU资源**: 每个实验需要2个GPU，建议分批运行
4. **数据集**: 所有实验使用相同的训练集和验证集
5. **评估**: 使用相同的评估脚本和阈值设置

---

## 预期贡献

通过以上消融实验，我们将验证：

1. **HF-LoRA的有效性**: 相比标准LoRA，HF-LoRA能更好地捕捉红外小目标的高频特征
2. **多粒度提示词库的有效性**: 从多个维度描述红外小目标能提升模型的语义理解能力
3. **组合效果**: HF-LoRA和提示词库的组合能产生协同效应
4. **最优配置**: 确定最佳的层组合和超参数设置

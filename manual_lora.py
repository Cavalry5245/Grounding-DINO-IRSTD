# manual_lora.py
# 手动实现 LoRA + HF-LoRA，替代 peft 库

import torch
import torch.nn as nn
import math
import json
from pathlib import Path
from collections import OrderedDict


# ============================================================
# 1. 标准 LoRA 层
# ============================================================

class ManualLoRALinear(nn.Module):
    """标准 LoRA 层"""

    def __init__(self, original_layer: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_output + lora_output

    def merge_weights(self):
        """将 LoRA 权重合并回原始层"""
        with torch.no_grad():
            merged_weight = (
                self.original_layer.weight.data
                + self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            )
            self.original_layer.weight.data = merged_weight
        return self.original_layer


# ============================================================
# 2. HF-LoRA 层（创新点：高频增强）
# ============================================================

class HighFreqLoRALinear(nn.Module):
    """
    HF-LoRA：带高频增强的 LoRA 层
    在标准 LoRA 的低秩中间表示上添加深度可分离卷积，捕获高频局部细节。
    """

    def __init__(self, original_layer: nn.Linear, r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 高频增强分支：深度可分离卷积
        self.hf_conv = nn.Conv1d(
            in_channels=r,
            out_channels=r,
            kernel_size=3,
            padding=1,
            groups=r,
            bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_normal_(self.hf_conv.weight)

        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)

        # 下投影
        x_down = self.lora_A(self.lora_dropout(x))

        # 高频分支
        if x_down.dim() == 2:
            x_conv = self.hf_conv(x_down.unsqueeze(0).transpose(1, 2))
            x_conv = x_conv.transpose(1, 2).squeeze(0)
        elif x_down.dim() == 3:
            x_conv = self.hf_conv(x_down.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
        else:
            original_shape = x_down.shape
            x_flat = x_down.reshape(-1, x_down.shape[-2], x_down.shape[-1])
            x_conv = self.hf_conv(x_flat.transpose(1, 2)).transpose(1, 2)
            x_conv = x_conv.reshape(original_shape)

        # 融合
        x_fused = x_down + x_conv

        # 上投影
        lora_output = self.lora_B(x_fused) * self.scaling
        return original_output + lora_output

    def merge_weights(self):
        """合并（警告：卷积分支无法精确合并）"""
        import warnings
        warnings.warn(
            "HF-LoRA contains a conv branch that cannot be exactly merged "
            "into a linear layer. Only the linear part will be merged.",
            UserWarning
        )
        with torch.no_grad():
            merged_weight = (
                self.original_layer.weight.data
                + self.scaling * (self.lora_B.weight @ self.lora_A.weight)
            )
            self.original_layer.weight.data = merged_weight
        return self.original_layer


# ============================================================
# 3. LoRA 注入函数
# ============================================================

def inject_lora(model, target_modules, r=8, lora_alpha=16, lora_dropout=0.05,
                bias='none', hf_lora_modules=None, logger=None):
    """
    遍历模型，将匹配的 nn.Linear 替换为 LoRA 层。

    Args:
        model: 原始模型
        target_modules: list of str，要替换的层末端名
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: dropout rate
        bias: 'none' | 'all' | 'lora_only'
        hf_lora_modules: list of str，使用 HF-LoRA 的层名。None 或空则全部用标准 LoRA。
        logger: 日志器
    """
    target_modules = list(set(target_modules))
    if hf_lora_modules is None:
        hf_lora_modules = []

    # 验证
    available_linear = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            available_linear.add(name.split('.')[-1])

    valid_targets = [t for t in target_modules if t in available_linear]
    invalid_targets = [t for t in target_modules if t not in available_linear]

    if not valid_targets:
        if logger:
            logger.error(f"No valid target modules found!")
            logger.info(f"Available linear modules: {sorted(available_linear)}")
        raise ValueError(f"No valid target modules. Available: {sorted(available_linear)}")

    if invalid_targets and logger:
        logger.warning(f"Skipped (not found): {invalid_targets}")

    # 替换
    replaced = []

    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if child_name in valid_targets and isinstance(child_module, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name

                if child_name in hf_lora_modules:
                    lora_layer = HighFreqLoRALinear(
                        child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                    )
                    lora_type = "HF-LoRA"
                else:
                    lora_layer = ManualLoRALinear(
                        child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                    )
                    lora_type = "LoRA"

                setattr(parent_module, child_name, lora_layer)
                replaced.append((full_name, lora_type))

    # bias 处理
    if bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for name, param in model.named_parameters():
            if 'original_layer.bias' in name and any(t in name for t in valid_targets):
                param.requires_grad = True

    # 日志
    standard_count = sum(1 for _, t in replaced if t == "LoRA")
    hf_count = sum(1 for _, t in replaced if t == "HF-LoRA")

    if logger:
        logger.info("=" * 60)
        logger.info(f"LoRA Injection: {len(replaced)} layers replaced")
        logger.info(f"  Standard LoRA: {standard_count} | HF-LoRA: {hf_count}")
        logger.info(f"  r={r}, alpha={lora_alpha}, scaling={lora_alpha/r}, dropout={lora_dropout}")
        logger.info("=" * 60)
        for name, lora_type in replaced[:40]:
            logger.info(f"  [{lora_type:>7s}] {name}")
        if len(replaced) > 40:
            logger.info(f"  ... and {len(replaced) - 40} more")
        logger.info("=" * 60)

    return model


# ============================================================
# 4. 冻结与解冻
# ============================================================

def freeze_base_model(model):
    """冻结所有参数，再解冻 LoRA 相关参数"""
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(key in name for key in ['lora_A', 'lora_B', 'hf_conv']):
            param.requires_grad = True


def unfreeze_layers(model, layers_to_unfreeze, logger=None):
    """解冻指定关键字匹配的层"""
    unfrozen_params = []
    unfrozen_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        should_unfreeze = any(layer_name in name for layer_name in layers_to_unfreeze)
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_params.append(name)
            unfrozen_count += param.numel()

    if logger:
        logger.info(f"Unfroze {len(unfrozen_params)} param tensors ({unfrozen_count:,} params):")
        for name in unfrozen_params[:30]:
            logger.info(f"  ✓ {name}")
        if len(unfrozen_params) > 30:
            logger.info(f"  ... and {len(unfrozen_params) - 30} more")

    return unfrozen_params


def print_trainable_summary(model, logger=None):
    """打印可训练参数统计"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_linear_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and ('lora_A' in n or 'lora_B' in n)
    )
    hf_conv_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and 'hf_conv' in n
    )
    head_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and any(h in n for h in ['class_embed', 'bbox_embed', 'label_enc'])
    )
    other_params = trainable_params - lora_linear_params - hf_conv_params - head_params

    info_lines = [
        "=" * 50,
        "Trainable Parameter Summary:",
        f"  Total params:      {total_params:>12,}",
        f"  Trainable params:  {trainable_params:>12,} ({100 * trainable_params / total_params:.4f}%)",
        f"    - LoRA (A+B):    {lora_linear_params:>12,}",
        f"    - HF-Conv:       {hf_conv_params:>12,}",
        f"    - Det. heads:    {head_params:>12,}",
        f"    - Other:         {other_params:>12,}",
        "=" * 50,
    ]

    if logger:
        for line in info_lines:
            logger.info(line)
    else:
        for line in info_lines:
            print(line)


# ============================================================
# 5. 保存与加载
# ============================================================

def save_lora_state_dict(model):
    """提取所有可训练参数"""
    lora_state_dict = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.clone().cpu()
    return lora_state_dict


def save_lora_checkpoint(model, save_dir, epoch=None, optimizer=None,
                         lr_scheduler=None, args=None, map_score=None,
                         ap50_score=None, logger=None):
    """保存 LoRA 检查点"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # LoRA 权重
    lora_state = save_lora_state_dict(model)
    torch.save(lora_state, save_dir / "lora_weights.pth")

    # 训练状态
    training_state = {}
    if epoch is not None:
        training_state['epoch'] = epoch
    if optimizer is not None:
        training_state['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        training_state['lr_scheduler'] = lr_scheduler.state_dict()
    if map_score is not None:
        training_state['mAP'] = map_score
    if ap50_score is not None:
        training_state['AP50'] = ap50_score
    if args is not None:
        training_state['args'] = vars(args) if not isinstance(args, dict) else args

    torch.save(training_state, save_dir / "training_state.pth")

    # LoRA 配置
    if args is not None:
        lora_config = {
            'r': getattr(args, 'lora_r', 8),
            'lora_alpha': getattr(args, 'lora_alpha', 16),
            'lora_dropout': getattr(args, 'lora_dropout', 0.05),
            'lora_bias': getattr(args, 'lora_bias', 'none'),
            'target_modules': getattr(args, 'lora_target_modules', []),
            'hf_lora_modules': getattr(args, 'hf_lora_modules', []),
            'lora_unfreeze_layers': getattr(args, 'lora_unfreeze_layers', []),
        }
        with open(save_dir / "lora_config.json", 'w') as f:
            json.dump(lora_config, f, indent=2)

    if logger:
        logger.info(f"Saved LoRA checkpoint ({len(lora_state)} tensors) to {save_dir}")

    return save_dir


def load_lora_weights(model, load_dir, logger=None):
    """加载 LoRA 权重到已注入 LoRA 的模型"""
    load_dir = Path(load_dir)
    weights_path = load_dir / "lora_weights.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at {weights_path}")

    lora_state_dict = torch.load(weights_path, map_location='cpu')
    model_state = model.state_dict()
    loaded = 0
    missing = []

    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded += 1
        else:
            missing.append(name)

    if logger:
        logger.info(f"Loaded {loaded} LoRA param tensors from {weights_path}")
        if missing:
            logger.warning(f"{len(missing)} params not found in model:")
            for m in missing[:10]:
                logger.warning(f"  ✗ {m}")

    return model


def merge_lora_weights(model, logger=None):
    """将所有 LoRA 权重合并回原始层"""
    merged_count = 0

    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if isinstance(child_module, (ManualLoRALinear, HighFreqLoRALinear)):
                merged_linear = child_module.merge_weights()
                setattr(parent_module, child_name, merged_linear)
                merged_count += 1

    if logger:
        logger.info(f"Merged {merged_count} LoRA layers back into base model")

    return model
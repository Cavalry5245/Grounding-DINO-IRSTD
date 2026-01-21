# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from groundingdino.util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# =============================================================================
# 新增：高分辨率分支相关类
# =============================================================================

class ResBlock(nn.Module):
    """简单的残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class HighResolutionBranch(nn.Module):
    """
    高分辨率旁支网络
    - 保持较高分辨率 (1/4, 1/8)
    - 轻量级设计
    - 可以捕获小目标的精细特征
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [48, 96, 192],
        out_dims: List[int] = [192, 384],  # 与Swin的stage0, stage1对齐
    ):
        super().__init__()
        
        # Stage 0: 1/2 分辨率
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )
        
        # Stage 1: 1/4 分辨率 (与Swin Stage0对齐)
        self.stage1 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            ResBlock(hidden_dims[1], hidden_dims[1]),
            ResBlock(hidden_dims[1], hidden_dims[1]),
        )
        
        # Stage 2: 1/8 分辨率 (与Swin Stage1对齐)
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            ResBlock(hidden_dims[2], hidden_dims[2]),
            ResBlock(hidden_dims[2], hidden_dims[2]),
        )
        
        # 输出投影层，对齐到Swin的通道数
        # self.out_proj_1 = nn.Conv2d(hidden_dims[1], out_dims[0], kernel_size=1)  # 1/4分辨率
        # self.out_proj_2 = nn.Conv2d(hidden_dims[2], out_dims[1], kernel_size=1)  # 1/8分辨率
        self.out_proj_1 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], out_dims[0], kernel_size=1),
            nn.BatchNorm2d(out_dims[0]),  # 新增：归一化
        )
        self.out_proj_2 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], out_dims[1], kernel_size=1),
            nn.BatchNorm2d(out_dims[1]),  # 新增：归一化
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            dict: 多尺度特征
                - 'hr_1': (B, C, H/4, W/4)   1/4分辨率
                - 'hr_2': (B, 2C, H/8, W/8)  1/8分辨率
        """
        # 1/2 分辨率
        x0 = self.stem(x)
        
        # 1/4 分辨率
        x1 = self.stage1(x0)
        out_1 = self.out_proj_1(x1)
        
        # 1/8 分辨率
        x2 = self.stage2(x1)
        out_2 = self.out_proj_2(x2)
        
        return {
            'hr_1': out_1,  # 1/4 分辨率 - 与Swin Stage0对齐
            'hr_2': out_2,  # 1/8 分辨率 - 与Swin Stage1对齐
        }


class FeatureFusionModule(nn.Module):
    """
    特征融合模块
    """
    
    def __init__(
        self,
        swin_dims: List[int],  # Swin实际返回的各层通道数，如 [192, 384, 768]
        fusion_type: str = 'add',
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # 拼接后降维
            self.fuse_0 = nn.Sequential(
                nn.Conv2d(swin_dims[0] * 2, swin_dims[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(swin_dims[0]),
                nn.ReLU(inplace=True),
            )
            self.fuse_1 = nn.Sequential(
                nn.Conv2d(swin_dims[1] * 2, swin_dims[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(swin_dims[1]),
                nn.ReLU(inplace=True),
            )
        
        # 可学习的融合权重
        # sigmoid(-3) ≈ 0.05，即初期 swin 占 95%，HR 占 5%，原来是0.5
        self.alpha_0 = nn.Parameter(torch.tensor(-1.0))
        self.alpha_1 = nn.Parameter(torch.tensor(-1.0))
    
    def forward(self, swin_features: Dict, hr_features: Dict) -> Dict:
        """
        swin_features: {0: NestedTensor, 1: NestedTensor, 2: NestedTensor} 或类似
        hr_features: {'hr_1': Tensor, 'hr_2': Tensor}
        """
#----- 统计迭代次数，便于观察权重变化（仅调试时使用）-----------------------------------
        # if self.training and hasattr(self, '_iter_count'):
        #     self._iter_count += 1
        # else:
        #     self._iter_count = 0
        
        # if self._iter_count % 100 == 0:
        #     alpha0 = torch.sigmoid(self.alpha_0).item()
        #     alpha1 = torch.sigmoid(self.alpha_1).item()
        #     print(f"[Fusion] alpha_0={alpha0:.4f}, alpha_1={alpha1:.4f}")
# -----------------------------------------------------------------------------------------------------------------------
        fused = {}
        
        # 获取swin的keys（按顺序）
        swin_keys = sorted(swin_features.keys())
        
        # === 融合第一层 ===
        first_key = swin_keys[0]
        swin_0 = swin_features[first_key].tensors
        hr_1 = hr_features['hr_1']
        
        # 空间尺寸对齐
        if hr_1.shape[-2:] != swin_0.shape[-2:]:
            hr_1 = F.interpolate(hr_1, size=swin_0.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.fusion_type == 'concat':
            fused_0 = self.fuse_0(torch.cat([swin_0, hr_1], dim=1))
        else:  # add
            alpha = torch.sigmoid(self.alpha_0)
            fused_0 = alpha * swin_0 + (1 - alpha) * hr_1
        
        fused[first_key] = NestedTensor(fused_0, swin_features[first_key].mask)
        
        # === 融合第二层 ===
        second_key = swin_keys[1]
        swin_1 = swin_features[second_key].tensors
        hr_2 = hr_features['hr_2']
        
        if hr_2.shape[-2:] != swin_1.shape[-2:]:
            hr_2 = F.interpolate(hr_2, size=swin_1.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.fusion_type == 'concat':
            fused_1 = self.fuse_1(torch.cat([swin_1, hr_2], dim=1))
        else:  # add
            alpha = torch.sigmoid(self.alpha_1)
            fused_1 = alpha * swin_1 + (1 - alpha) * hr_2
        
        fused[second_key] = NestedTensor(fused_1, swin_features[second_key].mask)
        
        # === 深层特征保持不变 ===
        for key in swin_keys[2:]:
            fused[key] = swin_features[key]
        
        return fused


class SwinWithHRBranch(nn.Module):
    """
    带高分辨率旁支的 Swin Transformer 包装器
    """
    
    def __init__(
        self,
        swin_backbone: nn.Module,
        swin_num_features: List[int],  # Swin完整的4层通道 [96, 192, 384, 768]
        return_interm_indices: List[int],  # 实际返回哪些层 [1,2,3] 或 [0,1,2,3]
        use_hr_branch: bool = True,
        fusion_type: str = 'add',
        freeze_swin: bool = False,
    ):
        super().__init__()
        
        self.swin = swin_backbone
        self.return_interm_indices = return_interm_indices
        
        # 实际返回的通道数
        actual_dims = [swin_num_features[i] for i in return_interm_indices]
        self.num_features = actual_dims
        self.use_hr_branch = use_hr_branch
        
        if use_hr_branch:
            # HR分支输出通道对齐到swin返回的前两层
            hr_out_dims = [actual_dims[0], actual_dims[1]]
            print(f"[INFO] HR Branch out_dims: {hr_out_dims}")
            
            self.hr_branch = HighResolutionBranch(
                in_channels=3,
                hidden_dims=[48, 96, 192],
                out_dims=hr_out_dims,
            )
            
            self.fusion = FeatureFusionModule(
                swin_dims=actual_dims,
                fusion_type=fusion_type,
            )
        
        if freeze_swin:
            self._freeze_swin()
    
    def _freeze_swin(self):
        for param in self.swin.parameters():
            param.requires_grad = False
        print("[INFO] Swin Transformer weights frozen!")
        
        # 确保 HR 分支和 Fusion 是可训练的
        if hasattr(self, 'hr_branch'):
            for param in self.hr_branch.parameters():
                param.requires_grad = True
        if hasattr(self, 'fusion'):
            for param in self.fusion.parameters():
                param.requires_grad = True
            print("[INFO] HR Branch and Fusion weights are trainable!")
    
    def forward(self, tensor_list: NestedTensor):
        swin_features = self.swin(tensor_list)
        
        if not self.use_hr_branch:
            return swin_features
        
        hr_features = self.hr_branch(tensor_list.tensors)
        fused_features = self.fusion(swin_features, hr_features)
        
        return fused_features


# =============================================================================
# 原有代码保持不变
# =============================================================================

class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now
        - use_hr_branch: 是否使用高分辨率分支 (新增)
        - hr_fusion_type: 融合方式 'add' 或 'concat' (新增)
        - freeze_swin: 是否冻结Swin权重 (新增)
    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)
    
    # 新增参数
    use_hr_branch = getattr(args, "use_hr_branch", False)
    hr_fusion_type = getattr(args, "hr_fusion_type", "add")
    freeze_swin = getattr(args, "freeze_swin", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
        
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        swin_backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )
        
        # Swin完整的4层通道数
        full_num_features = swin_backbone.num_features  # [96, 192, 384, 768] for swin_T
        # 实际返回的通道数
        bb_num_channels = [full_num_features[i] for i in return_interm_indices]
        
        if use_hr_branch:
            print(f"[INFO] Using High-Resolution Branch with fusion_type='{hr_fusion_type}'")
            print(f"[INFO] return_interm_indices: {return_interm_indices}")
            print(f"[INFO] bb_num_channels: {bb_num_channels}")
            
            backbone = SwinWithHRBranch(
                swin_backbone=swin_backbone,
                swin_num_features=full_num_features,      # 完整4层
                return_interm_indices=return_interm_indices,  # 实际返回的层
                use_hr_branch=True,
                fusion_type=hr_fusion_type,
                freeze_swin=freeze_swin,
            )
            backbone.num_features = bb_num_channels
        else:
            backbone = swin_backbone
            
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    
    return model
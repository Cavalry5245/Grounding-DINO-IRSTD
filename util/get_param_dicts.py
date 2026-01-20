import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    # import pdb;pdb.set_trace()
    if param_dict_type == 'default':
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]
        return param_dicts

    if param_dict_type == 'ddetr_in_mmdet':
        # 先判断是否启用了HR分支
        use_hr_branch = getattr(args, 'use_hr_branch', False)
        
        if use_hr_branch:
            # 方案：在原有基础上，单独设置HR分支的学习率
            param_dicts = [
                # 1. HR分支和融合模块 - 新模块用较高学习率
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if ("hr_branch" in n or "fusion" in n) and p.requires_grad],
                    "lr": args.lr,  # 与主学习率相同，或者可以设置更高如 args.lr * 2
                },
                # 2. Backbone（不包括HR分支）- 低学习率保护预训练权重
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_backbone_names) 
                            and "hr_branch" not in n 
                            and "fusion" not in n
                            and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                # 3. Linear projection层
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_linear_proj_names) 
                            and "hr_branch" not in n 
                            and "fusion" not in n
                            and p.requires_grad],
                    "lr": args.lr_linear_proj_mult,
                },
                # 4. 其他参数（transformer等）
                {
                    "params": [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, args.lr_backbone_names) 
                            and not match_name_keywords(n, args.lr_linear_proj_names) 
                            and "hr_branch" not in n 
                            and "fusion" not in n
                            and p.requires_grad],
                    "lr": args.lr,
                },
            ]
        else:
            # 原有逻辑保持不变
            param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, args.lr_backbone_names) 
                            and not match_name_keywords(n, args.lr_linear_proj_names) 
                            and p.requires_grad],
                    "lr": args.lr,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                    "lr": args.lr_linear_proj_mult,
                }
            ]        
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]
        return param_dicts

        # print("param_dicts: {}".format(param_dicts))
    
    # # 自定义 HR Branch 和 Fusion 参数分组策略
    # if param_dict_type == 'custom_hr_fusion':
    #     param_dicts = [
    #         {
    #             "params": [p for n, p in model_without_ddp.named_parameters() 
    #                       if ("hr_branch" in n or "fusion" in n) and p.requires_grad], 
    #             "lr": getattr(args, 'lr_hr_fusion', 1e-4)  # 默认使用1e-4，可通过args自定义
    #         },
    #         {
    #             "params": [p for n, p in model_without_ddp.named_parameters() 
    #                       if "hr_branch" not in n and "fusion" not in n and p.requires_grad], 
    #             "lr": getattr(args, 'lr_other', 1e-5)  # 默认使用1e-5，可通过args自定义
    #         }
    #     ]
    #     return param_dicts


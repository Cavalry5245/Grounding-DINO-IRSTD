# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from peft.tuners.lora import LoraLayer

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    # LoRA parameters
    parser.add_argument('--use_lora', action='store_true',
                        help='Enable LoRA fine-tuning')
    parser.add_argument('--lora_r', default=8, type=int,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                        default=[
                        # Encoder自注意力层
                        "sampling_offsets",
                        "attention_weights",
                        "value_proj",
                        "output_proj",
                        # 融合层
                        "v_proj",
                        "l_proj",
                        "values_v_proj",
                        "values_l_proj",
                        "out_v_proj",
                        "out_l_proj",
                        # FFN
                        "linear1",
                        "linear2",
                        # Text encoder
                        "query",
                        "key",
                        "value",
                        "dense",
                    ],
                        help='Target modules to apply LoRA (default: auto-detect linear layers)')
    parser.add_argument('--lora_bias', type=str, default='none',
                        choices=['none', 'all', 'lora_only'],
                        help='Bias training strategy')
    parser.add_argument('--lora_resume', default='',
                        help='Resume LoRA weights from peft checkpoint directory')
    parser.add_argument('--merge_lora_after_train', action='store_true',
                        help='Merge LoRA weights into base model after training')
    parser.add_argument('--lora_unfreeze_layers', type=str, nargs='+',
                        default=[
                            'class_embed',
                            'bbox_embed',
                            'label_enc',
                        ],
                        help='Layers to unfreeze (keep fully trainable)')

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def find_all_linear_names(model, exclude_keywords=None):
    """
    Find all linear layer names for LoRA targeting.
    Only returns the leaf module names that are nn.Linear.
    """
    if exclude_keywords is None:
        exclude_keywords = ['class_embed', 'bbox_embed', 'label_enc']
    
    linear_names = set()
    
    for name, module in model.named_modules():
        # 必须是 nn.Linear
        if not isinstance(module, torch.nn.Linear):
            continue
            
        # 检查排除关键词
        if any(ex in name for ex in exclude_keywords):
            continue
        
        # 获取最后一个组件名称（叶子名称）
        leaf_name = name.split('.')[-1]
        
        # 过滤掉数字名称（如 '0', '1'）和奇怪的名称
        if leaf_name.isdigit():
            continue
        if leaf_name in ['default', 'base_layer']:
            continue
        
        # 只保留有意义的名称
        valid_patterns = [
            'proj', 'linear', 'fc', 'dense', 'qkv',
            'query', 'key', 'value', 
            'sampling_offsets', 'attention_weights',
            'reduction', 'feat_map', 'enc_output'
        ]
        
        if any(p in leaf_name.lower() for p in valid_patterns):
            linear_names.add(leaf_name)
    
    result = list(linear_names)
    
    # 如果什么都没找到，返回一些安全的默认值
    if not result:
        result = [
            "sampling_offsets", "attention_weights", "value_proj", "output_proj",
            "linear1", "linear2", "query", "key", "value", "dense",
        ]
    
    return result

def print_trainable_parameters(model, logger=None):
    """Print trainable parameters info."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    message = (
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )
    if logger:
        logger.info(message)
    else:
        print(message)

def apply_lora(model, args, logger=None):
     # 确定目标模块
    if args.lora_target_modules is None:
        target_modules = [
            "sampling_offsets",
            "attention_weights", 
            "value_proj",
            "output_proj",
        ]
    else:
        target_modules = args.lora_target_modules
    
    # 确定需要解冻的层
    if hasattr(args, 'lora_unfreeze_layers') and args.lora_unfreeze_layers:
        layers_to_unfreeze = args.lora_unfreeze_layers
    else:
        # 默认解冻检测头
        layers_to_unfreeze = [
            'class_embed',
            'bbox_embed',
            'label_enc',
        ]
    
    if logger:
        logger.info(f"LoRA target modules: {target_modules}")
        logger.info(f"Layers to unfreeze: {layers_to_unfreeze}")
    
    # 验证目标模块存在
    available_linear = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            available_linear.add(name.split('.')[-1])
    
    valid_targets = [t for t in target_modules if t in available_linear]
    if not valid_targets:
        if logger:
            logger.error(f"No valid target modules found!")
            logger.info(f"Available linear modules: {sorted(available_linear)}")
        raise ValueError(f"No valid target modules. Available: {available_linear}")
    
    if logger and set(valid_targets) != set(target_modules):
        missing = set(target_modules) - set(valid_targets)
        logger.warning(f"Some target modules not found and will be skipped: {missing}")
    
    # 创建 LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=valid_targets,
        init_lora_weights=True,
    )
    
    if logger:
        logger.info("=" * 50)
        logger.info("LoRA Configuration:")
        logger.info(f"  - rank (r): {args.lora_r}")
        logger.info(f"  - alpha: {args.lora_alpha}")
        logger.info(f"  - scaling: {args.lora_alpha / args.lora_r}")
        logger.info(f"  - dropout: {args.lora_dropout}")
        logger.info(f"  - bias: {args.lora_bias}")
        logger.info(f"  - target_modules: {valid_targets}")
        logger.info("=" * 50)
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # ========== 手动解冻指定层 ==========
    unfrozen_params = []
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        # 跳过已经可训练的参数（LoRA 参数）
        if param.requires_grad:
            continue
            
        # 检查是否需要解冻
        should_unfreeze = False
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                should_unfreeze = True
                break
        
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_params.append(name)
            unfrozen_count += param.numel()
    
    if logger:
        logger.info(f"Manually unfroze {len(unfrozen_params)} parameter tensors ({unfrozen_count:,} parameters):")
        for name in unfrozen_params[:30]:
            logger.info(f"  ✓ {name}")
        if len(unfrozen_params) > 30:
            logger.info(f"  ... and {len(unfrozen_params) - 30} more")
    
    # ========== 打印最终统计 ==========
    if logger:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 分类统计
        lora_params = 0
        head_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                if 'lora_' in name:
                    lora_params += numel
                elif any(h in name for h in ['class_embed', 'bbox_embed', 'label_enc']):
                    head_params += numel
                else:
                    other_params += numel
        
        logger.info("=" * 50)
        logger.info("Trainable Parameter Summary:")
        logger.info(f"  Total params:      {total_params:>12,}")
        logger.info(f"  Trainable params:  {trainable_params:>12,} ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"    - LoRA:          {lora_params:>12,}")
        logger.info(f"    - Det. heads:    {head_params:>12,}")
        logger.info(f"    - Other:         {other_params:>12,}")
        logger.info("=" * 50)
    
    return model

def save_lora_model(model, output_dir, epoch=None, optimizer=None, lr_scheduler=None, args=None):
    """Save LoRA model using peft's save method."""
    save_dir = Path(output_dir)
    
    if epoch is not None:
        # 保存带 epoch 的检查点
        checkpoint_dir = save_dir / f"lora_checkpoint_epoch{epoch:04d}"
    else:
        checkpoint_dir = save_dir / "lora_checkpoint"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 LoRA 权重 (peft 格式)
    model.save_pretrained(checkpoint_dir)
    
    # 额外保存训练状态
    training_state = {
        'epoch': epoch,
    }
    if optimizer is not None:
        training_state['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        training_state['lr_scheduler'] = lr_scheduler.state_dict()
    if args is not None:
        training_state['args'] = vars(args)
    
    torch.save(training_state, checkpoint_dir / "training_state.pth")
    
    return checkpoint_dir


def load_lora_model(base_model, lora_path, logger=None):
    """Load LoRA weights from peft checkpoint."""
    if logger:
        logger.info(f"Loading LoRA weights from {lora_path}")
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model

def main(args):
    

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")

    # Load pretrained weights BEFORE applying LoRA
    if args.pretrain_model_path and not args.resume:
        logger.info(f"Loading pretrained model from {args.pretrain_model_path}")
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st = OrderedDict({
            k: v for k, v in utils.clean_state_dict(checkpoint).items()
            if check_keep(k, _ignorekeywordlist)
        })
        
        if ignorelist:
            logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        
        _load_output = model.load_state_dict(_tmp_st, strict=False)
        # logger.info(str(_load_output))

    # Apply LoRA if enabled
    if args.use_lora:
        if args.lora_resume:
            # 从已有的 LoRA checkpoint 加载
            logger.info(f"Loading LoRA model from {args.lora_resume}")
            model = load_lora_model(model, args.lora_resume, logger)
        else:
            # 应用新的 LoRA
            model = apply_lora(model, args, logger)
        
        print_trainable_parameters(model, logger)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("trainable params:\n" + json.dumps(
        {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, 
        indent=2
    ))

    # Optimizer setup
    if args.use_lora:
        # 只优化可训练参数 (LoRA 参数)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_dicts = [{'params': trainable_params}]
        logger.info(f"Optimizing {len(trainable_params)} LoRA parameter tensors")
    else:
        param_dicts = get_param_dict(args, model_without_ddp)
        
        # freeze some layers
        if args.freeze_keywords is not None:
            for name, parameter in model.named_parameters():
                for keyword in args.freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break
    
    logger.info("params after freezing:\n" + json.dumps(
        {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, 
        indent=2
    ))

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Build dataset
    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)

    # Resume from checkpoint (非 LoRA 的 resume)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')) and not args.use_lora:
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    
    if args.resume and not args.use_lora:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(
            clean_state_dict(checkpoint['model']), strict=False
        )

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # LoRA training state resume
    if args.use_lora and args.lora_resume:
        training_state_path = Path(args.lora_resume) / "training_state.pth"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location='cpu')
            if 'optimizer' in training_state and not args.eval:
                optimizer.load_state_dict(training_state['optimizer'])
            if 'lr_scheduler' in training_state and not args.eval:
                lr_scheduler.load_state_dict(training_state['lr_scheduler'])
            if 'epoch' in training_state:
                args.start_epoch = training_state['epoch'] + 1
            logger.info(f"Resumed training state from epoch {args.start_epoch - 1}")

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # logger.info(str(_load_output))

    # Evaluation mode
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        
        # 如果需要，合并 LoRA 权重以加速推理
        if args.use_lora and args.merge_lora_after_train:
            logger.info("Merging LoRA weights for inference...")
            model_without_ddp = model_without_ddp.merge_and_unload()
        
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args
        )
        
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                output_dir / "eval.pth"
            )

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return
    
    # Training loop
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None)
        )

        if not args.onecyclelr:
            lr_scheduler.step()

        # Save checkpoint
        if args.output_dir:
            if utils.is_main_process():
                if args.use_lora:
                    # 保存 LoRA checkpoint (peft 格式)
                    save_lora_model(
                        model_without_ddp, args.output_dir,
                        epoch=None,  # latest checkpoint
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        args=args
                    )
                    
                    # 定期保存带 epoch 的 checkpoint
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                        save_lora_model(
                            model_without_ddp, args.output_dir,
                            epoch=epoch,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            args=args
                        )
                else:
                    # 原始保存逻辑
                    checkpoint_paths = [output_dir / 'checkpoint.pth']
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                    
                    for checkpoint_path in checkpoint_paths:
                        weights = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }
                        utils.save_on_master(weights, checkpoint_path)

        # Evaluation
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device,
            args.output_dir, wo_class_error=wo_class_error, args=args,
            logger=(logger if args.save_log else None)
        )

        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        
        if _isbest and utils.is_main_process():
            if args.use_lora:
                # 保存最佳 LoRA checkpoint
                best_dir = output_dir / "lora_checkpoint_best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model_without_ddp.save_pretrained(best_dir)
                torch.save({
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'map': map_regular,
                    'args': vars(args),
                }, best_dir / "training_state.pth")
                logger.info(f"Saved best LoRA checkpoint at epoch {epoch} with mAP {map_regular:.4f}")
            else:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name
                        )

    # 训练结束后合并 LoRA 权重（可选）
    if args.use_lora and args.merge_lora_after_train and utils.is_main_process():
        logger.info("Merging LoRA weights into base model...")
        merged_model = model_without_ddp.merge_and_unload()
        merged_path = output_dir / "merged_model.pth"
        torch.save({'model': merged_model.state_dict()}, merged_path)
        logger.info(f"Merged model saved to {merged_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Cleanup
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
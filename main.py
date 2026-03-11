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
from util.utils import BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict

# ===== 手动 LoRA 实现（替代 peft 库）=====
from manual_lora import (
    inject_lora,
    unfreeze_layers,
    freeze_base_model,
    print_trainable_summary,
    save_lora_checkpoint,
    load_lora_weights,
    merge_lora_weights,
)
# ==========================================

from tqdm import tqdm


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
                            # Image Backbone (Swin Transformer)
                            "qkv",
                            "proj",
                            "fc1",
                            "fc2",
                            # Transformer Encoder & Decoder (Deformable DETR)
                            "sampling_offsets",
                            "attention_weights",
                            "value_proj",
                            "output_proj",
                            # Fusion Layers (Bi-Directional Attention)
                            "v_proj",
                            "l_proj",
                            "values_v_proj",
                            "values_l_proj",
                            "out_v_proj",
                            "out_l_proj",
                            # FFN (Feed Forward Networks in Transformer)
                            "linear1",
                            "linear2",
                        ],
                        help='Target modules to apply LoRA')
    parser.add_argument('--lora_bias', type=str, default='none',
                        choices=['none', 'all', 'lora_only'],
                        help='Bias training strategy')
    parser.add_argument('--lora_resume', default='',
                        help='Resume LoRA weights from checkpoint directory')
    parser.add_argument('--merge_lora_after_train', action='store_true',
                        help='Merge LoRA weights into base model after training')
    parser.add_argument('--lora_unfreeze_layers', type=str, nargs='+',
                        default=['class_embed', 'bbox_embed', 'label_enc'],
                        help='Layers to unfreeze (keep fully trainable)')

    # HF-LoRA parameters (创新点：高频增强 LoRA)
    parser.add_argument('--hf_lora_modules', type=str, nargs='+',
                        default=[],
                        help='Modules to use HF-LoRA instead of standard LoRA. '
                             'Example: --hf_lora_modules qkv fc1 fc2 '
                             'If empty, all modules use standard LoRA.')

    # Save options
    parser.add_argument('--save_best_only', action='store_true', default=True,
                        help='Only save the best model checkpoint')

    return parser


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def apply_lora(model, args, logger=None):
    """Apply manual LoRA (and optionally HF-LoRA) to the model and unfreeze specified layers."""
    target_modules = args.lora_target_modules if args.lora_target_modules else [
        "sampling_offsets", "attention_weights", "value_proj", "output_proj",
    ]

    layers_to_unfreeze = (
        args.lora_unfreeze_layers
        if hasattr(args, 'lora_unfreeze_layers') and args.lora_unfreeze_layers
        else ['class_embed', 'bbox_embed', 'label_enc']
    )

    # 获取 HF-LoRA 目标模块（如果启用）
    hf_lora_modules = (
        args.hf_lora_modules
        if hasattr(args, 'hf_lora_modules') and args.hf_lora_modules
        else []
    )

    if logger:
        logger.info(f"LoRA target modules: {target_modules}")
        logger.info(f"HF-LoRA modules: {hf_lora_modules if hf_lora_modules else 'None (all standard LoRA)'}")
        logger.info(f"Layers to unfreeze: {layers_to_unfreeze}")

    # ===== 注入 LoRA（替代 peft 的 get_peft_model）=====
    model = inject_lora(
        model,
        target_modules=target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        hf_lora_modules=hf_lora_modules,
        logger=logger,
    )

    # 冻结所有基础参数，只保留 LoRA 参数可训练
    freeze_base_model(model)

    # 手动解冻指定层（检测头等）
    unfreeze_layers(model, layers_to_unfreeze, logger=logger)

    # 打印可训练参数统计
    print_trainable_summary(model, logger=logger)

    return model


def main(args):
    utils.setup_distributed(args)

    # Load config file and update args
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
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    if not getattr(args, 'debug', None):
        args.debug = False

    # Setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'),
                         distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))

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

    # Fix seed for reproducibility
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

        model.load_state_dict(_tmp_st, strict=False)

    # Apply LoRA if enabled
    if args.use_lora:
        # 先注入 LoRA 结构
        model = apply_lora(model, args, logger)
        model.to(device)

        # 如果需要从检查点恢复 LoRA 权重
        if args.lora_resume:
            logger.info(f"Loading LoRA weights from {args.lora_resume}")
            model = load_lora_weights(model, args.lora_resume, logger=logger)
            model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))

    # Optimizer setup
    if args.use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_dicts = [{'params': trainable_params}]
        logger.info(f"Optimizing {len(trainable_params)} LoRA parameter tensors")
    else:
        param_dicts = get_param_dict(args, model_without_ddp)
        if args.freeze_keywords is not None:
            for name, parameter in model.named_parameters():
                for keyword in args.freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Build dataset
    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = [
                build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx])
                for idx in range(num_of_dataset_train)
            ]
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
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train),
            epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

    output_dir = Path(args.output_dir)

    # Resume from checkpoint (non-LoRA)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')) and not args.use_lora:
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if args.resume and not args.use_lora:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

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

    # Evaluation mode
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'

        # 合并 LoRA 权重用于推理加速（可选）
        if args.use_lora and args.merge_lora_after_train:
            logger.info("Merging LoRA weights for inference...")
            model_without_ddp = merge_lora_weights(model_without_ddp, logger=logger)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args
        )

        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    # Training loop
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)

    epoch_progress_bar = tqdm(range(args.start_epoch, args.epochs),
                              desc="Training Progress",
                              total=args.epochs - args.start_epoch,
                              position=0)

    for epoch in epoch_progress_bar:
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

        # Evaluation
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device,
            args.output_dir, wo_class_error=wo_class_error, args=args,
            logger=(logger if args.save_log else None)
        )

        map_regular = test_stats['coco_eval_bbox'][0]
        ap50 = test_stats['coco_eval_bbox'][1]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)

        if utils.is_main_process():
            if _isbest:
                logger.info(f"★ New best mAP: {map_regular:.4f} (AP50: {ap50:.4f}) at epoch {epoch}")
            else:
                logger.info(f"Current mAP: {map_regular:.4f} (AP50: {ap50:.4f}) | "
                        f"Best mAP: {best_map_holder.best_all.best_res:.4f} at epoch {best_map_holder.best_all.best_ep}")

        # Save checkpoint
        if args.output_dir and utils.is_main_process():
            save_best_only = getattr(args, 'save_best_only', False)

            if args.use_lora:
                # ===== LoRA 检查点保存（手动实现）=====
                if save_best_only:
                    if _isbest:
                        best_dir = output_dir / "lora_checkpoint_best"
                        save_lora_checkpoint(
                            model_without_ddp, best_dir,
                            epoch=epoch,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            args=args,
                            map_score=map_regular,
                            ap50_score=ap50,
                            logger=logger,
                        )
                        logger.info(f"Saved best LoRA checkpoint at epoch {epoch} "
                                    f"with mAP={map_regular:.4f}, AP50={ap50:.4f}")
                else:
                    # 保存最新检查点（覆盖）
                    latest_dir = output_dir / "lora_checkpoint"
                    save_lora_checkpoint(
                        model_without_ddp, latest_dir,
                        epoch=epoch,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        args=args,
                        map_score=map_regular,
                        ap50_score=ap50,
                        logger=logger,
                    )

                    # 按间隔保存带 epoch 编号的检查点
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                        epoch_dir = output_dir / f"lora_checkpoint_epoch{epoch:04d}"
                        save_lora_checkpoint(
                            model_without_ddp, epoch_dir,
                            epoch=epoch,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            args=args,
                            map_score=map_regular,
                            ap50_score=ap50,
                            logger=logger,
                        )

                    # 保存最佳检查点
                    if _isbest:
                        best_dir = output_dir / "lora_checkpoint_best"
                        save_lora_checkpoint(
                            model_without_ddp, best_dir,
                            epoch=epoch,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            args=args,
                            map_score=map_regular,
                            ap50_score=ap50,
                            logger=logger,
                        )
                        logger.info(f"Saved best LoRA checkpoint at epoch {epoch} "
                                    f"with mAP={map_regular:.4f}, AP50={ap50:.4f}")
            else:
                # ===== 非 LoRA 检查点保存（保持不变）=====
                if save_best_only:
                    if _isbest:
                        checkpoint_path = output_dir / 'checkpoint_best.pth'
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'mAP': map_regular,
                            'AP50': ap50,
                            'args': args,
                        }, checkpoint_path)
                        logger.info(f"Saved best checkpoint at epoch {epoch} "
                                    f"with mAP={map_regular:.4f}, AP50={ap50:.4f}")
                else:
                    checkpoint_paths = [output_dir / 'checkpoint.pth']
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'mAP': map_regular,
                            'AP50': ap50,
                            'args': args,
                        }, checkpoint_path)

                    if _isbest:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'mAP': map_regular,
                            'AP50': ap50,
                            'args': args,
                        }, output_dir / 'checkpoint_best_regular.pth')
                        logger.info(f"Saved best checkpoint at epoch {epoch} "
                                    f"with mAP={map_regular:.4f}, AP50={ap50:.4f}")

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'best_mAP': best_map_holder.best_all.best_res,
            'best_epoch': best_map_holder.best_all.best_ep,
        }

        try:
            log_stats['now_time'] = str(datetime.datetime.now())
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        log_stats['epoch_time'] = str(datetime.timedelta(seconds=int(epoch_time)))

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
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    # Merge LoRA weights after training (optional)
    if args.use_lora and args.merge_lora_after_train and utils.is_main_process():
        logger.info("Merging LoRA weights into base model...")
        merged_model = merge_lora_weights(model_without_ddp, logger=logger)
        merged_path = output_dir / "merged_model.pth"
        torch.save({
            'model': merged_model.state_dict(),
            'best_mAP': best_map_holder.best_all.best_res,
            'best_epoch': best_map_holder.best_all.best_ep,
        }, merged_path)
        logger.info(f"Merged model saved to {merged_path}")

    # Print final summary
    if utils.is_main_process():
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info(f"Best mAP: {best_map_holder.best_all.best_res:.4f} "
                     f"at epoch {best_map_holder.best_all.best_ep}")
        logger.info("=" * 50)

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

    # Cleanup
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
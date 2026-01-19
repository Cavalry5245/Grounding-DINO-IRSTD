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

import random
from torch.utils.data import Dataset

# ==========================================
# Prompt 增强包装类
# ==========================================
class PromptAugmentationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # 针对红外小目标的同义词池
        # 注意：Grounding DINO 强烈建议句子以 . 结尾
        self.prompt_pool = [
            "infrared small target .",       # 标准
            "small thermal object .",        # 强调热成像
            "dim target in infrared image .",# 强调暗弱
            "bright spot .",                 # 强调视觉特征
            "anomaly point .",               # 强调异常点
            "small object .",                # 泛化
            "infrared drone or bird .",      # 强调潜在类别
            "thermal signature ."            # 强调热特征
        ]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. 获取原始数据
        img, target = self.dataset[idx]
        
        # 2. 50% 概率使用增强，50% 概率保持原样 (保留一定的稳定性)
        # 你可以根据需要调整这个概率，比如改为 0.8 或 1.0
        if random.random() < 0.8: 
            new_caption = random.choice(self.prompt_pool)
            target["caption"] = new_caption
            
            new_class_name = new_caption.replace(" .", "").strip()
            target["cap_list"] = [new_class_name]
            if "label_names" in target:
                 target["label_names"] = [new_class_name]
        else:
            if "cap_list" not in target:
                raw_cap = target["caption"]
                # 假设原始 caption 也是以 . 结尾的标准格式
                raw_class = raw_cap.replace(" .", "").strip()
                target["cap_list"] = [raw_class]

        return img, target

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
    parser.add_argument('--use_prompt_aug', action='store_true', 
                        help="是否开启随机文本提示增强 (Random Prompt Augmentation)")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


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


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)

    # # 查看层命名，正式训练时不需要
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name)
    # exit() # 打印完直接退出

    # ========================================================
    # LoRA 注入代码模块 
    # ========================================================
    # 1. 引入 PEFT
    from peft import LoraConfig, get_peft_model

    # 2. 定义 LoRA 配置
    # r: LoRA 的秩，决定参数量。红外小目标任务简单，r=16 或 32 足够。
    # lora_alpha: 缩放系数，通常设为 r 的 2倍。
    # target_modules: 这是一个坑点！不同版本的 Grounding DINO 命名不一样。
    # 我们使用模糊匹配策略，尽量覆盖 Attention 里的 Q 和 V。
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], # 尝试匹配常见的 Attention 层名称
        lora_dropout=0.1,
        bias="none",
        task_type="OBJECT_DETECTION" # 或者是 None，PEFT 对自定义模型比较宽容
    )

    # 3. 打印一下原来的参数量 (用来写论文对比)
    print(f"Original Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # 4. 注入 LoRA
    # 注意：Grounding DINO 官方代码的 model 可能被包裹了 (model.module)，或者是一个复杂的类
    # 如果报错，可能需要针对 model.backbone 或 model.transformer 分别注入
    try:
        model = get_peft_model(model, lora_config)
        # === 新增：解决梯度警告 ===
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 如果是 DDP 包裹的模型，可能要深入一层
            # 或者手动定义一个 hook (稍微麻烦点，通常上面那句就够了)
            pass
        print("✅ LoRA injected successfully using generic PEFT wrapper!")
    except Exception as e:
        print(f"⚠️ Standard PEFT injection failed: {e}")
        print("Trying manual injection strategies...")
        # 如果失败，这里可能需要根据你具体代码结构调整
        pass

    # 5. 打印可训练参数情况 (截图这张表放在论文里！)
    model.print_trainable_parameters()

    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # param_dicts = get_param_dict(args, model_without_ddp)
    # =========================================================================
    # LoRA 专用参数分组逻辑
    # 1. 筛选出所有需要更新的参数 (也就是 requires_grad=True 的 LoRA 参数)
    rec_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    
    # 2. 简单的分组策略：对 LoRA 参数统一应用 args.lr
    # 如果你想精细一点，区分 weight_decay，可以用下面的写法，但通常简单写法足够了
    param_dicts = [
        {
            "params": rec_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]

    # 3. 安全检查：一定要打印这个数量！
    # 如果打印出来是 0，说明 LoRA 没注入成功，或者所有参数都被冻结了
    print(f"************************************************************")
    print(f"* 🔥 Optimizer Check: Found {len(rec_params)} tensors to optimize.")
    print(f"*    (Should match the number of LoRA modules)             *")
    print(f"************************************************************")

    
    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break
    logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

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

    if args.use_prompt_aug:
        print("**********************************************************")
        print("* 🔥 提示词增强 (Prompt Augmentation) 已开启！             *")
        print(f"* 包含词库: {len(PromptAugmentationDataset(None).prompt_pool)} 个同义词")
        print("**********************************************************")
        
        # 只对训练集做增强，验证集(Val)永远不要动，保持标准！
        dataset_train = PromptAugmentationDataset(dataset_train)
    else:
        print("**********************************************************")
        print("* 🛑 提示词增强未开启，使用默认标签。                     *")
        print("**********************************************************")

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
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)


        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

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
        logger.info(str(_load_output))

 
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    
 
    
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)
    
    print(f"Training will run for {args.epochs} epochs, starting from epoch {args.start_epoch}")

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
            print("Sampler epoch set")

        print("Calling train_one_epoch...")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None))
        print(f"train_one_epoch completed for epoch {epoch}")
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            print("Stepping learning rate scheduler")
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
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
                
        # eval
        print("Starting evaluation...")
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        print("Evaluation completed")
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
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

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        print(f"Completed epoch {epoch}")

    # remove the copied files.
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

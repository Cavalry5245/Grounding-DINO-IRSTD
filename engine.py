# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator

from pathlib import Path

try:
    from util.eval_utils import ExtendedMetrics, JSONResultsSaver
    from util.plot_utils import EvalPlotter, PredictionVisualizer
    HAS_EXTENDED_EVAL = True
except ImportError:
    HAS_EXTENDED_EVAL = False
    print("Warning: Extended evaluation modules not found. Extended features disabled.")



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    print(f"Starting epoch {epoch}, data_loader has {len(data_loader)} batches")

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        print(f"Processing batch {_cnt}")
        
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        
        print(f"Samples shape: {samples.tensors.shape}")
        print(f"Targets length: {len(targets)}")

        # # 调试代码
        # # 打印第一个样本的 Target Box 看一眼
        # print("\n🐞 DEBUG: 检查数据格式")
        # try:
        #     sample_boxes = targets[0]["boxes"]
        #     print(f"   Target Box Shape: {sample_boxes.shape}")
        #     print(f"   Target Box Values (First 3): \n{sample_boxes[:3]}")
            
        #     # 自动判断是否有问题
        #     if sample_boxes.numel() > 0 and sample_boxes.max() > 1.1:
        #         print("   ❌ 严重错误: 发现坐标值大于 1.0！模型需要归一化坐标！")
        #         import sys; sys.exit(1) # 强制报错停止
        #     else:
        #         print("   ✅ 坐标看起来是归一化的 (0-1之间)")
        # except Exception as e:
        #     print(f"   ⚠️ 无法打印 Box: {e}")
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, captions=captions)
            print(f"Model forward pass successful")
            loss_dict = criterion(outputs, targets, cap_list, captions)
            print(f"Loss computation successful: {loss_dict}")

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        print(f"Loss value: {loss_value}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
                
        # # 仅在调试时处理几个批次
        # if _cnt >= 3:
        #     break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
# model: 要评估的模型 criterion: 损失函数 postprocessors: 后处理器字典 data_loader: 数据加载器 base_ds: 基础数据集 device: 计算设备（如CPU或GPU） 
# output_dir: 输出目录 wo_class_error: 是否不计算分类错误 args: 配置参数 logger: 日志记录器
    """
    
    扩展参数 (通过args传入):
        args.save_json: 是否保存JSON结果
        args.plot_curves: 是否绘制曲线
        args.visualize: 是否可视化预测结果
        args.save_metrics: 是否保存详细指标
        args.extended_eval: 一键启用所有扩展功能
    """
    # 将模型和损失函数设置为评估模式
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # ==================== 新增：初始化扩展评估器 ====================
    extended_metrics = None
    json_saver = None
    plotter = None
    visualizer = None
    
    # 检查是否启用扩展功能
    enable_extended = HAS_EXTENDED_EVAL and getattr(args, 'extended_eval', False)
    enable_save_json = HAS_EXTENDED_EVAL and (getattr(args, 'save_json', False) or enable_extended)
    enable_plot = HAS_EXTENDED_EVAL and (getattr(args, 'plot_curves', False) or enable_extended)
    enable_visualize = HAS_EXTENDED_EVAL and (getattr(args, 'visualize', False) or enable_extended)
    enable_metrics = HAS_EXTENDED_EVAL and (getattr(args, 'save_metrics', False) or enable_extended)
    
    # 获取类别名称
    names = {}
    if args.use_coco_eval:
        try:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            names = {cat['id']: cat['name'] for cat in coco.cats.values()}
        except:
            pass
    elif hasattr(args, 'label_list'):
        names = {i: name for i, name in enumerate(args.label_list)}
    
    # 根据参数初始化扩展功能
    if enable_metrics or enable_plot:
        nc = len(names) if names else 80
        extended_metrics = ExtendedMetrics(nc=nc, save_dir=output_dir)
        if logger:
            logger.info("Extended metrics collection enabled")
    
    if enable_save_json:
        json_saver = JSONResultsSaver(save_dir=output_dir)
        if logger:
            logger.info("JSON results saving enabled")
    
    if enable_plot:
        plotter = EvalPlotter(save_dir=output_dir, names=names)
        if logger:
            logger.info("Curve plotting enabled")
    
    if enable_visualize:
        visualizer = PredictionVisualizer(save_dir=output_dir, names=names)
        if logger:
            logger.info("Prediction visualization enabled")
    # ================================================================

    _cnt = 0 # 用于跟踪处理的数据批次数量
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        # ==================== 新增：更新扩展评估器 ====================
        if extended_metrics is not None:
            # 准备GT格式 (需要将归一化cxcywh转换为xyxy像素坐标)
            gt_for_metrics = []
            for t in targets:
                gt_dict = {'labels': t['labels']}
                
                # 转换GT boxes: 归一化cxcywh -> xyxy像素坐标
                boxes = t['boxes'].clone()
                orig_size = t['orig_size']  # [h, w]
                
                # cxcywh归一化 -> xyxy像素
                cx, cy, w, h = boxes.unbind(-1)
                x1 = (cx - w / 2) * orig_size[1]
                y1 = (cy - h / 2) * orig_size[0]
                x2 = (cx + w / 2) * orig_size[1]
                y2 = (cy + h / 2) * orig_size[0]
                gt_dict['boxes'] = torch.stack([x1, y1, x2, y2], dim=-1)
                
                gt_for_metrics.append(gt_dict)
            
            extended_metrics.update(results, gt_for_metrics)
        
        if json_saver is not None:
            image_ids = [t['image_id'].item() for t in targets]
            # 如果使用COCO评估，需要转换类别ID
            coco80_to_91 = None
            if args.use_coco_eval:
                # PostProcess已经处理了类别映射，这里不需要再次转换
                pass
            json_saver.update(results, image_ids, coco80_to_91)
        
        if visualizer is not None and _cnt < visualizer.max_batches:
            visualizer.visualize_batch(samples, targets, results)
        # ================================================================
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # ==================== 新增：保存扩展结果并打印 ====================
    ext_results = None
    if extended_metrics is not None:
        ext_results = extended_metrics.save_results(names=names)
        
        if ext_results:
            # 添加到stats
            stats['precision'] = ext_results['precision']
            stats['recall'] = ext_results['recall']
            stats['f1'] = ext_results['f1']
        
        # 绘制曲线
        if plotter is not None and ext_results:
            plotter.plot_all_curves(ext_results)
    
    if json_saver is not None:
        json_saver.save('predictions.json')
    
    # 打印扩展指标
    if ext_results is not None:
        print("\n" + "=" * 70)
        print("Extended Evaluation Metrics:")
        print("=" * 70)
        print(f"  {'Metric':<20} {'Value':>15}")
        print("-" * 40)
        print(f"  {'Precision':<20} {ext_results['precision']:>15.4f}")
        print(f"  {'Recall':<20} {ext_results['recall']:>15.4f}")
        print(f"  {'F1-Score':<20} {ext_results['f1']:>15.4f}")
        print(f"  {'mAP@0.5':<20} {ext_results['ap50']:>15.4f}")
        print(f"  {'mAP@0.75':<20} {ext_results['ap75']:>15.4f}")
        print(f"  {'mAP@0.5:0.95':<20} {ext_results['map']:>15.4f}")
        print("=" * 70)
        
        # 打印每个类别的指标 (如果类别数不太多)
        if len(ext_results['unique_classes']) <= 20:
            print("\nPer-Class Metrics:")
            print("-" * 90)
            print(f"  {'Class':<20} {'GT':>8} {'Pred':>8} {'P':>10} {'R':>10} {'AP50':>10} {'mAP':>10}")
            print("-" * 90)
            for i, cls in enumerate(ext_results['unique_classes']):
                cls_name = names.get(int(cls), str(int(cls)))[:20]
                n_gt = int(ext_results['n_gt_per_class'][i])
                n_pred = int(ext_results['n_pred_per_class'][i])
                p = ext_results['p_per_class'][i]
                r = ext_results['r_per_class'][i]
                ap50 = ext_results['ap_per_class'][i, 0]
                mAP = ext_results['ap_per_class'][i].mean()
                print(f"  {cls_name:<20} {n_gt:>8} {n_pred:>8} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {mAP:>10.4f}")
            print("-" * 90)
        print("")
    # ================================================================

    return stats, coco_evaluator



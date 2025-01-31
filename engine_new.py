# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
import random
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from crop_utils import create_crops_v3

#NOTE: Repvit merge modifications here
from tqdm.auto import tqdm
from torch.optim import AdamW
import torchvision.transforms as T
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.utils import save_image, draw_bounding_boxes

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import io
import requests
import os.path as osp
import shutil
import time
import warnings
import time

#from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
#from mmcv.utils import DictAction

#from mmseg.apis import multi_gpu_test, single_gpu_test
from PIL import Image

import sys
#sys.path.append('/home/ubuntu/roisul/RepViT/segmentation')
#import repvit
#from align_resize import AlignResize
import numpy as np

#Debug visualization imports
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main(dataloader, batch, model, eval_on_format_results, eval_kwargs):
    config = "/home/ubuntu/roisul/RepViT/segmentation/configs/sem_fpn/aug_16_repvit_100k.py"
    checkpoint = "/home/ubuntu/roisul/RepViT/repvit-J13.pth"
    work_dir='/home/ubuntu/roisul/RepViT/segmentation/tools/work_dirs/avn_failure/july13'
    aug_test=False
    out=None
    format_only=False
    eval=None
    show=False
    show_dir='/home/ubuntu/roisul/RepViT/segmentation/tools/work_dirs/avn_failure/july13'
    img_folder='./test'
    gpu_collect=False
    tmpdir=None
    options=None
    eval_options=None
    launcher='none'
    opacity=0.5
    local_rank=0
    distributed = False
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            model,
            dataloader,
            show,
            show_dir,
            False,
            opacity,
            pre_eval=eval is not None and not eval_on_format_results,
            format_only=format_only or eval_on_format_results,
            format_args=eval_kwargs,
            img=batch)

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            tmpdir,
            gpu_collect,
            False,
            pre_eval=eval is not None and not eval_on_format_results,
            format_only=format_only or eval_on_format_results,
            format_args=eval_kwargs)
    return results

def get_tight_bbox(msk):
    # destination size
    h_m, w_m = msk.shape[-2], msk.shape[-1]

    x = torch.linspace(1, h_m, h_m)#.to(msk.device)
    y = torch.linspace(1, w_m, w_m)#.to(msk.device)

    mesh = torch.stack(torch.meshgrid(x, y), dim=0)

    coords_on_car = torch.unique((mesh*msk.unsqueeze(0)).view(2, -1), dim=1)[:, 1:] - 1

    xcoords = coords_on_car[0, :]
    ycoords = coords_on_car[1, :]

    bbox_xmin = (xcoords).min()
    bbox_xmax = (xcoords).max()

    bbox_ymin = (ycoords).min()
    bbox_ymax = (ycoords).max()
    return bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax

def translate_bounding_box(bbox, crop_bbox):
    """
    Translates a bounding box with respect to a crop.
    
    Args:
        bbox (tuple): Bounding box in the format (x1, y1, x2, y2).
        crop_box (tuple): Crop box in the format (crop_x1, crop_y1, crop_x2, crop_y2).

    Returns:
        tuple: Translated bounding box in the format (new_x1, new_y1, new_x2, new_y2).
    """
    x1, y1, x2, y2 = bbox
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox[0][0], crop_bbox[0][1], crop_bbox[1][0], crop_bbox[1][1]
    #print(bbox)
    #print(crop_bbox)
    #print('==========')
    # Translate the bounding box by subtracting the top-left corner of the crop
    new_x1 = x1 - crop_x1
    new_y1 = y1 - crop_y1
    new_x2 = x2 - crop_x1
    new_y2 = y2 - crop_y1
    
    # Ensure the new bounding box stays within the cropped area
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(crop_x2 - crop_x1, new_x2)
    new_y2 = min(crop_y2 - crop_y1, new_y2)
    return [new_x1, new_y1, new_x2, new_y2]

def normalize_bbox(bbox, img_size):
    """
    Normalize the bounding box coordinates and convert to (x, y, w, h) format.

    Parameters:
    - bbox: A tuple (x_min, y_min, x_max, y_max).
    - img_size: A tuple (width, height) representing the size of the image.

    Returns:
    - A tensor in the format (x, y, w, h) with normalized values.
    """
    x_min, y_min, x_max, y_max = bbox
    img_width, img_height = img_size

    # Normalize coordinates to [0, 1]
    x_min_normalized = x_min / img_width
    y_min_normalized = y_min / img_height
    x_max_normalized = x_max / img_width
    y_max_normalized = y_max / img_height

    # Convert to (x, y, w, h)
    #x_normalized = x_min_normalized
    #y_normalized = y_min_normalized
    #w_normalized = x_max_normalized - x_min_normalized
    #h_normalized = y_max_normalized - y_min_normalized

    w_normalized = x_max_normalized - x_min_normalized
    h_normalized = y_max_normalized - y_min_normalized
    x_center_normalized = x_min_normalized + w_normalized / 2
    y_center_normalized = y_min_normalized + h_normalized / 2 
    
    # Create a tensor
    #bbox_tensor = torch.tensor([x_normalized, y_normalized, w_normalized, h_normalized])
    bbox_tensor = torch.tensor([x_center_normalized, y_center_normalized, w_normalized, h_normalized])

    return bbox_tensor

def is_inside(bbox1, bbox2):
    """
    Check if bbox1 is completely inside bbox2.

    Parameters:
    - bbox1: A tuple or list with the format (x_min, y_min, x_max, y_max) representing the inner bounding box.
    - bbox2: A tuple or list with the format (x_min, y_min, x_max, y_max) representing the outer bounding box.

    Returns:
    - True if bbox1 is inside bbox2, False otherwise.
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2[0][0], bbox2[0][1], bbox2[1][0], bbox2[1][1]

    return (x_min1 >= x_min2 and
            y_min1 >= y_min2 and
            x_max1 <= x_max2 and
            y_max1 <= y_max2)

def xywh_to_xyxy(box,H,W):
    # from 0..1 to 0..W, 0..H
    
    #TODO: Remove after debug
    #box = box.cpu()
    
    #box = box.to("cuda:0") * torch.Tensor([W, H, W, H]).to("cuda:0")
    box = box * torch.Tensor([W, H, W, H])
    # from xywh to xyxy
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    # draw
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return [x0, y0, x1, y1]

def adjust_bounding_box(bbox, ori_img_shape, crop_size=(512, 512)):
    """
    Adjust the bounding box tensor to fit 512x512 crops.

    Parameters:
    - bbox: A tensor with the format [x_min, y_min, x_max, y_max].
    - crop_size: Size of the crops (width, height).

    Returns:
    - Adjusted bounding box as a tensor [x_min, y_min, x_max, y_max].
    """
    crop_width, crop_height = crop_size
    _, h, w = ori_img_shape

    # Extract coordinates
    x_min, y_min, x_max, y_max = bbox

    # Adjust x_max to the nearest multiple of crop_width
    if (x_max - x_min) % crop_width != 0:
        x_max -= (x_max - x_min) % crop_width
    x_max = min(w, x_max + crop_width)

    # Adjust y_max to the nearest multiple of crop_height
    if (y_max - y_min) % crop_height != 0:
        y_max -= (y_max - y_min) % crop_height
    y_max = min(h, y_max + crop_height)
    
    return torch.tensor([x_min, y_min, x_max, y_max])

def l2_loss_corners(bbox1, bbox2):
    """
    Calculate the L2 loss between the top-left and bottom-right coordinates of two bounding boxes.
    
    Parameters:
    bbox1 (list of floats): Bounding box in [x1, y1, x2, y2] format.
    bbox2 (list of floats): Bounding box in [x1, y1, x2, y2] format.
    
    Returns:
    tuple: A tuple containing the L2 loss for the top-left and bottom-right coordinates.
           (top_left_loss, bottom_right_loss)
    """
    
    # Check if input is valid
    if isinstance(bbox1, int) or isinstance(bbox2, int) or len(bbox1) != 4 or len(bbox2) != 4:
        return -17, -17
    
    # Extract top-left and bottom-right coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate L2 loss for top-left coordinates (x1, y1)
    top_left_loss = np.sqrt((x1_1 - x1_2) ** 2 + (y1_1 - y1_2) ** 2)
    
    # Calculate L2 loss for bottom-right coordinates (x2, y2)
    bottom_right_loss = np.sqrt((x2_1 - x2_2) ** 2 + (y2_1 - y2_2) ** 2)
    
    return top_left_loss, bottom_right_loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    #repvit_model = repvit_model.to(device)
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    
    cntr = 0
    for samples_tnsr, ori_samples_tnsr, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        caption = 'scraped . others . scratch . missing . broken . dent .'
        cap_list = ['scraped', 'others', 'scratch', 'missing', 'broken', 'dent']
        

        for i, sample in enumerate(samples_tnsr):
            if i == 0:
                batch_crops = sample
            elif i > 0:
                batch_crops = torch.cat([batch_crops, sample])
        
        for i, sample in enumerate(ori_samples_tnsr):
            if i == 0:
                batch_ori_crops = sample
            elif i > 0:
                batch_ori_crops = torch.cat([batch_ori_crops, sample])
        
        batch_targets = []
        for batch in targets:
            for target in batch:
                batch_targets.append(target)
        
        final_crops = []
        final_ori_crops = []
        final_targets = []
        for i, t in enumerate(batch_targets):
            if len(t) != 0:
                final_crops.append(batch_crops[i])
                final_ori_crops.append(batch_ori_crops[i])
                final_targets.append(batch_targets[i])
        
        if len(final_crops) == 0:
            continue
        final_targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in final_targets]
        

        with torch.cuda.amp.autocast(enabled=args.amp):
            final_crops = torch.cat(final_crops, dim=0)
            final_ori_crops = torch.stack(final_ori_crops, dim=0)
            
            final_crops = final_crops.to(device)
            final_ori_crops = final_ori_crops.to(device)
            
            crop_captions = [caption] * len(final_crops)
            crop_cap_list = [cap_list] * len(final_crops)
            outputs = model(final_crops, captions=crop_captions)
            loss_dict = criterion(outputs, final_targets, crop_cap_list, crop_captions)
            weight_dict = criterion.weight_dict
            
            #DEBUG:Visualization code starts
            # Extracting the image tensor
            #image_tensors = torch.load('pt_files2/samples_img.pt')
            #target = torch.load('pt_files2/targets.pt')
            for i in range(len(final_ori_crops)):
                ori_crop = final_ori_crops[i]
                H,W = final_targets[i]['size'].cpu().tolist()
                scaled_lst = []
                for j in range(len(final_targets[i]['boxes'])):
                    bbox = final_targets[i]['boxes'][j].cpu()
                    bbox = xywh_to_xyxy(bbox, H, W)
                    #box = box * torch.Tensor([W, H, W, H])#.to("cuda:0")
                    #bbox = [box[0].item(), box[1].item(), (box[0]+box[2]).item(), (box[1]+box[3]).item()]
                    scaled_lst.append(bbox)
                
                scaled_dmg_tnsr = torch.tensor(scaled_lst)
                crop_img = draw_bounding_boxes(ori_crop, scaled_dmg_tnsr, width=3)
                crop_img = crop_img /255.0
                save_image(crop_img, f'pt_files2/crop_img_{i}.png')
            #    
            ###Print ori samples
            #scaled_dmg_bbox_tnsr = torch.Tensor(scaled_dmg_bboxes)
            ##ori_img = draw_bounding_boxes(ori_samples, scaled_dmg_bbox_tnsr, width=3)
            #ori_img = ori_samples / 255.0
            #save_image(ori_img, f'pt_files2/ori_img.png')
            ##DEBUG:Visualization code ends
            
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        import ipdb;ipdb.set_trace()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

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
def evaluate(model, 
            criterion, 
            #repvit_model, 
            #rep_eval_on_format_results, 
            #rep_eval_kwargs, 
            postprocessors, 
            data_loader, 
            base_ds, 
            device, 
            output_dir, 
            wo_class_error=False, 
            args=None, 
            logger=None):
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

    _cnt = 0
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
    
    for samples_tnsr, ori_samples_tnsr, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        
        caption = 'scraped . others . scratch . missing . broken . dent .'
        cap_list = ['scraped', 'others', 'scratch', 'missing', 'broken', 'dent']
        
        final_crops = []
        final_ori_crops = []
        final_targets = []
        for i, t in enumerate(targets):
            if len(t) != 0:
                final_crops.append(samples_tnsr[i])
                final_ori_crops.append(ori_samples_tnsr[i])
                final_targets.append(targets[i])
        if len(final_crops) == 0:
            continue
        final_targets = [{k: v.to(device) for k, v in t[0].items() if torch.is_tensor(v)} for t in final_targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            final_crops = torch.cat(final_crops, dim=0)
            final_ori_crops = torch.stack(final_ori_crops, dim=0)
            
            final_crops = final_crops.to(device)
            final_ori_crops = final_ori_crops.to(device)
            
            crop_captions = [caption] * len(final_crops)
            crop_cap_list = [cat_list] * len(final_crops)
            
            outputs = model(final_crops, captions=crop_captions)
            loss_dict = criterion(outputs, final_targets, crop_cap_list, crop_captions)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            #Eval crop visualization
#            box_threshold = 0.3
#            for i in range(len(outputs["pred_logits"])):
#                logits = outputs["pred_logits"].sigmoid()[i]  # (nq, 256)
#                boxes = outputs["pred_boxes"][i]  # (nq, 4)
#
#                # filter output
#                logits_filt = logits.cpu().clone()
#                boxes_filt = boxes.cpu().clone()
#                filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#                logits_filt = logits_filt[filt_mask]  # num_filt, 256
#                boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
#                if len(boxes_filt) != 0:
#                    #print(boxes_filt)
#                    # get phrase
#                    tokenlizer = model.tokenizer
#                    tokenized = tokenlizer(caption)
#                    # build pred
#                    with_logits = True
#                    text_threshold = 0.25
#                    pred_phrases = []
#                    for logit, box in zip(logits_filt, boxes_filt):
#                        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#                        if with_logits:
#                            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#                        else:
#                            pred_phrases.append(pred_phrase)
#                    print(pred_phrases)
#                    #print(final_targets)
#                gt_bb_lst = []
#                for tgt in final_targets[i]["boxes"]:
#                    gt = xywh_to_xyxy(tgt, 512,512)
#                    gt = torch.Tensor(gt)
#                    gt_bb_lst.append(gt)
#                scale_gt = torch.stack(gt_bb_lst)
#                if len(boxes_filt) != 0:
#                    pred_bb_lst = []
#                    for pred_bbox in boxes_filt:
#                        pred = xywh_to_xyxy(pred_bbox, 512, 512)
#                        pred = torch.Tensor(pred)
#                        pred_bb_lst.append(pred)
#                    scale_pred = torch.stack(pred_bb_lst)
#                    scale_pred = torch.tensor(scale_pred).unsqueeze(dim=0)
#                    scale_gt = torch.tensor(scale_gt).unsqueeze(dim=0)
#                    print(scale_pred)
#                    print(scale_gt)
#            
#                    crop_vis = final_ori_crops[i]
#                    rand = torch.randint(1,1000,(1,1))[0][0].item()
#                    #dmg_tnsr = torch.cat([scale_pred, scale_gt])
#                    img_vis = draw_bounding_boxes(crop_vis, scale_gt[0], colors="red", width=3)
#                    img_vis = draw_bounding_boxes(img_vis, scale_pred[0], colors="green", width=3)
#                    img_vis = img_vis/255.
#                    save_image(img_vis, f'eval_vis/img_{rand}.png')
#            import ipdb;ipdb.set_trace()
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        print('Val loss', loss_value)
        orig_target_sizes = torch.stack([torch.Tensor([1080,1920]) for t in final_targets], dim=0).to(device)

        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
#        if 'segm' in postprocessors.keys():
#            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#            
#        res = {target['image_id'].item(): output for target, output in zip(final_targets, results)}
#
#        if coco_evaluator is not None:
#            coco_evaluator.update(res)
#
#        if panoptic_evaluator is not None:
#            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#            for i, target in enumerate(targets):
#                image_id = target["image_id"].item()
#                file_name = f"{image_id:012d}.png"
#                res_pano[i]["image_id"] = image_id
#                res_pano[i]["file_name"] = file_name
#
#            panoptic_evaluator.update(res_pano)
#        
#        if args.save_results:
#
#            for i, (tgt, res) in enumerate(zip(targets, results)):
#                """
#                pred vars:
#                    K: number of bbox pred
#                    score: Tensor(K),
#                    label: list(len: K),
#                    bbox: Tensor(K, 4)
#                    idx: list(len: K)
#                tgt: dict.
#
#                """
#                # compare gt and res (after postprocess)
#                gt_bbox = tgt['boxes']
#                gt_label = tgt['labels']
#                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
#
#                _res_bbox = res['boxes']
#                _res_prob = res['scores']
#                _res_label = res['labels']
#                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
#       
#
#                if 'gt_info' not in output_state_dict:
#                    output_state_dict['gt_info'] = []
#                output_state_dict['gt_info'].append(gt_info.cpu())
#
#                if 'res_info' not in output_state_dict:
#                    output_state_dict['res_info'] = []
#                output_state_dict['res_info'].append(res_info.cpu())
#
#            # # for debug only
#            # import random
#            # if random.random() > 0.7:
#            #     print("Now let's break")
#            #     break
#        _cnt += 1
#        if args.debug:
#            if _cnt % 15 == 0:
#                print("BREAK!"*5)
#                break
#
#    if args.save_results:
#        import os.path as osp
#        
#        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
#        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
#        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
#        print("Saving res to {}".format(savepath))
#        torch.save(output_state_dict, savepath)
#        
#
#
#    # gather the stats from all processes
#    metric_logger.synchronize_between_processes()
#    print("Averaged stats:", metric_logger)
#    if coco_evaluator is not None:
#        coco_evaluator.synchronize_between_processes()
#    if panoptic_evaluator is not None:
#        panoptic_evaluator.synchronize_between_processes()
#
#    # accumulate predictions from all images
#    if coco_evaluator is not None:
#        coco_evaluator.accumulate()
#        coco_evaluator.summarize()
#        
#    panoptic_res = None
#    if panoptic_evaluator is not None:
#        panoptic_res = panoptic_evaluator.summarize()
#    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
#    if coco_evaluator is not None:
#        if 'bbox' in postprocessors.keys():
#            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#        if 'segm' in postprocessors.keys():
#            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#    if panoptic_res is not None:
#        stats['PQ_all'] = panoptic_res["All"]
#        stats['PQ_th'] = panoptic_res["Things"]
#        stats['PQ_st'] = panoptic_res["Stuff"]

#    return stats, coco_evaluator, loss_value
    return loss_value




# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import math
import random
import time
import pandas as pd
from pathlib import Path
import os, sys
import numpy as np
import torch
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder, to_device
import util.misc as utils

from crop_utils import create_crops_v3
import datasets
from datasets import build_dataset, get_coco_api_from_dataset

from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import warnings
warnings.filterwarnings("ignore")

import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import os.path as osp
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset as rep_build_dataset

from datasets.cocogrounding_eval import CocoGroundingEvaluator
#Repvit imports

#NOTE: Repvit merge modifications here
from tqdm.auto import tqdm
from torch.optim import AdamW
import torchvision.transforms as T
from torchvision import ops
import groundingdino.datasets.transforms as F
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import io
import requests
import os.path as osp
import shutil
import time
import warnings
import time

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from PIL import Image

import sys
sys.path.append('/home/ubuntu/roisul/RepViT/segmentation')
import repvit
from align_resize import AlignResize
import numpy as np

#Debug visualization imports
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    parser.add_argument('--num_workers', default=1, type=int)
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
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def repvit_stuff():
    config = "/home/ubuntu/roisul/RepViT/segmentation/configs/sem_fpn/aug_16_repvit_100k.py"
    checkpoint = "/home/ubuntu/roisul/RepViT/repvit-J13.pth"
    work_dir='/home/ubuntu/roisul/RepViT/segmentation/tools/work_dirs/avn_failure/july13'
    aug_test=False
    out=None
    format_only=False
    eval=None
    show=False
    #show_dir='/home/ubuntu/roisul/RepViT/segmentation/tools/work_dirs/avn_failure/july13'
    show_dir=None
    img_folder='./test'
    gpu_collect=False
    tmpdir=None
    options=None
    eval_options=None
    launcher='none'
    opacity=0.5
    local_rank=0

    cfg = mmcv.Config.fromfile(config)
    if options is not None:
        cfg.merge_from_dict(options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(work_dir, f'eval_{timestamp}.json')

    dataset = rep_build_dataset(cfg.data.test)

    #data_loader = build_dataloader(
    #    dataset,
    #    samples_per_gpu=1,
    #    workers_per_gpu=1,#cfg.data.workers_per_gpu,
    #    dist=distributed,
    #    shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = model.load_state_dict(torch.load('/home/ubuntu/roisul/RepViT/repvit-Sep2.pth')['state_dict'])
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if eval_options is None else eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        eval is not None and 'cityscapes' in eval)
    if eval_on_format_results:
        assert len(eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None
    return model, eval_on_format_results, eval_kwargs


def repvit_main(dataloader, batch, model, eval_on_format_results, eval_kwargs):
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

    x = torch.linspace(1, h_m, h_m).to(msk.device)
    y = torch.linspace(1, w_m, w_m).to(msk.device)

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

def translate_bbox_to_original_image(bbox, crop_position, crop_size=(512, 512), original_size=(1920, 1080)):
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = bbox
    crop_x, crop_y = crop_position

    x_min_original = x_min_crop + crop_x
    y_min_original = y_min_crop + crop_y
    x_max_original = x_max_crop + crop_x
    y_max_original = y_max_crop + crop_y

    return (x_min_original, y_min_original, x_max_original, y_max_original)




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
    box = box.to("cuda:0") * torch.Tensor([W, H, W, H]).to("cuda:0")
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

def create_crops_v2(image_tensor, ori_tensor, bbox, crop_size=(512, 512), stride=(256, 256)):
    """
    Create overlapping crops from the adjusted bounding box tensor.

    Parameters:
    - image_tensor: The image tensor to crop from.
    - ori_tensor: The original tensor to crop from.
    - bbox: A tensor with the format [x_min, y_min, x_max, y_max].
    - crop_size: Size of the crops (width, height).
    - stride: Step size for the sliding window (overlap control).

    Returns:
    - A list of crops, original image crops, and their respective bounding boxes.
    """
    crop_width, crop_height = crop_size
    stride_x, stride_y = stride

    x_min, y_min, x_max, y_max = bbox.tolist()  # Convert tensor to list

    crops = []
    ori_crops = []
    crop_bboxes = []

    # Loop through the adjusted bounding box with overlap using stride
    for x in range(x_min, x_max, stride_x):
        for y in range(y_min, y_max, stride_y):
            # Ensure the crop does not exceed image dimensions
            x_end = min(x + crop_width, image_tensor.shape[-1])
            y_end = min(y + crop_height, image_tensor.shape[-2])
            
            if (x_end - x == crop_width) and (y_end - y == crop_height):
                top_left = (x, y)
                bottom_right = (x_end, y_end)
                
                # Crop from the image tensor
                crop = image_tensor[:, :, y:y_end, x:x_end]
                ori_crop = ori_tensor[:, y:y_end, x:x_end]
                
                crops.append(crop)
                ori_crops.append(ori_crop)
                crop_bboxes.append((top_left, bottom_right))

    return crops, ori_crops, crop_bboxes

def get_coco_bbox(kpts, h, w, dmg):
    # Convert kpts from XY to XYWH
    kpt_x, kpt_y = kpts[0] * w, kpts[1] * h
    small = 8*2
    med = 16*2
    big = 32*2
    #change 3
    if dmg == 'small':
        x1, y1 = kpt_x-small, kpt_y-small
        x2, y2 = kpt_x+small, kpt_y+small
    elif dmg == 'medium':
        x1, y1 = kpt_x-med, kpt_y-med
        x2, y2 = kpt_x+med, kpt_y+med
    elif dmg == 'large':
        x1, y1 = kpt_x-big, kpt_y-big
        x2, y2 = kpt_x+big, kpt_y+big
    else:
        x1, y1 = kpt_x-small, kpt_y-small
        x2, y2 = kpt_x+small, kpt_y+small

    bbox = [x1,y1,x2,y2]
    bbox = [round(val,1) for val in bbox]

    return bbox

def gen_coco_cat(categories):
    content = [
        # List of dictionaries for each category
        {"id": 0, "name": "damages", "supercategory": "none"},
        *[
            {"id": i + 1, "name": category, "supercategory": "damages"}
            for i, category in enumerate(categories)
        ]
    ]

    # Creating a dictionary to map category names to category IDs
    cat_id_dct = {category: i  for i, category in enumerate(categories)}

    return content, cat_id_dct

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].
    
    Returns:
    - IoU value between box1 and box2.
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def calculate_center_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].
    
    Returns:
    - Distance between the centers of box1 and box2.
    """
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5, distance_threshold=20):
    
    data_per_predbox = dict()
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        min_dist = 1920
        gt_box_iou_id = -1
        gt_box_dist_id = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            #print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                gt_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                gt_box_dist_id = j

        lst1 = [max_iou, gt_box_iou_id, min_dist, gt_box_dist_id]
        data_per_predbox[i] = lst1
    #print(data_per_predbox)
    #print('---------------------')
    data_per_gtbox = dict()
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0
        min_dist = 1920
        pred_box_iou_id = -1
        pred_box_dist_id = -1
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            #print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                pred_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                pred_box_dist_id = j
        lst2 = [max_iou, pred_box_iou_id, min_dist, pred_box_dist_id]
        data_per_gtbox[i] = lst2
    #print(data_per_gtbox)
    return data_per_predbox, data_per_gtbox
            
        

def evaluate_detections_backup(pred_boxes, gt_boxes, iou_threshold=0.5, distance_threshold=20):
    """
    Evaluate model performance for object detection using precision, recall, and accuracy.
    
    Parameters:
    - pred_boxes: List of predicted bounding boxes in the format [x_min, y_min, x_max, y_max].
    - gt_boxes: List of ground truth bounding boxes in the format [x_min, y_min, x_max, y_max].
    - iou_threshold: IoU threshold above which a detection is considered a True Positive.
    - distance_threshold: Distance threshold within which a detection is considered a match.
    
    Returns:
    - A dictionary with accuracy, precision, and recall values.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    # Track matched ground truth boxes
    matched_gt = [False] * len(gt_boxes)
    
    # Calculate True Positives (TP) and False Positives (FP)
    for pred_box in pred_boxes:
        #match_found = False
        local_tp = 0
        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            #high iou, min dist
            # Check if the box qualifies as a True Positive
            if (iou >= iou_threshold or distance <= distance_threshold) and not matched_gt[i]:
                local_tp += 1
                matched_gt[i] = True
                match_found = True
                #break #Prevents single pred to be considered with multiple GTs. Might have to remove break statement
        
        #if not match_found:
        #    fp += 1  # No match found for this prediction, so it is a false positive
        if local_tp > 0:
            tp += local_tp
        else:
            fp += 1
    # Calculate False Negatives (FN)
    fn = matched_gt.count(False)  # Unmatched ground truth boxes

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Accuracy
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    #dct = {
    #    "tp": tp,
    #    "fp": fp,
    #    "tn": tn,
    #    "fn": fn        
    #        }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def calculate_area(box):
    """
    Calculate the area of a bounding box.
    
    Parameters:
    - box: A bounding box in the format [x_min, y_min, x_max, y_max].
    
    Returns:
    - The area of the bounding box.
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def remove_overlapping_bboxes(bboxes, labels, confs, iou_threshold=0.95):
    """
    Remove overlapping bounding boxes based on IoU threshold, keeping the one with the larger area.
    
    Parameters:
    - bboxes: List of bounding boxes in the format [x_min, y_min, x_max, y_max].
    - iou_threshold: IoU threshold above which bounding boxes are considered overlapping.
    
    Returns:
    - A list of unique bounding boxes.
    """
    unique_bboxes = []
    
    # Keep track of whether a box is already selected
    keep = [True] * len(bboxes)
    
    for i in range(len(bboxes)):
        if not keep[i]:
            continue
        
        # Compare bbox i with all other boxes
        for j in range(i + 1, len(bboxes)):
            if not keep[j]:
                continue
            
            iou = calculate_iou(bboxes[i], bboxes[j])
            
            # If IoU exceeds the threshold, keep the box with the larger area
            if iou > iou_threshold:
                area_i = calculate_area(bboxes[i])
                area_j = calculate_area(bboxes[j])
                
                if area_i >= area_j:
                    keep[j] = False  # Discard bbox j if bbox i has larger area
                else:
                    keep[i] = False  # Discard bbox i if bbox j has larger area
                    break  # Stop comparing bbox i, as it's been discarded
    
    # Only keep the boxes that were not marked as duplicates
    unique_bboxes = [bboxes[i] for i in range(len(bboxes)) if keep[i]]
    unique_labels = [labels[i] for i in range(len(labels)) if keep[i]]
    unique_confs = [confs[i] for i in range(len(confs)) if keep[i]]
    return unique_bboxes, unique_labels, unique_confs

def download_from_cdn(url: str) -> bytes:
    res = requests.get(url, stream=True)
    data = b""
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            data += chunk
    return data

def size_checks(img: Image) -> np.ndarray:
    if img.size != (1920, 1080) and img.size == (1080, 1920): #the image is vertical
        img = np.array(img.rotate(90, expand=True))
    elif img.size == (3840, 2160) or (img.size[0] == img.size[1] and img.size[0] >= 1920): #the img is large
        img = img.resize((1920, 1080))
        img = np.array(img)
    elif (img.size[0] < 1920 and img.size[0] != 1080) and (img.size[1] < 1080): #the image is small
        img = img.resize((1920, 1080))
        img = np.array(img)
    else:
        img = np.array(img)
    assert img.shape == (1080, 1920, 3), print(f"Unexpected shape: {img.shape}", flush=True) 
    return img

def get_img(bucket: str, key: str, pc: int) -> np.ndarray:
    
    if bucket != "cdn":
        byte_data = download_from_s3(bucket=bucket, key=key)
    else:
        byte_data = download_from_cdn(key)
    
    img = Image.open(io.BytesIO(byte_data))
    img = size_checks(img) 

    return img 

def str_2_lst(row):
    import json
    test = row['photo_lst']
    if type(row['photo_lst'] == str):
        test = json.loads(row['photo_lst'])
        if type(test) == str:
            test = eval(test)
    return test

def get_kp_lst(row):
    kp_dct = eval(row['kp_lst'])
    if type(kp_dct) == str:
        kp_dct = eval(kp_dct)
    kp_lst = []
    if len(kp_dct) > 0:
        print(kp_dct)
        for ele in kp_dct:
            if type(ele) == list:
                kp_lst.append(ele)
            elif type(ele) == dict:
                kp_lst.append([ele["x"], ele["y"]])
    return kp_lst

def preprocess(df):
    df['photo_lst'] = df.apply(str_2_lst, axis=1)
    df['kp_lst'] = df.apply(get_kp_lst, axis=1)
    return df

def select_sessions(df):
    #R8TR
    r8_df = pd.read_csv('/home/ubuntu/roisul/dmg_test_r8tr.csv')
    r8_df = r8_df.loc[:, r8_df.columns != "Unnamed: 0"]
    r8_df.columns = r8_df.iloc[0]
    r8_df = r8_df[1:].reset_index(drop=True)
    r8tr_sessions = r8_df['Session Key'].tolist()

    #Undamaged VN
    undam_df = pd.read_csv('/home/ubuntu/roisul/AI_damage_detection_consistency_Nodamages_Compare_from_V10.csv')
    nodmg_csv = pd.read_csv('/home/ubuntu/roisul/AI_damage_detection_consistency_Nodamages_Compare.csv')
    undmg = undam_df.merge(nodmg_csv[["VIN", "PRODUCT"]], left_on='SessID', right_on='PRODUCT') 
    undmg_sessions = undmg['SessID']

    #Failure list of cases
    fail_sessions = ["AMWT-0HBEPB1XGB",
        "AMWT-10KZKNLSYY",
        "AMWT-1UC08HX2QC",
        "AMWT-WCOLSLZSOV",
        "AMWT-B0B1REOT3R",
        "AMWT-PEKY7ADDZG",
        "AMWT-RCZPUXFY8W",
        "AMWT-KENBQCBWKA"
        ]
    return r8tr_sessions, undmg_sessions, fail_sessions

def get_seg(bucket, fname, pc):
    payload = {"bucket": bucket, "file": fname, "photocode": pc}
    response = requests.post("https://segmentation.ai-dev.paveapi.com", json=payload)
    car_bbox = response.json()["car_bbox"] 
    car_bbox = np.asarray(car_bbox, dtype=np.float64)[0]
    #ymin, ymax, xmin, xmax -> x_min, y_min, x_max, y_max
    car_bbox = np.asarray([car_bbox[2], car_bbox[0], car_bbox[3], car_bbox[1]])
    return car_bbox

def img_preprocessing(img):
    from torchvision import transforms
    img_pil = transforms.ToPILImage()(img)

    transform = F.Compose(
        [
            F.ToTensor(),
            F.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(img_pil, None)  # 3, h, w
    return image

@torch.no_grad()
def evaluate(model, 
            criterion, 
            repvit_model, 
            ckpt_name,
            exp_name,
            rep_eval_on_format_results, 
            rep_eval_kwargs, 
            postprocessors, 
            data_loader, 
            device, 
            output_dir, 
            args=None, 
            logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    #coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


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
    
    data_pth = f"/home/ubuntu/roisul/241129.parquet"
    data = pd.read_parquet(data_pth)
    data = data.sample(100000, random_state=42)
    data = data[data['SessID'].str.startswith('AM')]
    #r8tr_sess, undmg_sess, fail_sess = select_sessions(data)
    #data1 = data[data['SessID'].isin(r8tr_sess)]
    #data2 = data[data['SessID'].isin(undmg_sess)]
    #data3 = data[data['SessID'].isin(fail_sess)]
    #data = data2#pd.concat([data1,data2])
    #sess_lst = ["AMWT-OUKCXPHSRR", "AMWT-WOMPSN3X4L"]
    #data = data[data['SessID'].isin(sess_lst)]
    data = preprocess(data)
    
    #data = data[data['dmg_count'] > 0]

    categories = ['dent', 'scratch', 'missing', 'scraped', 'broken', 'others']
    coco_category, cat_id_dct = gen_coco_cat(categories)    
    
    accumulated_dicts = []
    all_names_lst = []
    all_car_bbox = []
    all_gts = []
    all_preds = []
    all_pred_confs = []
    #all_metrics = []
    all_metrics_per_pred = []
    all_metrics_per_gt = []
    all_gt_lbls = []
    all_pred_lbls = []
    
    complete_csv = pd.read_csv('/home/ubuntu/roisul/Open-GroundingDino/test_results/training_amz_250k_2412_checkpoint0027_merged.csv')
    complete_lst = complete_csv['cdn_url'].tolist()
    start = time.time()
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        print(row['SessID'])
        pc_lst = [4,5,7,8]
        for pc in pc_lst:
            try:
                col_name = f'PhotoCode_{pc}'
                key = row[col_name]
                if key in complete_lst:
                    print('Already in complete csv')
                    continue

                file_name = key.split('/')[-1]
                ori_img = get_img("cdn", key, pc)
                ori_img = torch.from_numpy(ori_img)
                ori_img = ori_img.permute(2, 0, 1)
                width, height = ori_img.shape[-1], ori_img.shape[-2]
                img = img_preprocessing(ori_img)
                #mean = [0.485, 0.456, 0.406]
                #std = [0.229, 0.224, 0.225]
                #img = F.normalize(ori_img.float(), mean=mean, std=std)
                img = img.unsqueeze(dim=0)
                
                photo_lst = row['photo_lst']
                try:
                    damage_name_lst = eval(row["dmg_name_lst"])
                    if type(damage_name_lst) == str:
                        damage_name_lst = eval(row["dmg_name_lst"])
                except:
                    damage_name_lst = eval(row["damage_name_lst"])
                    if type(damage_name_lst) == str:
                        damage_name_lst = eval(row["damage_name_lst"])
                kp_lst  = row['kp_lst']
                component_lst = row['component_lst']

                codes = [int(x['code']) for x in photo_lst]
                freq = Counter(codes)
                idxs = [i for i in range(len(photo_lst)) if int(photo_lst[i]['code']) == pc]
                dmg_kpts = [kp_lst[i] for i in idxs]
                damage_name_lst = [damage_name_lst[i] for i in idxs]
                component_lst = [component_lst[i] for i in idxs]

                scaled_gt_bbox_lst = []
                gt_lbl_lst = []
                gt_lbl_name_lst = []
                for j, cat in enumerate(damage_name_lst):
                    #Text categories
                    if 'DENT' in cat:
                        lbl_cat = 'dent'
                    elif 'SCRATCH' in cat:
                        lbl_cat = 'scratch'
                    elif 'MISSING' in cat:
                        lbl_cat = 'missing'
                    elif 'SCRAPED' in cat:
                        lbl_cat = 'scraped'
                    elif 'BROKEN' in cat:
                        lbl_cat = 'broken'
                    else:
                        lbl_cat = 'others'

                    #Bbox size categories
                    if 'MAJOR' in cat:
                        size_cat = 'large'
                    elif 'MEDIUM' in cat:
                        size_cat = 'medium'
                    elif 'MINOR' in cat:
                        size_cat = 'small'
                    else:
                        size_cat = 'small'
                  
                    category_id = cat_id_dct[lbl_cat]
                    kpts = dmg_kpts[j]
                    bbox = get_coco_bbox(kpts, height, width, size_cat) 
                    scaled_gt_bbox_lst.append(bbox)
                    gt_lbl_lst.append(category_id)
                    gt_lbl_name_lst.append(lbl_cat)
                
                samples = img.to(device)
                ori_samples = ori_img.to(device)
                ori_samples = ori_samples.to(device)
                if ori_samples.shape != (3,1080,1920):
                    print('res not supported, skipping')
                    continue
                
                #img1 = ori_samples.permute(1,2,0)
                #img1 = img1.cpu().float().numpy()
                #results = repvit_main(data_loader, img1, repvit_model, rep_eval_on_format_results, rep_eval_kwargs)
                #mask_tensor = torch.tensor(results[0], dtype=torch.bool).to("cuda")
                #xmin, xmax, ymin, ymax = get_tight_bbox(mask_tensor)
                
                #car_bbox_resp = np.array([ymin.item(),xmin.item(), ymax.item(),xmax.item()])
                #car_bbox_resp = torch.from_numpy(car_bbox_resp).int()
                
                try:
                    car_bbox_resp = get_seg("cdn", key, pc)
                except:
                    print('segmentation failed')
                    continue
                car_bbox_resp = torch.from_numpy(car_bbox_resp).int()
                bs = samples.shape[0]
                input_captions = [caption] * bs
                with torch.cuda.amp.autocast(enabled=args.amp):
                    #Function to create 512x512 crops and stack it as batch (samples)
                    crops, ori_crops, crop_bboxes = create_crops_v3(samples, ori_samples, car_bbox_resp)
                    crops = torch.cat(crops, dim=0)
                    crop_captions = [caption] * len(crops)
                    crop_cap_list = [cat_list] * len(crops)
                    
                    outputs = model(crops, captions=crop_captions)
                    #Eval crop visualization
                    box_threshold = 0.18
                    nms_iou_threshold = 0.2
                    scaled_pred_bbox_lst = []
                    pred_lbl_lst = []
                    pred_conf_lst = []
                    for i in range(len(outputs["pred_logits"])):
                        logits = outputs["pred_logits"].sigmoid()[i]  # (nq, 256)
                        boxes = outputs["pred_boxes"][i]  # (nq, 4)
                        # filter output
                        logits_filt = logits.cpu().clone()
                        boxes_filt = boxes.cpu().clone()
                        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
                        logits_filt = logits_filt[filt_mask]  # num_filt, 256
                        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
                        #print(boxes_filt)
                        if len(boxes_filt) != 0:
                            # get phrase
                            tokenlizer = model.tokenizer
                            tokenized = tokenlizer(caption)
                            # build pred
                            with_logits = True
                            text_threshold = 0.25
                            pred_phrases = []
                            for logit, box in zip(logits_filt, boxes_filt):
                                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                                if with_logits:
                                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                                    pred_lbl_lst.append(pred_phrase)
                                    pred_conf_lst.append(logit.max().item())
                                    
                                    scale_pred = xywh_to_xyxy(box, 512, 512)
                                    crop_bbox = crop_bboxes[i]
                                    scaled_pred_bbox = translate_bbox_to_original_image(scale_pred, crop_bbox[0])
                                    scaled_pred_bbox_lst.append(scaled_pred_bbox)
                                else:
                                    pred_phrases.append(pred_phrase)
                        
                        #Qualitative vis
                        if len(boxes_filt) != 0:
                            pred_bb_lst = []
                            for pred_bbox in boxes_filt:
                                pred = xywh_to_xyxy(pred_bbox, 512, 512)
                                pred = torch.Tensor(pred)
                                pred_bb_lst.append(pred)
                            scale_pred = torch.stack(pred_bb_lst)
                            #scale_pred = torch.tensor(scale_pred).unsqueeze(dim=0)
                    #        #scale_gt = torch.tensor(scale_gt).unsqueeze(dim=0)
                    #        #print(scale_pred)
                    #        #print(scale_gt)
                            
                            crop_vis = ori_crops[i]
                            rand = torch.randint(1,1000,(1,1))[0][0].item()
                            from torchvision.utils import save_image
                            from torchvision.utils import draw_bounding_boxes
                            #dmg_tnsr = torch.cat([scale_pred, scale_gt])
                    #        img_vis = draw_bounding_boxes(crop_vis, scale_gt, colors="red", width=3)
                            img_vis = draw_bounding_boxes(crop_vis, scale_pred,labels=pred_phrases, colors="green", width=3)
                            img_vis = img_vis/255.
                            #save_image(img_vis, f'test_vis_crops/img_{row["file_name"]}_{rand}.png')
                    
                    
                    if not isinstance(scaled_pred_bbox_lst, torch.Tensor):
                        pred = torch.Tensor(scaled_pred_bbox_lst)

                        if pred.dim() == 1:
                            pred = pred.unsqueeze(0)

                    if not isinstance(pred_lbl_lst, np.ndarray):
                        label = np.array(pred_lbl_lst)

                    if not isinstance(pred_conf_lst, torch.Tensor):
                        conf = torch.Tensor(pred_conf_lst)
                    if len(pred[0]) != 0:
                        idx = ops.nms(boxes=pred, scores=conf, iou_threshold=nms_iou_threshold)
                        pred = pred[idx]
                        conf = conf[idx]
                        label = label[idx]

    #                if len(scaled_gt_bbox_lst) != 0:
    #                    scaled_gt_bbox_tnr = torch.Tensor(scaled_gt_bbox_lst)
    #                else:
    #                    scaled_gt_bbox_tnr = torch.Tensor([])
    #                if (len(pred) != 0 and len(scaled_gt_bbox_tnr) != 0):
    #                    img_vis1 = draw_bounding_boxes(ori_img, pred, labels=label, colors="green", width=3)
    #                    img_vis2 = draw_bounding_boxes(ori_img, scaled_gt_bbox_tnr.int(), colors="blue", width=3)
    #                    img_vis = img_vis1 * 0.5 + img_vis2 * 0.5
    #                    img_vis = img_vis/255.
    #                    save_image(img_vis, f'debug_{exp_name}/{file_name}')
                    
                    #unique_bboxes, unique_lbls, unique_confs = remove_overlapping_bboxes(scaled_pred_bbox_lst, pred_lbl_lst, pred_conf_lst, iou_threshold=0.80)
                    #NOTE: Change required here if num categories change
                    #id_map = {'dent':0, 'scratch':1, 'missing':2, 'scraped':3, 'broken':4, 'others':5}

                    #unique_maps = [id_map[lbl] if lbl in id_map else 5 for lbl in unique_lbls]
                    
                    #data_per_predbox, data_per_gtbox = evaluate_detections(unique_bboxes, scaled_gt_bbox_lst, iou_threshold=0.5, distance_threshold=400)
                    #metrics = evaluate_detections(scaled_pred_bbox_lst, scaled_gt_bbox_lst, iou_threshold=0.5, distance_threshold=200)
                    #print(row["file_name"])
                    #print(scaled_gt_bbox_lst)
                    #print(scaled_pred_bbox_lst)
                    
                    all_names_lst.append(file_name)
                    all_car_bbox.append(car_bbox_resp.tolist())
                    all_gts.append(scaled_gt_bbox_lst)
                    all_preds.append(pred.tolist())
                    all_pred_lbls.append(label.tolist())
                    all_pred_confs.append(conf.tolist())
                    #all_metrics_per_pred.append(data_per_predbox)
                    #all_metrics_per_gt.append(data_per_gtbox)
                    all_gt_lbls.append(gt_lbl_name_lst)
                    
                    #precision, recall, ap, mean_ap = get_metrics(final_preds, final_gts, iou_threshold=0.2, num_classes=6)
                    #print(precision, recall, ap, mean_ap)
                    opt = {}
                    opt['cdn_url'] = key
                    opt['fname'] = file_name
                    opt['car_bbox'] = car_bbox_resp.tolist()
                    opt['damage_name_lst'] = damage_name_lst
                    opt['component_lst'] = component_lst
                    opt['gt_bboxes'] = scaled_gt_bbox_lst
                    opt['pred_bboxes'] = pred.tolist()
                    opt['pred_labels'] = label.tolist()
                    opt['pred_confs'] = conf.tolist()
                    accumulated_dicts.append(opt)

                    #if (idx + 1) % interval == 0:
                    result = pd.DataFrame(accumulated_dicts)
                    if not os.path.exists(f"test_results/"):
                        os.makedirs(f"test_results/")
                    path = f"test_results/{exp_name}_{ckpt_name}_amzcontd.csv"
                    result.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)
                    accumulated_dicts.clear() 
            except Exception as e:
                print(e)
                print('Hit error skipping')
                continue
#    end = time.time()
#    print(end-start)
#    import ipdb;ipdb.set_trace()
#    dct = dict()
#    dct["fname"] = all_names_lst
#    dct['car_bbox'] = all_car_bbox
#    dct["gt_bbox"] = all_gts
#    dct["pred_bbox"] = all_preds
#    dct["metrics_per_pred"] = all_metrics_per_pred
#    dct["metrics_per_gt"] = all_metrics_per_gt
#    dct["gt_labels"] = all_gt_lbls
#    dct["pred_labels"] = all_pred_lbls
#    dct["pred_conf"] = all_pred_confs
#    df = pd.DataFrame(dct)
#    df['recall'] = df['metrics'].apply(lambda x: x['recall'])
#    df['accuracy'] = df['metrics'].apply(lambda x: x['accuracy'])
#    df['precision'] = df['metrics'].apply(lambda x: x['precision'])
#
#    df.to_csv(f"test_results/{exp_name}_{ckpt_name}.csv")

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

    dino_args = torch.load("dino_triton_files/args_values.pt")
    dino_args.device = "cuda:0"
    print('Checkpoint path',args.pretrain_model_path)
    dino_args.pretrain_model_path = args.pretrain_model_path
    dino_args.config_file = 'dino_triton_files/cfg_odvg.py'
    dino_args.options={'text_encoder_type': 'dino_triton_files/bert'}
    dino_args.text_encoder_type="dino_triton_files/bert"
    device = dino_args.device
    
    model, criterion, postprocessors = build_model_main(dino_args)
    model.to(device)

    # freeze some layers
    if dino_args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            #new addition begins to freeze everything except attn
            if 'attn' in name:
                parameter.requires_grad_(True)
                print('UnFROZEN' , name)
            #new addition ends
            else:
                for keyword in dino_args.freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        print('FROZEN' , name)
                        break
    if (not dino_args.resume) and dino_args.pretrain_model_path:
        checkpoint = torch.load(dino_args.pretrain_model_path, map_location='cpu')['model']
        _ignorekeywordlist = dino_args.finetune_ignore if dino_args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model.load_state_dict(_tmp_st, strict=False)
    
    #Add repvit
    repvit_model, rep_eval_on_format_results, rep_eval_kwargs = repvit_stuff()
    
    
    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    #NOTE: Change val batch size here
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0) #args.num_workers

    start_time = time.time()
    
    ckpt_path = args.pretrain_model_path
    ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
    exp_name = ckpt_path.split('/')[-2]
    # eval
    evaluate(
        model, criterion, repvit_model, ckpt_name, exp_name, rep_eval_on_format_results, rep_eval_kwargs, postprocessors, data_loader_val, device, args.output_dir, args=args, logger=(logger if args.save_log else None)
    )
    print('eval done')   
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

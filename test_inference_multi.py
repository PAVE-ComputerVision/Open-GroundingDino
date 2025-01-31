
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
from torch.utils.data import DataLoader, DistributedSampler
from crop_utils import create_crops_v3
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder, to_device
import util.misc as utils

from tqdm import tqdm
import datasets
from dmg_dataset import DmgDataset
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
import torchvision.transforms.functional as F
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
            print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                gt_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                gt_box_dist_id = j

        lst1 = [max_iou, gt_box_iou_id, min_dist, gt_box_dist_id]
        data_per_predbox[i] = lst1
    print(data_per_predbox)
    print('---------------------')
    data_per_gtbox = dict()
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0
        min_dist = 1920
        pred_box_iou_id = -1
        pred_box_dist_id = -1
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                pred_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                pred_box_dist_id = j
        lst2 = [max_iou, pred_box_iou_id, min_dist, pred_box_dist_id]
        data_per_gtbox[i] = lst2
    print(data_per_gtbox)
    return data_per_predbox, data_per_gtbox
            
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


@torch.no_grad()
def evaluate(model, 
            criterion, 
            ckpt_name,
            exp_name,
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
    
    categories = ['dent', 'scratch', 'missing', 'scraped', 'broken', 'others']
    coco_category, cat_id_dct = gen_coco_cat(categories)    
    
    all_names_lst = []
    all_gts = []
    all_preds = []
    #all_metrics = []
    all_metrics_per_pred = []
    all_metrics_per_gt = []
    all_gt_lbls = []
    all_pred_lbls = []
    
    for batch_idx, (images, ori_images, car_bboxes) in enumerate(tqdm(data_loader, desc="Processing batches")):
    #for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        start = time.time()
        images = images.to(device)
        ori_images = ori_images.to(device)
        
        #if ori_images.shape != (3,1080,1920):
        #    print('res not supported, skipping')
        #    continue

        bs = images.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):
            #Function to create 512x512 crops and stack it as batch (samples)
            crop_lst = []
            img_ids = []
            a = time.time()
            print('Convert  img to device', a-start)
            for i, sample in enumerate(images):
                sample = sample.unsqueeze(dim=0)
                ori_sample = ori_images[i]
                car_bbox_resp = car_bboxes[i]
                crops, ori_crops, crop_bboxes = create_crops_v3(sample, ori_sample, car_bbox_resp)
                crop_lst.append(crops)
                img_ids.append([i]*len(crops))
            b = time.time()
            print('Get crops',b-a)
            final_crops = [crop for crops in crop_lst for crop in crops]
            final_ids = [img_id for img in img_ids for img_id in img]
            c = time.time()
            print('Get final crops', c-b)
            chunk_size = 64
            chunks = []
            captions = []
            cap_lsts = []
            for i in range(0, len(final_crops), chunk_size):
                chunk = final_crops[i:i+chunk_size]
                crops = torch.stack(chunk, dim=0)
                crop_captions = [caption] * len(chunk)
                crop_cap_list = [cat_list] * len(chunk)
                #print(len(crops))
                d1 = time.time()
                outputs = model(crops, captions=crop_captions)
                d2 = time.time()
                print('Forward pass', d2-d1)
            d = time.time()
            print('Run inference', d-c)
            print('-----')
            #Eval crop visualization
            box_threshold = 0.3
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
                print(boxes_filt)
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
            

            scaled_pred_bbox_tnr = torch.Tensor(scaled_pred_bbox_lst)
            if len(scaled_gt_bbox_lst) != 0:
                scaled_gt_bbox_tnr = torch.Tensor(scaled_gt_bbox_lst)
            else:
                scaled_gt_bbox_tnr = torch.Tensor([])
            if (len(scaled_pred_bbox_tnr) != 0 and len(scaled_gt_bbox_tnr) != 0):
                img_vis1 = draw_bounding_boxes(ori_img, scaled_pred_bbox_tnr, labels=pred_lbl_lst, colors="green", width=3)
                img_vis2 = draw_bounding_boxes(ori_img, scaled_gt_bbox_tnr.int(), colors="blue", width=3)
                img_vis = img_vis1 * 0.5 + img_vis2 * 0.5
                img_vis = img_vis/255.
                save_image(img_vis, f'test_vis/{row["file_name"]}')
            continue
            unique_bboxes, unique_lbls, unique_confs = remove_overlapping_bboxes(scaled_pred_bbox_lst, pred_lbl_lst, pred_conf_lst, iou_threshold=0.80)
#            #NOTE: Change required here if num categories change
#            id_map = {'dent':0, 'scratch':1, 'missing':2, 'scraped':3, 'broken':4, 'others':5}
#
#            unique_maps = [id_map[lbl] if lbl in id_map else 5 for lbl in unique_lbls]
#            
#            data_per_predbox, data_per_gtbox = evaluate_detections(unique_bboxes, scaled_gt_bbox_lst, iou_threshold=0.5, distance_threshold=200)
#            #metrics = evaluate_detections(scaled_pred_bbox_lst, scaled_gt_bbox_lst, iou_threshold=0.5, distance_threshold=200)
#            print(row["file_name"])
#            print(scaled_gt_bbox_lst)
#            print(scaled_pred_bbox_lst)
#            
#            all_names_lst.append(row["file_name"])
#            all_gts.append(scaled_gt_bbox_lst)
#            all_preds.append(unique_bboxes)
#            all_metrics_per_pred.append(data_per_predbox)
#            all_metrics_per_gt.append(data_per_gtbox)
#            all_gt_lbls.append(gt_lbl_name_lst)
#            all_pred_lbls.append(pred_lbl_lst)
#            #precision, recall, ap, mean_ap = get_metrics(final_preds, final_gts, iou_threshold=0.2, num_classes=6)
#            #print(precision, recall, ap, mean_ap)
#    dct = dict()
#    dct["fname"] = all_names_lst
#    dct["gt_bbox"] = all_gts
#    dct["pred_bbox"] = all_preds
#    dct["metrics_per_pred"] = all_metrics_per_pred
#    dct["metrics_per_gt"] = all_metrics_per_gt
#    dct["gt_labels"] = all_gt_lbls
#    dct["pred_labels"] = all_pred_lbls
#    df = pd.DataFrame(dct)
##    df['recall'] = df['metrics'].apply(lambda x: x['recall'])
##    df['accuracy'] = df['metrics'].apply(lambda x: x['accuracy'])
##    df['precision'] = df['metrics'].apply(lambda x: x['precision'])
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
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)
    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            #new addition begins to freeze everything except attn
            if 'attn' in name:
                parameter.requires_grad_(True)
                print('UnFROZEN' , name)
            #new addition ends
            else:
                for keyword in args.freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        print('FROZEN' , name)
                        break
    # freeze some layers
    # if args.freeze_keywords is not None:
    #     for name, parameter in model.named_parameters():
    #         for keyword in args.freeze_keywords:
    #             if keyword in name:
    #                 parameter.requires_grad_(False)
    #                 break
    logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    logger.debug("build dataset ... ...")
    
    csv_path = '/home/ubuntu/roisul/qc_dmg_250k/test/metadata.csv'
    dataset_val = DmgDataset(csv_path)
    data_loader_val = DataLoader(dataset_val, batch_size=4, num_workers=0)
    #dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    #NOTE: Change val batch size here
    #data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
    #                             drop_last=False, collate_fn=utils.collate_fn, num_workers=0) #args.num_workers

    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

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

    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)
    
    ckpt_path = args.pretrain_model_path
    ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
    exp_name = ckpt_path.split('/')[-2]
    # eval
    evaluate(
        model, criterion, ckpt_name, exp_name, postprocessors, data_loader_val, base_ds, device, args.output_dir,
        wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
    )
    print('eval done')   
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

#NOTE: Repvit merge modifications here
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from datasets import build_dataset, get_coco_api_from_dataset
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

import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import os.path as osp
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset as rep_build_dataset

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

def xywh_to_xyxy(box,H,W):
    # from 0..1 to 0..W, 0..H
    box = box * torch.Tensor([W, H, W, H])
    # from xywh to xyxy
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    # draw
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return x0, y0, x1, y1

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        #box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        #box[:2] -= box[2:] / 2
        #box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        #x0, y0, x1, y1 = box
        #x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        x0, y0, x1, y1 = xywh_to_xyxy(box,H,W)
        print(f'BBOX {x0, y0, x1, y1}' )
        print(label)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
        # draw.text((x0, y0), str(label), fill=color)

        # font = ImageFont.load_default()
        # if hasattr(font, "getbbox"):
        #     bbox = draw.textbbox((x0, y0), str(label), font)
        # else:
        #     w, h = draw.textsize(str(label), font)
        #     bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        # draw.rectangle(bbox, fill=color)
        # draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    ori_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
        ]
    )
    ori_tnsr = ori_transform(image_pil)

    return image_pil, image, ori_tnsr


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

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

def get_grounding_output(model, repvit_model, rep_eval_on_format_results, rep_eval_kwargs, data_loader, image, ori_tnsr, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    ori_samples = ori_tnsr.to(device)
    with torch.no_grad():
        assert ori_samples.shape == (3,1080,1920)
        img1 = ori_samples.permute(1,2,0)
        img1 = img1.cpu().float().numpy()
        results = main(data_loader, img1, repvit_model, rep_eval_on_format_results, rep_eval_kwargs)
        mask_tensor = torch.tensor(results[0], dtype=torch.bool).to("cuda")
        try:
            xmin, xmax, ymin, ymax = get_tight_bbox(mask_tensor)
        except:
            import ipdb;ipdb.set_trace()
        car_bbox_resp = np.array([ymin.item(),xmin.item(), ymax.item(),xmax.item()])
        car_bbox_resp = torch.from_numpy(car_bbox_resp).int()
        import ipdb;ipdb.set_trace()
            

    #TODO: Do loop
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    
    
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

    args = parser.parse_args()
    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    
    lst = ["/home/ubuntu/roisul/qc_dmg_test/test/7-9c6af68d-679b-49b2-96f5-ea16937d9a6f-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9c6af68d-679b-49b2-96f5-ea16937d9a6f-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9c1a8bd7-06c9-4465-ba47-d73e3f4422d0-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9bba8760-c67a-434a-8c31-a314a484e7b2-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/4-9bf135c8-8cbe-496f-b272-bb9c93d58c19-1080x1920.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9c6af68d-679b-49b2-96f5-ea16937d9a6f-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9c635fd5-be9a-4577-877b-f2fc4b4a5ffc-1920x1080.jpg",
            "/home/ubuntu/roisul/qc_dmg_test/test/7-9c6af68d-679b-49b2-96f5-ea16937d9a6f-1920x1080.jpg"]
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image, ori_tnsr = load_image(image_path)
    # load model

    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    repvit_model, rep_eval_on_format_results, rep_eval_kwargs = repvit_stuff()
    
    dataset_meta_val_0 = {'root': '/home/ubuntu/roisul', 'anno': '/home/ubuntu/roisul/utils/test_annot_coco.json', 'label_map': None, 'dataset_mode': 'coco'}
    dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta_val_0)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0) #args.num_workers
    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")


    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, repvit_model, rep_eval_on_format_results, rep_eval_kwargs, data_loader_val, image, ori_tnsr, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
    )
    print("BOXES",boxes_filt)
    # visualize pred
    size = image_pil.size
    print("SIZE",size)
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    save_path = os.path.join(output_dir, image_path.split('/')[-1])
    image_with_box.save(save_path)
    print(f"BOXES {boxes_filt}")
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")

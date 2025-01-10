import argparse
import os
import io
import sys
import time

import numpy as np
import torch
import requests
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def xywh_to_xyxy(box,H,W):

    # from 0..1 to 0..W, 0..H
    box = box * torch.Tensor([W, H, W, H])
    # from xywh to xyxy
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    # draw
    x0, y0, x1, y1 = box.tolist()
    #x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return x0, y0, x1, y1


def download_from_cdn(url: str) -> bytes:
    res = requests.get(url, stream=True)
    data = b""
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            data += chunk
    return data

def get_img(key):
    byte_data = download_from_cdn(key)
    img_pil = Image.open(io.BytesIO(byte_data))
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(img_pil, None)  # 3, h, w
    return img_pil, image

def get_bbox(tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    
    # draw boxes and masks
    for box in boxes:
        # from 0..1 to 0..W, 0..H
        box = torch.Tensor(box) * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        bbox = [x0, y0, x1, y1]
    return bbox

#for prod
import torchvision.transforms as transform

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    #assert len(boxes) == len(labels), "boxes and labels must have same length"
    
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box in boxes:
        # from 0..1 to 0..W, 0..H
        box = torch.Tensor(box) * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), "blur", font)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    
    #image_pil_with_mask = pre_mod(image_pil,bbox)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    print(f"THE INFERENCE TIME = {time.time() - start}")
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
    #parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
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
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    #image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    #image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    #df = pd.read_parquet('demo/AMZ_DF_V5.parquet')
    #df = pd.read_csv('~/roisul/utils/vin_val.csv')
    #df = pd.read_csv('~/roisul/utils/val_seg.csv')
    df = pd.read_csv('~/data/seg_csvs/failed_cases_repvit.csv')
    #completed = pd.read_csv('dino_inference.csv')
    df = df.iloc[int(args.idx)::int(args.chunks)]
    #done_lst = completed.input.tolist()
    done_lst = ['']
    #pc_lst = [4,5,7,8,10,11,12,13]
    #pc_cols = []
    #for pc in pc_lst:
        #pc_cols.append(f'PhotoCode_{pc}')
    results = []
    a = time.time()
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        cnt = 0
        #for col in pc_cols:
            #cdn_url = row[col]
            #if cdn_url in done_lst:
                #print('skipped')
                #continue
        try:
            #cdn_urls
            cdn_url = row['input']
            image_pil, image = get_img(cdn_url)
            #load image
        #image_path = row['Filename']
        #image_pil, image = load_image(image_path)
        except:
            continue
        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )

        # visualize pred
        size = image_pil.size
        W,H = size
        try:
            x0, y0, x1, y1 = xywh_to_xyxy(boxes_filt[0],H,W)
            boxes = [x0, y0, x1, y1]
        except:
            boxes = [17, 17, 17, 17]
            pred_phrases = '-17'
        try:
            pred_dict = {
                    "boxes":boxes,
                    #"boxes": boxes_filt.tolist(),
                    "size": [size[1], size[0]],  # H,W
                    "labels": pred_phrases,
                }
        except:
            continue
        print(pred_dict)
        out_dct = {"input": cdn_url, "output": pred_dict, "bbox":boxes}
        #image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        #image_with_box.save(os.path.join(output_dir, f"pred{cnt}.jpg"))
        results.append(out_dct)
        cnt += 1
        #print(out_dct)
        #out_df = pd.DataFrame(results)
        #out_df.to_csv(f"dino_inference_seg_{args.idx}.csv",index=False)
    b = time.time()
    print(b-a)
            

            # import ipdb; ipdb.set_trace()
            #image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            #image_with_box.save(os.path.join(output_dir, "pred.jpg"))

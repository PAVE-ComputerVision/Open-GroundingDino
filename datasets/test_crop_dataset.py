"""
Script to test and time dataset objects that create vehicle bounding box crops

This is to visualize and make sure that dataset objects return correct results
"""
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler

import sys
sys.path.append('/home/ubuntu/roisul/Open-GroundingDino/')
import util.misc as utils
from main import get_args_parser
from util.slconfig import DictAction, SLConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
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

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]
    
    dataset_train = build_crop_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
    sampler_val = torch.utils.data.SequentialSampler(dataset_train) #Change to Random after debugging
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=0) #args.num_workers

    for samples_tnsr, ori_samples_tnsr, targets in data_loader_train:
        print(samples_tnsr)
        print(ori_samples_tnsr)
        print(targets)
        samples_tnsr = torch.concat(samples_tnsr)
        ori_samples_tnsr = torch.concat(ori_samples_tnsr)
        import ipdb;ipdb.set_trace()
        
        crops = torch.cat(final_crops, dim=0)
        crop_captions = [captions[0]] * len(final_crops)
        crop_cap_list = [cap_list[0]] * len(final_crops)
        outputs = model(crops, captions=crop_captions)
        loss_dict = criterion(outputs, final_targets, crop_cap_list, crop_captions)



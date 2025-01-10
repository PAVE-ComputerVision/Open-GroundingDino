import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

class DmgDataset(Dataset):
    """
    PyTorch Dataset to load images from a list of file paths.

    Parameters:
    - file_paths: List of strings, where each string is the path to an image.
    - transform: Optional torchvision transforms to apply to the images.
    """
    def __init__(self, file_path):
        self.csv_path = file_path
        self.data = pd.read_csv(self.csv_path)
        self.data = self.data[self.data['dmg_count'] > 0]
        #self.transform = transform

    def __len__(self):
        return 100#len(self.data)

    def __getitem__(self, idx):
        # Load the image from the file path
        row = self.data.iloc[idx]
        fname = row["file_name"]
        dmg_kpts = json.loads(row['dmg_kpts'])
        car_bbox = json.loads(row['car_bbox'])
        dir_name = os.path.dirname(self.csv_path)
        img_path = os.path.join(dir_name, fname)
        try:
            img = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        except FileNotFoundError:
            raise RuntimeError(f"Image at {img_path} not found.")
        
        # Apply transformations if specified
        #if self.transform:
        #    image = self.transform(image)
        width, height = img.size

        transform = T.Compose([
                T.PILToTensor()
                ])

        ori_img = transform(img)

        img = F.to_tensor(img)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = F.normalize(img, mean=mean, std=std)
        #img = img.unsqueeze(dim=0)
        
        
#        scaled_gt_bbox_lst = []
#        gt_lbl_lst = []
#        gt_lbl_name_lst = []
#        for j, cat in enumerate(dmg_name_lst):
#            #Text categories
#            if 'DENT' in cat:
#                lbl_cat = 'dent'
#            elif 'SCRATCH' in cat:
#                lbl_cat = 'scratch'
#            elif 'MISSING' in cat:
#                lbl_cat = 'missing'
#            elif 'SCRAPED' in cat:
#                lbl_cat = 'scraped'
#            elif 'BROKEN' in cat:
#                lbl_cat = 'broken'
#            else:
#                lbl_cat = 'others'
#
#            #Bbox size categories
#            if 'MAJOR' in cat:
#                size_cat = 'large'
#            elif 'MEDIUM' in cat:
#                size_cat = 'medium'
#            elif 'MINOR' in cat:
#                size_cat = 'small'
#            else:
#                size_cat = 'small'
#          
#            category_id = cat_id_dct[lbl_cat]
#            kpts = dmg_kpts[j]
#            bbox = get_coco_bbox(kpts, height, width, size_cat) 
#            scaled_gt_bbox_lst.append(bbox)
#            gt_lbl_lst.append(category_id)
#            gt_lbl_name_lst.append(lbl_cat)
        samples = img
        #ori_samples = ori_img.to(device)
        #ori_samples = ori_samples.to(device)
        
        car_bbox = torch.Tensor(car_bbox)

        ymin, ymax, xmin, xmax = car_bbox
        car_bbox_resp = np.array([ymin.item(),xmin.item(), ymax.item(),xmax.item()])
        car_bbox_resp = torch.from_numpy(car_bbox_resp).int()
        return samples, car_bbox_resp

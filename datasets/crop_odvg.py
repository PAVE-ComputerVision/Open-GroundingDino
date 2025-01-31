from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import datasets.transforms as T
import torchvision.transforms as transforms
sys.path.append('/home/ubuntu/roisul/Open-GroundingDino/')
from crop_utils import create_crops_v3
from engine_new import xywh_to_xyxy, translate_bounding_box, normalize_bbox, is_inside

class CropODDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.max_labels = max_labels
        self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()
        self.device = "cuda"

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with  open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        print(f"  == total labels: {len(self.label_map)}")

    def pil_to_tensor(img):
        return img_tensor

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        #print(abs_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        
        transform = transforms.Compose([
                transforms.PILToTensor()
                ])
        ori_img_tnsr = transform(image)
            
        anno = meta["detection"]
        instances = [obj for obj in anno["instances"]]
        boxes = [obj["bbox"] for obj in instances]
        
        #class_weights =  [obj['weight'] for obj in instances]
        car_bboxes = [obj["car_bbox"] for obj in instances]
        car_bboxes = [car_bboxes[0]]

        # generate vg_labels
        # pos bbox labels
        ori_classes = [str(obj["label"]) for obj in instances]
        pos_labels = set(ori_classes)
        # neg bbox labels 
        neg_labels = self.label_index.difference(pos_labels)
         
        vg_labels = list(pos_labels)
        num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
        if num_to_add > 0:
            #vg_labels.extend(random.sample(neg_labels, num_to_add))
            vg_labels.extend(random.sample(sorted(neg_labels), num_to_add)) #changed to solve multiclass training bug
        
        # shuffle
        for i in range(len(vg_labels)-1, 0, -1):
            j = random.randint(0, i)
            vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

        caption_list = [self.label_map[lb] for lb in vg_labels]
        caption_dict = {item:index for index, item in enumerate(caption_list)}

        caption = ' . '.join(caption_list) + ' .'
        classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor(classes, dtype=torch.int64)

        car_bboxes = torch.as_tensor(car_bboxes, dtype=torch.float32).reshape(-1, 4)
        
        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        target["car_bboxes"] = car_bboxes
        #target["class_weights"] = class_weights
        # size, cap_list, caption, bboxes, labels
        #print(target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        

        samples = image
        samples = samples#.to(self.device)
        samples = samples.unsqueeze(dim=0)
        car_bbox_resp = target["car_bboxes"][0].int()
        crops, ori_crops, crop_bboxes = create_crops_v3(samples, ori_img_tnsr, car_bbox_resp)

        scaled_dmg_bboxes = []
        dmg_bboxes = target['boxes']
        dmg_labels = target['labels']
        img_size = target['size']
        for box in dmg_bboxes:
            new_bb = []
            bb = xywh_to_xyxy(box, img_size[0], img_size[1])
            scaled_dmg_bboxes.append(bb) 
        
        crop_targets = []
        batch_crops = []
        batch_ori_crops = []
        for crop, ori_crop, crop_bbox in zip(crops, ori_crops, crop_bboxes):
            tgt = dict()
            final_unnorm = []
            final_dmg_bboxes = []
            labels = []
            tgt["size"] = torch.Tensor([crop.shape[2], crop.shape[3]]).int()#.to(self.device)
            for i, dmg_bbox in enumerate(scaled_dmg_bboxes):
                if is_inside(dmg_bbox, crop_bbox):
                    relative_dmg_bbox = translate_bounding_box(dmg_bbox, crop_bbox)
                    relative_dmg_bbox = torch.Tensor(relative_dmg_bbox)
                    final_unnorm.append(relative_dmg_bbox)
                    new_bbox = normalize_bbox(relative_dmg_bbox, tgt["size"])

                    if min(new_bbox) < 0:
                        continue
                    final_dmg_bboxes.append(new_bbox)

                    lbl = dmg_labels[i].item()
                    labels.append(lbl)
                else:
                    pass

            if len(final_dmg_bboxes) > 0:
                tgt["boxes"] = torch.stack(final_dmg_bboxes)#.to(self.device)
                tgt["unnorm"] = torch.stack(final_unnorm)#.to(self.device)
            else:
                tgt["boxes"] = torch.Tensor([])#.to(self.device) #TODO: Check if no dmg values are set to empty array
            tgt["labels"] = torch.Tensor(labels).int()#.to(self.device)
            crop_targets.append(tgt)
            batch_crops.append(crop)
            batch_ori_crops.append(ori_crop)

        final_crops = []
        final_ori_crops = []
        final_targets = []
        for i in range(len(batch_crops)):
            if len(crop_targets[i]['boxes']) != 0:
                final_crops.append(batch_crops[i])
                final_ori_crops.append(batch_ori_crops[i])
                final_targets.append(crop_targets[i])

                #Target visualiation
                #scaled_dmg_tnsr = crop_targets[i]['unnorm']
                #crop_img = draw_bounding_boxes(batch_ori_crops[i], scaled_dmg_tnsr, width=3, colors = "red")
                #crop_img = crop_img /255.0
                #save_image(crop_img, f'debug/crop_img_{i}.png')

        #NOTE:Skip if no bounding boxes are usable
        #if len(final_crops) == 0:
        #    continue

        max_num_crops = 16
        if len(final_crops) > max_num_crops:
            idxs = random.sample(range(len(final_crops)), max_num_crops)
            final_crops = [final_crops[i] for i in idxs]
            final_ori_crops = [final_ori_crops[i] for i in idxs]
            final_targets = [final_targets[i] for i in idxs]
        #print(len(final_crops)) 
        
        if len(final_crops) > 0:
            crops = torch.cat(final_crops, dim=0)
            final_ori_crops = torch.stack(final_ori_crops, dim=0)
        else:
            crop = torch.Tensor([])
            final_ori_crops = torch.Tensor([])
        return crops, final_ori_crops, final_targets
    

    def __len__(self) -> int:
        if len(self.metas) < 10000:
            return len(self.metas)
        else:
            return 10000
#        return len(self.metas)

class CropODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with  open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def pil_to_tensor(img):
        return img_tensor

    def __getitem__(self, index: int):

        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        print(abs_path)
        import ipdb;ipdb.set_trace()
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        
        transform = transforms.Compose([
                transforms.PILToTensor()
                ])
        ori_img_tnsr = transform(image)
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            
            #class_weights =  [obj['weight'] for obj in instances]
            car_bboxes = [obj["car_bbox"] for obj in instances]
            car_bboxes = [car_bboxes[0]]

            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels 
            neg_labels = self.label_index.difference(pos_labels)
             
            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
            if num_to_add > 0:
                #vg_labels.extend(random.sample(neg_labels, num_to_add))
                vg_labels.extend(random.sample(sorted(neg_labels), num_to_add)) #changed to solve multiclass training bug
            
            # shuffle
            for i in range(len(vg_labels)-1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item:index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)

            car_bboxes = torch.as_tensor(car_bboxes, dtype=torch.float32).reshape(-1, 4)
            #class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list  = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            caption_list = uni_caption_list
        print(self.dataset_mode)
        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        target["car_bboxes"] = car_bboxes
        #target["class_weights"] = class_weights
        # size, cap_list, caption, bboxes, labels
        print(self.transforms)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, ori_img_tnsr, target
    

    def __len__(self) -> int:
        if len(self.metas) < 10000:
            return len(self.metas)
        else:
            return 10000
#        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            #T.RandomHorizontalFlip(),
            #T.RandomSelect(
            #    T.RandomResize(scales, max_size=max_size),
            #    T.Compose([
            #        T.RandomResize(scales2_resize),
            #        T.RandomSizeCrop(*scales2_crop),
            #        T.RandomResize(scales, max_size=max_size),
            #    ])
            #),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            #T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_crop_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = CropODDataset(img_folder, ann_file, label_map, max_labels=args.max_labels,
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
    )
    return dataset


if __name__=="__main__":
    dataset_vg = ODVGDataset("path/GRIT-20M/data/","path/GRIT-20M/anno/grit_odvg_10k.jsonl",)
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)] 
    print(data)
    dataset_od = ODVGDataset("pathl/V3Det/",
        "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
        "path/V3Det/annotations/v3det_label_map.json",
    )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)] 
    print(data)

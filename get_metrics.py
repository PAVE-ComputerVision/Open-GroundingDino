import torch
import pandas as pd
import numpy as np
from collections import Counter
from torchvision.utils import save_image, draw_bounding_boxes
from util.infer_utils import get_img, download_from_cdn, size_checks

def filter_bboxes(row, thresh):
    confs = row['pred_confs']
    bboxes = row['pred_bboxes']
    new_bboxes = []
    for i in range(len(confs)):
        if confs[i] > thresh:
            new_bboxes.append(bboxes[i])
    return new_bboxes

def bbox_threshold_search(x):
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]:
        x['filtered_bboxes'] = x.apply(filter_bboxes, thresh=thresh, axis=1)
        x['num_filtered'] = x['filtered_bboxes'].apply(lambda x: len(x))
        nodmgs = x[x['num_gts'] == 0]
        dmgs = x[x['num_gts'] > 0]
        correct_count = nodmgs[nodmgs['num_filtered']==0].shape[0]
        correct_dmg_count = dmgs[dmgs['num_filtered']>0].shape[0]
        print(f"Threshold {thresh}: # nodmgs correct count: {correct_count}/{nodmgs.shape[0]} | #dmg correct count: {correct_dmg_count}/{dmgs.shape[0]} ")
        print('----------')


def get_vis(df):
    for i, row in df.iterrows():
        url = row['cdn_url']
        pc = row['pc']
        gt_bboxes = row['gt_bboxes']
        pred_bboxes = row['filtered_bboxes']
        
        img = get_img('cdn', url, pc)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        
        if len(pred_bboxes) != 0:
            pred_tnsr = torch.Tensor(pred_bboxes).int()
        else:
            pred_tsnr = torch.Tensor([])
        if len(gt_bboxes) != 0:
            gt_tnsr = torch.Tensor(gt_bboxes).int()
        else:
            gt_tnsr = torch.Tensor([])
        
        if len(pred_tnsr) != 0:
            img_vis1 = draw_bounding_boxes(img, pred_tnsr, colors="green", width=3)
        if len(gt_tnsr) != 0:
            img_vis2 = draw_bounding_boxes(img, gt_tnsr, colors="blue", width=3)
        if len(pred_tnsr) != 0 and len(gt_tnsr) != 0:
            img_vis = img_vis1 * 0.5 + img_vis2 * 0.5
        elif len(pred_tnsr) != 0 and not len(gt_tnsr) != 0:
            img_vis = img_vis1
        elif not len(pred_tnsr) != 0 and len(gt_tnsr) != 0:
            img_vis = img_vis2
        
        img_vis = img_vis/255.
        print(row['fname'])
        save_image(img_vis, f"vis/{row['fname']}")
    
def get_dmg_names(df):
    data_pth = f"/home/ubuntu/roisul/241129.parquet"
    data = pd.read_parquet(data_pth)
    data = data.sample(100000, random_state=42)
    dmg_names = []
    comp_names = []
    for i, val in df.iterrows():
        gt_bboxes = val['gt_bboxes']
        sess = val['session']
        pc = val['pc']
        row = data[data['SessID'] == sess]
        if len(gt_bboxes) == 0:
            dmg_names.append([])
            comp_names.append([])
        else:
            photo_lst = row['photo_lst']
            if len(photo_lst) == 0:
                dmg_names.append([])
                comp_names.append([])
                continue
            else:
                photo_lst = eval(photo_lst.item())
                damage_name_lst = eval(row["damage_name_lst"].item())
                component_lst = eval(row['component_lst'].item())

                codes = [int(x['code']) for x in photo_lst]
                freq = Counter(codes)
                idxs = [i for i in range(len(photo_lst)) if int(photo_lst[i]['code']) == pc]
                damage_name_lst = [damage_name_lst[i] for i in idxs]
                component_lst = [component_lst[i] for i in idxs]
                assert len(damage_name_lst) == len(gt_bboxes)
                dmg_names.append(damage_name_lst)
                comp_names.append(component_lst)
                print(i)
    df['damage_name_lst'] = dmg_names
    df['component_lst'] = comp_names
    return df
if __name__ == '__main__':
    #Load 2 csvs
    #infer_27 = pd.read_csv('test_results/training_amz_250k_2412_checkpoint0027_merged.csv')
    #infer_14 = pd.read_csv('test_results/training_amz_250k_2412_checkpoint0014_dmg_names.csv')
    #Get common rows
    #common = infer_14.merge(infer_27, on='cdn_url')
    #Get common cdn urls
    #common_cdns = common['cdn_url'].tolist()

    #NOTE: Filter the required csv for metrics
    #x = infer_27[infer_27['cdn_url'].isin(common_cdns)]
    x = pd.read_csv('test_results/inference_v2_dino_log.csv')
    x = x[x['cdn_url'].str.contains('AM')]
    #Preprocess
    x['session'] = x['cdn_url'].apply(lambda x: x.split('/')[-3])
    x['pc'] = x['fname'].apply(lambda x: int(x.split('-')[0]))
    x['gt_bboxes'] = x['gt_bboxes'].apply(lambda x: eval(x))
    x['pred_bboxes'] = x['pred_bboxes'].apply(lambda x: eval(x))
    x['pred_confs'] = x['pred_confs'].apply(lambda x: eval(x))
    x['num_gts'] = x['gt_bboxes'].apply(lambda x: len(x))
    x['num_preds'] = x['pred_bboxes'].apply(lambda x: len(x))
    #x = get_dmg_names(x)
    x = x[x['pc'].isin([4,7])]
    bbox_threshold_search(x)
    thresh = 0.30
    x['filtered_bboxes'] = x.apply(filter_bboxes, thresh=thresh, axis=1)
    x['num_filtered'] = x['filtered_bboxes'].apply(lambda x: len(x))
    dmgs = x[x['num_gts']> 0]
    print(dmgs['num_gts'].describe())
    print(dmgs['num_filtered'].describe())
    
    nodmgs = x[x['num_gts'] == 0]
    print(nodmgs['num_gts'].describe())
    print(nodmgs['num_filtered'].describe())
    
    correct_count = nodmgs[nodmgs['num_filtered']==0].shape[0]
    correct_dmg_count = dmgs[dmgs['num_filtered']>0].shape[0]
    print(f"Threshold {thresh}: # nodmgs correct count: {correct_count}/{nodmgs.shape[0]} | #dmg correct count: {correct_dmg_count}/{dmgs.shape[0]} ")
    wrong_preds = nodmgs[nodmgs['num_filtered']>0]
    import ipdb;ipdb.set_trace()
    get_vis(wrong_preds)




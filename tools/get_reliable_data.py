import os
import requests
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from inference_from_csv import get_bbox


def draw_bounding_boxes(cdn_url, boxes, output_path=None):
    """
    Draw bounding boxes on an image.

    Parameters:
    - image_path: Path to the image file.
    - boxes: List of bounding boxes in xyxy format.
    - output_path: Path to save the image with bounding boxes (optional).
    """
    response = requests.get(cdn_url)
    if response.status_code == 200:
        image_stream = BytesIO(response.content)
        image = Image.open(image_stream)

        # Load the image
        #image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Draw each bounding box
        a_x1, a_y1, a_x2, a_y2 = boxes[0]
        draw.rectangle([a_x1, a_y1, a_x2, a_y2], outline="red", width=3)
        
        b_x1, b_y1, b_x2, b_y2 = boxes[1]
        draw.rectangle([b_x1, b_y1, b_x2, b_y2], outline="blue", width=3)

        # Display the image
        #plt.imshow(image)
        plt.axis('off')  # Hide the axis
        #plt.show()

        # Optionally save the image with bounding boxes
        if output_path:
            image.save(output_path)

def l2_loss_top_left(row):
    """
    Calculate the L2 loss between the top-left coordinates of two bounding boxes.
    
    Parameters:
    bbox1 (list of floats): Bounding box in [x1, y1, x2, y2] format.
    bbox2 (list of floats): Bounding box in [cx, cy, w, h] format.
    
    Returns:
    float: The L2 loss between the top-left coordinates of the bounding boxes.
    """
    
    bbox1 = row['dino_bbox']
    bbox2 = row['swints_bbox']

    #print(bbox1)
    #print(bbox2)
    
    if len(bbox1) != 4 or len(bbox2) != 4:

        import ipdb;ipdb.set_trace()
        return -17
    
    # Extract top-left coordinates of the first bbox
    x1_1, y1_1, _, _ = bbox1
    x1_2, y1_2, _, _ = bbox2
    
    # Convert second bbox to top-left coordinates
    #cx2, cy2, w2, h2 = bbox2
    #x1_2 = cx2 - w2 / 2
    #y1_2 = cy2 - h2 / 2
    
    # Calculate L2 loss
    loss = np.sqrt((x1_1 - x1_2) ** 2 + (y1_1 - y1_2) ** 2)
    return loss

def l2_loss_bottom_right(row):
    """
    Calculate the L2 loss between the bottom-right coordinates of two bounding boxes.
    
    Parameters:
    bbox1 (list of floats): Bounding box in [x1, y1, x2, y2] format.
    bbox2 (list of floats): Bounding box in [cx, cy, w, h] format.
    
    Returns:
    float: The L2 loss between the bottom-right coordinates of the bounding boxes.
    """
    bbox1 = row['dino_bbox']
    bbox2 = row['swints_bbox']
    
    #print(bbox1)
    #print(bbox2)
    
    if len(bbox1) != 4 or len(bbox2) != 4:
        return -17
    # Extract bottom-right coordinates of the first bbox
    _, _, x2_1, y2_1 = bbox1
    _, _, x2_2, y2_2 = bbox2
    
    # Convert second bbox to bottom-right coordinates
    #cx2, cy2, w2, h2 = bbox2
    #x2_2 = cx2 + w2 / 2
    #y2_2 = cy2 + h2 / 2
    
    # Calculate L2 loss
    loss = np.sqrt((x2_1 - x2_2) ** 2 + (y2_1 - y2_2) ** 2)
    
    return loss

def get_session_from_fname(row):
    fname = row['input']
    return fname.split('/')[-3].split('-')[-1]

def get_pc_from_fname(row):
    fname = row['input']
    return fname.split('/')[-1].split('-')[0]

def load_dino_df(path):
    #if not os.path.exists('/home/ubuntu/GroundingDINO/demo/dino_df_processed.csv'):
    dino_df = pd.read_csv(path)
    dino_df['session'] = dino_df.apply(get_session_from_fname, axis=1)
    dino_df['pc'] = dino_df.apply(get_pc_from_fname, axis=1)
    #skip if pc col not int; can be pc for some filename format
    dino_df = dino_df[dino_df['pc'].str.isdigit()]
    dino_df['pc'] = dino_df['pc'].apply(lambda x: int(x))
    dino_df['output'] = dino_df['output'].apply(lambda x: eval(x))
    dino_df['dino_bbox'] = dino_df['output'].apply(get_bbox)
    dino_df.to_csv('/home/ubuntu/GroundingDINO/demo/dino_df_processed.csv')
    #else:
    #    dino_df = pd.read_csv('/home/ubuntu/GroundingDINO/demo/dino_df_processed.csv')
    return dino_df

def melt_yolo_df(df):
    melted = x.melt(id_vars=['Session'], value_vars=['PhotoCode_4','PhotoCode_5','PhotoCode_7','PhotoCode_8','PhotoCode_10','PhotoCode_11','PhotoCode_12','PhotoCode_13'])
    melted['pc'] = melted['variable'].apply(lambda x: int(x.split('_')[0]))
    return melted

if __name__ == '__main__':
    #Load dino dataset
    #dino_df = load_dino_df('/home/ubuntu/GroundingDINO/dino_inference.csv')
    
    #Load melted yolo dataset
    # melted_yolo = load_yolo_df('/home/ubuntu/GroundingDINO/demo/AMZ_Captures_onlyones_train_bbox.csv')
    #melted_yolo = pd.read_csv('/home/ubuntu/GroundingDINO/demo/AMZ_Captures_onlyones_train_bbox.csv')
    
    #seg_df = pd.read_csv('/home/ubuntu/GroundingDINO/fast_infer_seg_with_bbox.csv')
    #merged = dino_df.merge(seg_df, left_on='input', right_on='filename')
    #merged = merged[~merged['bbox'].str.contains('-68')]
    #merged['bbox'] = merged['bbox'].apply(lambda x: x.replace('.',','))
    #merged['bbox'] = merged['bbox'].apply(lambda x: eval(x))
    
    #merged = pd.read_parquet('/home/ubuntu/GroundingDINO/dino_seg_bbox_merge.parquet')
    #merged['dino_bbox'] = merged['dino_bbox'].apply(lambda x: eval(x))
    #merged['bbox'] = merged['bbox'].apply(lambda x: eval(x))
    
    print('Done')
    #Visualization
    #test = pd.read_parquet('dino_seg_bbox_merge_wloss.parquet')
    #test = test[(test['tl_loss'] > 1000) & (test['br_loss'] > 1000)]
    #test = test[(test['tl_loss'] <20 ) & (test['br_loss'] <20)]
    test = pd.read_csv('~/roisul/csvs/merged_dino_swints.csv')
    test['swints_bbox'] = test['swints_bbox'].apply(lambda x: eval(x))
    test['dino_bbox'] = test['dino_bbox'].apply(lambda x: eval(x))
    print(test.shape)
    #quit()
    for i, row in test.head().iterrows():
        print(i)
        #cdn_url = row['Filename']
        #fname = cdn_url.split('/')[-1]
        #dino_bbox = row['dino_bbox']
        #seg_bbox = row['swints_bbox']
        top_l = l2_loss_top_left(row)
        bot_r = l2_loss_bottom_right(row)
        print(f"top left loss {top_l} bot right loss {bot_r}")
        #draw_bounding_boxes(cdn_url, (seg_bbox,dino_bbox), f'seg_dino_bbox_comparison_same/{fname}')

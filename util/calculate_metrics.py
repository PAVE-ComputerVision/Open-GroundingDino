import numpy as np
import pandas as pd

def preprocess(data):
    for col in data.columns:
        if ((col == 'Unnamed: 0') or (col == "fname")):
            continue
        else:
            print(col)
            data[col] = data[col].apply(lambda x: eval(x))
    data['num_dmgs'] = data['gt_labels'].apply(lambda x: len(x))
    return data

def get_tp(row, iou_thresh, dist_thresh):
    tp = 0
    metrics_per_pred = row['metrics_per_pred']
    for pred_id, gt_values in metrics_per_pred.items():
        iou = gt_values[0]
        iou_id = gt_values[1]
        dist = gt_values[2]
        dist_id = gt_values[3]

        if ((iou >= iou_thresh) or (dist <= dist_thresh)):
            tp += 1
    return tp

def get_fp(row, iou_thresh, dist_thresh):
    fp = 0
    metrics_per_pred = row['metrics_per_pred']
    for pred_id, gt_values in metrics_per_pred.items():
        iou = gt_values[0]
        iou_id = gt_values[1]
        dist = gt_values[2]
        dist_id = gt_values[3]

        if ((iou < iou_thresh) and (dist > dist_thresh)):
            fp += 1
    return fp

def get_fn(row, iou_thresh, dist_thresh):
    fp = 0
    metrics_per_gt = row['metrics_per_gt']
    for gt_id, pred_values in metrics_per_gt.items():
        iou = pred_values[0]
        iou_id = pred_values[1]
        dist = pred_values[2]
        dist_id = pred_values[3]

        if ((iou < iou_thresh) and (dist > dist_thresh)):
            fp += 1
    return fp

def get_acc(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    #accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy

def get_spec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    specificity = tn/ (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def get_prec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    return precision

def get_rec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    return recall

if __name__ == "__main__":
    #load data
    path1 = '/home/ubuntu/roisul/Open-GroundingDino/test_results/training_2_multiclass_dmgtype2_100k_contd_checkpoint0004.csv'
    path2 = '/home/ubuntu/roisul/Open-GroundingDino/test_results/training_2_multiclass_dmgtype2_50k_checkpoint0004.csv'
    path3 = '/home/ubuntu/roisul/Open-GroundingDino/test_results/training_2_multiclass_dmgtype2_100k_contd_checkpoint0007.csv'
    for path in [path1,path2, path3]:
        data = pd.read_csv(path)
        data = preprocess(data)
        iou_thresh=0.5
        dist_thresh=400
        
        #print(data['num_dmgs'].value_counts())
        #data = data[data['num_dmgs'] <= 15]
        data['tp'] = data.apply(get_tp,iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['fp'] = data.apply(get_fp,iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['tn'] = 10
        data['fn'] = data.apply(get_fn,iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['accuracy'] = data.apply(get_acc, axis=1)
        data['specificity'] = data.apply(get_spec, axis=1)
        data['precision'] = data.apply(get_prec, axis=1)
        data['recall'] = data.apply(get_rec, axis=1)

        print('Accuracy', data['accuracy'].mean())
        print('Specificity', data['specificity'].mean())
        print('Precision', data['precision'].mean())
        print('Recall', data['recall'].mean())
    import ipdb;ipdb.set_trace()

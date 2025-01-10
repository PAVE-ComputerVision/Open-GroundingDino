
#Add code to get bbox preds from outputs
box_threshold = 0.3
pred_lst = []
gt_lst = []
loss_lst = []
for i in range(len(outputs["pred_logits"])):
    logits = outputs["pred_logits"].sigmoid()[i]  # (nq, 256)
    boxes = outputs["pred_boxes"][i]  # (nq, 4)

    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    #if len(boxes_filt) == len(final_targets[i]["boxes"]):
    #    for j in range(len(boxes_filt)):
    #        scale_pred = xywh_to_xyxy(boxes_filt[j], 512, 512)
    #        scale_gt = xywh_to_xyxy(final_targets[i]["boxes"][j], 512,512)
    
    scale_gt = xywh_to_xyxy(final_targets[i]["boxes"][0], 512,512)
    if len(boxes_filt) == 0:
        scale_pred = []
        avg_loss = -1
    else:
        scale_pred = xywh_to_xyxy(boxes_filt[0], 512, 512)
        tl_loss, br_loss = l2_loss_corners(scale_pred, scale_gt)
        avg_loss = (tl_loss+br_loss/2)
    pred_lst.append(scale_pred)
    gt_lst.append(scale_gt)
    loss_lst.append(avg_loss)

    print(pred_lst)
    print(gt_lst)
    print(loss_lst)
    
    #scale_gt = xywh_to_xyxy(final_targets[i]["boxes"][0], 512,512)
    #avg_loss = -1
    #pred_lst.append(scale_pred)
    #gt_lst.append(scale_gt)
    #loss_lst.append(avg_loss)



opt = {}
opt['fname'] = all_names
opt['crop_preds'] = all_preds
opt['crop_gts'] = all_gts
opt['avg_loss'] = all_loss


import pandas as pd
res = pd.DataFrame(opt)
pth = osp.join(args.output_dir, 'result.csv')
res.to_csv(pth)


total_loss = 0; cnt = 0
for crop in res['avg_loss']:
    for loss in crop:
        if loss >= 0:
            total_loss += loss
            cnt += 1
final_loss = total_loss/cnt
loss_track.append(final_loss)
sve_pth = os.path.join(args.output_dir, 'bboxloss_track.pt')
torch.save(loss_track, sve_pth)

updated_loss_track = torch.load(sve_pth)
updated_loss_track.sort()

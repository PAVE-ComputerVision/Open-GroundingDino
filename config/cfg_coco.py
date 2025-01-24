data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None
batch_size = 1
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = 'relu'
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 91
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
max_labels = 80                               # pos + neg
lr = 0.000001                                   # base learning rate
backbone_freeze_keywords = None               # only for gdino backbone
freeze_keywords = None                        # for whole model, e.g. ['backbone.0', 'bert'] for freeze visual encoder and text encoder
lr_backbone = 1e-05                           # specific learning rate
lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_mult = 1e-05
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']
weight_decay = 0.0001
param_dict_type = 'ddetr_in_mmdet'
ddetr_lr_param = False
epochs = 8
lr_drop = 10
save_checkpoint_interval = 10
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [10, 20]
frozen_weights = None
dilation = False
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
mask_loss_coef = 1.0
dice_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0
decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False
dn_scalar = 100

use_coco_eval = False
#label_list = ['small','medium','large']
label_list = ['dent','scratch','missing','scraped','broken','others']
#label_list = ['medium']
#label_list = ['BADGES_MISSING', 'BENT_MAJOR', 'BENT_MEDIUM', 'BROKEN_MAJOR', 'BROKEN_MEDIUM', 'BURNS_MAJOR', 'BURN_HOLE_MEDIUM', 'COLLISION_DAMAGE_MAJOR', 'CRACKED_MAJOR', 'CRACKED_MEDIUM', 'CRACKED_PAINT_MAJOR', 'CRACKED_PAINT_MEDIUM', 'CURB_RASH_MAJOR', 'CURB_RASH_MEDIUM', 'DENTED_MAJOR_NOT_THROUGH_PAINT', 'DENTED_MAJOR_THROUGH_PAINT', 'DENTED_MEDIUM_NOT_THROUGH_PAINT', 'DENTED_MEDIUM_THROUGH_PAINT', 'DENTED_MINOR_NOT_THROUGH_PAINT', 'DENT_MAJOR_SEAM_LINE', 'DENT_MEDIUM_SEAM_LINE', 'GOUGED_MEDIUM', 'HAIL_DAMAGE_MEDIUM', 'HAIL_DAMAGE_MINOR', 'HEAVY_DAMAGE_MAJOR', 'HOLE_MAJOR', 'HOLE_MEDIUM', 'HOLE_MINOR', 'LOOSE_MAJOR', 'MISALIGNED_MAJOR', 'MISMATCHED_MAJOR', 'MISMATCHED_WHEEL_MAJOR', 'MISSING_CENTER_CAP_MAJOR', 'MISSING_MAJOR', 'MISSING_WHEEL_COVER_MAJOR', 'MULTIPLE_DENTS_MAJOR_THROUGH_PAINT', 'MULTIPLE_DENTS_MEDIUM_NOT_THROUGH_PAINT', 'MULTIPLE_SCRATCHES_MAJOR_THROUGH_PAINT', 'MULTIPLE_SCRATCHES_MEDIUM_THROUGH_PAINT', 'MULTIPLE_SCRATCHES_MINOR_NOT_THROUGH_PAINT', 'OXIDIZED_MAJOR', 'PEELING_MAJOR', 'PEELING_MEDIUM', 'PIECE_MISSING', 'PREVIOUS_REPAIR_MAJOR', 'RUSTED_MAJOR', 'RUSTED_MEDIUM', 'SCRAPED_MAJOR', 'SCRATCH_MAJOR_NOT_THROUGH_PAINT', 'SCRATCH_MAJOR_THROUGH_PAINT', 'SEVERE_DAMAGE_MAJOR', 'TAPED_REPAIR', 'TAPED_REPAIR_MAJOR', 'WRONG_COLOR_MAJOR']


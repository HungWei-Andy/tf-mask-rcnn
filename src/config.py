import numpy as np

class Config(object):
    pass

_config = Config()
_config.DEBUG = False#True
_config.num_classes = 80
_config.image_size = 224
_config.min_size = 2
_config.rpn_nms_thresh = 0.7
_config.crop_size = 7
_config.mask_crop_size = 14
_config.crop_channel = 256
_config.proposal_count_infer = 1000 # 300 for C4, 1000 for FPN
_config.batch_size = 3
_config.rpn_bbox_stddev = np.array([0.1, 0.1, 0.2, 0.2])
_config.delta_loc = 10
_config.rois_per_img = 512 # 64 for C4, 512 for FPN
_config.rois_fg_ratio = 0.25
_config.rois_fg_thresh = 0.5
_config.rois_bg_thresh_low = 0.5
_config.rois_bg_thresh_hegh = 0.1
_config.rpn_positive_iou = 0.7
_config.rpn_negative_iou = 0.3
#_config.rpn_pos_ratio = #
#_config.rpn_batch_size = #

cfg = _config

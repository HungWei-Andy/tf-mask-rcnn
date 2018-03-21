import numpy as np

class Config(object):
    pass

_config = Config()
_config.lr = 0.001
_config.momentum = 0.9
_config.weight_decay = 0.0001

_config.use_fpn = False

if _config.use_fpn:
  _config.proposal_count_infer = 1000
  _config.rois_per_img = 512
  _config.mask_crop_size = 14
  _config.crop_channel = 256
else:
  _config.proposal_count_infer = 300
  _config.rois_per_img = 64
  _config.mask_crop_size = 7
  _config.crop_channel = 1024

_config.summary_dir = 'logs/summary'
_config.DEBUG = False#True
_config.eps = 1e-8
_config.log_eps = 1e-12
_config.image_mean = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)
_config.num_classes = 80+1
_config.image_size = 448
_config.min_size = 2
_config.rpn_nms_thresh = 0.7
_config.crop_size = 7
_config.batch_size = 2 # if batch_size > 1, remember to resize properly when loading image
_config.bbox_mean = np.array([0.0, 0.0, 0.0, 0.0])
_config.bbox_stddev = np.array([0.1, 0.1, 0.2, 0.2])
_config.delta_loc = 10
_config.rois_fg_ratio = 0.25
_config.fg_per_img = int(_config.rois_per_img*_config.rois_fg_ratio)
_config.rois_fg_thresh = 0.5
_config.rois_bg_thresh_low = -0.1
_config.rois_bg_thresh_high = 0.5
_config.rpn_positive_iou = 0.7
_config.rpn_negative_iou = 0.3
_config.print_every = 2
_config.save_every = 10000
_config.iterations = 70000
_config.rpn_pos_ratio = 0.5
_config.rpn_batch_size = 256

cfg = _config

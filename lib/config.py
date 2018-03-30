import numpy as np

class Config(object):
    pass

_config = Config()
_config.optimizer = 'adam'
_config.lr = 0.0001
_config.momentum = 0.9
_config.weight_decay = 0.0001
_config.decay_step = 1000000
_config.decay_rate = 0.1
_config.output_dir = 'model/output/mobilenet050_adam'
_config.model_name = 'model.ckpt'
_config.eval_iou_thresh = 0.5

_config.cls_only = False

_config.network = 'mobilenet050' # resnet50/mobilenet050
if _config.network == 'resnet50':
  _config.pretrained_model = 'model/pretrained_model/ori_resnet/resnet50.npy'
  _config.conv1_label = 'conv1'
elif _config.network == 'mobilenet050':
  _config.pretrained_model = 'model/pretrained_model/mobilenet_v1/mobilenet_v1_0.5_224.ckpt'
  _config.conv1_label = 'MobilenetV1/Conv2d_0'

_config.use_fpn = True
if _config.use_fpn:
  _config.proposal_count_infer = 1000
  _config.rois_per_img = 512
  _config.mask_crop_size = 14
else:
  _config.proposal_count_infer = 300
  _config.rois_per_img = 64
  _config.mask_crop_size = 7

_config.rpn_only = False#True
if _config.rpn_only:
  _config.batch_size = 20
else:
  _config.batch_size = 2

_config.summary_dir = 'logs/summary'
_config.DEBUG = False#True
_config.eps = 1e-10
_config.log_eps = 1e-10
_config.image_mean = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)
_config.num_classes = 80+1
_config.image_size = 448
_config.min_size = 2
_config.rpn_nms_thresh = 0.7
_config.crop_size = 7
_config.bbox_mean = np.array([0.0, 0.0, 0.0, 0.0])
_config.bbox_stddev = np.array([1.0, 1.0, 1.0, 1.0])#np.array([0.1, 0.1, 0.2, 0.2])
_config.delta_loc = 1
_config.rois_fg_ratio = 0.5
_config.fg_per_img = int(_config.rois_per_img*_config.rois_fg_ratio)
_config.rois_fg_thresh = 0.45
_config.rois_bg_thresh_low = 0
_config.rois_bg_thresh_high = 0.5
_config.rpn_positive_iou = 0.7
_config.rpn_negative_iou = 0.3
_config.print_every = 10
_config.save_every = 2500
_config.iterations = 200000
_config.rpn_pos_ratio = 0.5
_config.rpn_batch_size = 256

cfg = _config

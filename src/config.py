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
_config.proposal_count_train = 2000
_config.proposal_count_infer = 1000
_config.batch_size = 3
_config.rpn_bbox_stddev = np.array([0.1, 0.1, 0.2, 0.2])
cfg = _config

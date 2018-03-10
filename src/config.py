import numpy as np

class Config(object):
    pass

_config = Config()
_config.DEBUG = False#True
_config.image_size = 224
_config.min_size = 2
_config.rpn_nms_thresh = 0.7
_config.crop_size = 7
_config.proposal_count_train = 2000
_config.proposal_count_infer = 1000
_config.batch_size = 3
_config.rpn_bbox_stddev = np.array([0.1, 0.1, 0.2, 0.2])
cfg = _config

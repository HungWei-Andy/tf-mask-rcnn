
class Config(object):
    pass

_config = Config()
_config.image_size = 224
_config.min_size = 2
_config.rpn_nms_thresh = 0.7
_config.crop_size = 7
_config.proposal_count_train = 2000
_config.proposal_count_infer = 1000
cfg = _config

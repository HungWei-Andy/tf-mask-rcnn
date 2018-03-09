import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from rpn import decode_roi, refine_rois, crop_proposal
# TODO: argscope for detailed setting in fpn and rpn

def resnet50(X):
    _, layers = resnet_v1.resnet_v1_50(X)
    return layers

def create_resnet50_feat(X):
    all_layers = resnet50(X)
    # 1. feats, 2. ratios
    pass

def fpn(layers):
    num_layers = len(layers)
    outputs = []
    outputs.append(slim.conv2d(layers[-1], 256, 1))
    
    for curi in range(num_layers-2, -1, -1):
        cur_feat = slim.conv2d(layers[i], 256, 1)
        upsamp_feat = tf.image.resize_bilinear(outputs[0], cur_feat.get_shape)
        outputs = [cur_feat+upsamp_feat] + outputs
    return outputs

def mask_rcnn(X, network_feat_fn):
    feats, shrink_ratios = network_feat_fn(X)
    rpn_feats, crop_feats = fpn(feats)
    anchors, loc, cls = rpn_logits(rpn_feats, shrink_ratios)
    rois = decode_roi(anchors, loc, cls)
    rois = refine_rois(rois)
    region_feats = crop_feat_by_roi(crop_feats, rois)

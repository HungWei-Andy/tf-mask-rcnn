import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from rpn import rpn_logits, decode_rois, refine_rois, crop_proposals
from config import cfg
# TODO: argscope for detailed setting in fpn and rpn

def resnet50(X, training):
    _, layers = resnet_v1.resnet_v1_50(X, is_training=training)
    output_layers = [
        layers['resnet_v1_50/block1'],
        layers['resnet_v1_50/block2'],
        layers['resnet_v1_50/block3'],
        layers['resnet_v1_50/block4']
    ]
    shrink_ratios = [3, 4, 5, 5]

    ############### DEBUG ###############
    if cfg.DEBUG:
        for layer in output_layers:
            print('resnet layer shape: {}', layer.get_shape().as_list())
        for key in layers.keys():
            var = layers[key]
            print(key, var, var.shape)
    #####################################
    return output_layers, shrink_ratios

def fpn(layers, ratios):
    crop_channel = cfg.crop_channel
    num_layers = len(layers)
    outputs = []

    # pyramid
    outputs.append(slim.conv2d(layers[-1], crop_channel, 1)) 
    for curi in range(num_layers-2, -1, -1):
        cur_feat = slim.conv2d(layers[curi], crop_channel, 1)
        ups_feat = tf.image.resize_images(outputs[0], cur_feat.get_shape()[1:3])
        outputs = [cur_feat+ups_feat] + outputs

    # add P6 for RPN prediction
    outputs.append(slim.max_pool2d(layers[-1], 1, 2))
    ratios.append(ratios[-1]+1)

    return outputs, outputs[:-1], ratios

def mask_rcnn(X, network_feat_fn, training):
    feats, shrink_ratios = network_feat_fn(X, training)
    rpn_feats, crop_feats, shrink_ratios = fpn(feats, shrink_ratios)
    anchors, loc, cls = rpn_logits(rpn_feats, shrink_ratios)
    rois = decode_rois(anchors, loc, cls)
    proposals = refine_rois(rois, training)
    region_feats = crop_proposals(crop_feats, proposals, training)
    return region_feats 

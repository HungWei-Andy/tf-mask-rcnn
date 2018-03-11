import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.resnet import ResNet50
from rpn import rpn_logits, decode_rois, refine_rois, crop_proposals, classifier
from config import cfg
# TODO: argscope for detailed setting in fpn and rpn

def resnet50(X, training):
    net = ResNet50(istrain=training)
    y = net(X)
    output_layers = [
        net.conv2,
        net.conv3,
        net.conv4,
        net.conv5
    ]
    shrink_ratios = [2, 3, 4, 5]

    for layer in output_layers:
        print(layer.get_shape().as_list(), layer.name)

    ############### DEBUG ###############
    if cfg.DEBUG:
        for var in tf.global_variables():
            print(var.name, var.get_shape().as_list())
        #for layer in output_layers:
        #    print('resnet layer shape: {}', layer.get_shape().as_list())
        #for key in layers.keys():
        #    var = layers[key]
        #    print(key, var, var.shape)
    #####################################
    return output_layers, shrink_ratios, net

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
    feats, shrink_ratios, net = network_feat_fn(X, training)
    rpn_feats, crop_feats, shrink_ratios = fpn(feats, shrink_ratios)
    anchors, loc, cls = rpn_logits(rpn_feats, shrink_ratios)
    rois = decode_rois(anchors, loc, cls)
    proposals = refine_rois(rois, training)
    region_feats = crop_proposals(crop_feats, proposals, training)
    class_logits, class_probs, reg_logits = classifer(region_feats, training)
    return region_feats 

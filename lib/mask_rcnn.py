import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from targets import rpn_targets, classifier_targets
from rpn import rpn_logits, decode_roi, refine_rois, crop_proposals
from loss import compute_rpn_loss, compute_cls_loss
from cnn import classifier, mask_classifier, mixture_conv_bn_relu
from config import cfg
from nets.feat_extractors import feat_extractor_maker

def fpn(layers, ratios):
    crop_channel = layers[-1].shape[-1]
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

def mask_rcnn(X, training, network, gt_boxes=None, gt_classes=None, gt_masks=None):
    '''
    X: NHWC tensor
    gt_boxes: N length of list (num_boxes, 4) coordinates for each image
    gt_classes: N length of list (num_boxes,) label for each image
    gt_masks: N length of list (num_boxes, H, W) binary mask for each iamge
    '''
    feat_extractor = feat_extractor_maker(network)
    feats, shrink_ratios = feat_extractor(X, training)
    if cfg.use_fpn:
        rpn_feats, crop_feats, shrink_ratios = fpn(feats, shrink_ratios)
    else:
        rpn_feats, crop_feats, shrink_ratios = [feats[-1]], [feats[-1]], [shrink_ratios[-1]]

    with tf.variable_scope('RPN') as scope:    
        anchors, rpn_loc, rpn_cls = rpn_logits(rpn_feats, shrink_ratios)

    rois = decode_roi(anchors, rpn_loc, rpn_cls)
    if training:
        rpn_gt_labels, rpn_gt_terms = rpn_targets(anchors, gt_boxes)
        proposals, cls_gt_labels, cls_gt_terms, cls_gt_masks = classifier_targets(
                                                rois['box'], gt_boxes, gt_classes, gt_masks)
    else:
        proposals = refine_rois(rois, training)

    with tf.variable_scope('CLS'):
        if cfg.use_fpn:
            cls_feats = crop_proposals(crop_feats, cfg.crop_size, proposals, training)
            mask_feats = crop_proposals(crop_feats, cfg.mask_crop_size, proposals, training)
        else:
            feat = crop_proposals(crop_feats, cfg.crop_size, proposals, training)
            feat = mixture_conv_bn_relu(feat, 2048, 1, training)
            cls_feats = mask_feats = feat
        class_logits, class_probs, bbox_logits = classifier(cls_feats, training)
        mask_logits = mask_classifier(mask_feats, training)
 
    # create loss
    if training:
        loss = {}
        compute_rpn_loss(rpn_cls, rpn_loc, rpn_gt_labels, rpn_gt_terms, loss)
        compute_cls_loss(class_logits, bbox_logits, mask_logits, cls_gt_labels,
                         cls_gt_terms, cls_gt_masks, loss)
        loss['all'] = loss['rpn'] + loss['classifier']
        return loss, feat_extractor
    return proposals, class_probs, bbox_logits, mask_logits


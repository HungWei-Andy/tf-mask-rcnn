import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.resnet import ResNet50
from rpn_train import rpn_targets, classifier_targets
from rpn import rpn_logits, decode_roi, refine_rois, crop_proposals
from loss import compute_rpn_loss, compute_cls_loss
from cnn import classifier, mask_classifier
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

    ############### DEBUG ###############
    if cfg.DEBUG:
        pass
        #for var in tf.global_variables():
        #    print(var.name, var.get_shape().as_list())
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

def mask_rcnn(X, training, network_feat_fn=None, gt_boxes=None, gt_classes=None, gt_masks=None):
    '''
    X: NHWC tensor
    gt_boxes: N length of list (num_boxes, 4) coordinates for each image
    gt_classes: N length of list (num_boxes,) label for each image
    gt_masks: N length of list (num_boxes, H, W) binary mask for each iamge
    '''
    if network_feat_fn is None:
      network_feat_fn = resnet50
    feats, shrink_ratios, net = network_feat_fn(X, training)
    rpn_feats, crop_feats, shrink_ratios = fpn(feats, shrink_ratios)
    anchors, rpn_loc, rpn_cls = rpn_logits(rpn_feats, shrink_ratios)
 
    rois = decode_roi(anchors, rpn_loc, rpn_cls, X)
    if training:
        rpn_gt_labels, rpn_gt_terms = rpn_targets(anchors, gt_boxes)
        proposals, cls_gt_labels, cls_gt_terms, cls_gt_masks = classifier_targets(
                                                rois['box'], gt_boxes, gt_classes, gt_masks)
        #return proposals, net
    else:
        proposals = refine_rois(rois)

    cls_feats = crop_proposals(crop_feats, cfg.crop_size, proposals, training)
    mask_feats = crop_proposals(crop_feats, cfg.mask_crop_size, proposals, training)
    class_logits, class_probs, bbox_logits = classifier(cls_feats, training)
    mask_logits = mask_classifier(mask_feats, training)

    # create loss
    if training:
        loss = {}
        #return {'rpn_cls': rpn_cls, 'rpn_loc': rpn_loc, 'rpn_gt_labels': rpn_gt_labels,
        #        'rpn_gt_terms': rpn_gt_terms, 'class_logits': class_logits,
        #        'bbox_logits': bbox_logits, 'mask_logits': mask_logits,
        #        'cls_gt_labels': cls_gt_labels, 'cls_gt_terms': cls_gt_terms,
        #        'cls_gt_masks': cls_gt_masks, 'mask_feats': mask_feats, 'gt_masks': gt_masks}, net
        compute_rpn_loss(rpn_cls, rpn_loc, rpn_gt_labels, rpn_gt_terms, cfg.delta_loc, loss)
        compute_cls_loss(class_logits, bbox_logits, mask_logits, cls_gt_labels,
                         cls_gt_terms, cls_gt_masks, loss)
        loss['all'] = loss['rpn'] + loss['classifier']
        return loss, net
    return class_logits, class_probs, bbox_logits, mask_logits


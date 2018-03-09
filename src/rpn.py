import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from nms import nms
from config import cfg
# TODO: argscope for detailed setting in fpn and rpn

def create_anchors(feats, stride, scales, aspect_ratios=[0.5, 1, 2], base_size=16):
    inp_shape = feats.get_shape().as_list()
    height, width = inp_shape[1], inp_shape[2]
    num_ratios = len(aspect_ratios)
    num_scales = len(scales)
    ctr = 0.5*(base_size-1)

    aspr = np.array(aspect_ratios)
    fixed_area = base_size**2
    ratio_wh = np.zeros((num_ratios, 2))
    ratio_wh[:,0] = np.round(np.sqrt(fixed_area/aspr))
    ratio_wh[:,1] = np.round(np.sqrt(fixed_area*aspr))

    scs = np.array(scales).reshape(-1, 1, 1)
    scale_wh = scs * ratio_wh[np.newaxis, :, :]
    scale_wh = scale_wh.reshape(-1, 2)
    
    base_anchors = np.hstack((ctr-0.5*(scale_wh-1), ctr+0.5*(scale_wh-1)))
    
    anchors = np.zeros((width, height, num_ratios*num_scales, 4))
    anchors[:, :, :, [0,2]] += stride * np.arange(width).reshape(-1,1,1,1)
    anchors[:, :, :, [1,3]] += stride * np.arange(height).reshape(1,-1,1,1)
    anchors += base_anchors.reshape(1,1,-1,4)
    anchors = tf.convert_to_tensor(anchors)
    return anchors

def rpn_logits(feats, ratios):
    out_anchors = []
    out_loc = []
    out_cls = []
    N_batch_tensor = tf.shape(feats)[0]
    for i, feat in enumerate(feats):
        ratio = ratios[i]
        
        # create anchors
        anchors = create_anchors(feat, 2**ratio, [2**(ratio-2), 2**(ratio-1), 2**ratio])
        num_anchors = anchors.get_shape().as_list()[-2]
        
        # predict cls, coordinate
        conv_feat = slim.conv2d(feats, 512, 3)
        loc = slim.conv2d(conv_feat, num_anchors*4, 1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                    activation_fn=tf.nn.sigmoid)
        cls = slim.conv2d(conv_feat, num_anchors*2, 1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
        cls = tf.nn.softmax(tf.reshape(cls, (-1, 2)))

        # reshape into size(N, -1)
        out_anchors.append(tf.reshape(anchors, (-1, 4))) # shape: [H*W*N_anchor, 4]
        out_loc.append(tf.reshape(loc, (N_batch_tensor, -1, 4)) # shape: [N, H*W*num_anchor, 4]
        out_cls.append(tf.reshape(cls, (N_batch_tensor, -1, 2)) # shape: [N, H*W*num_anchor, 2]
    return out_anchors, out_loc, out_cls

def decode_roi(anchors, loc, cls):
    '''
    Inputs
      - anchors: anchor boxes, a tensor of shape [H*W*N_anchor, 4]
      - loc: the location rpn logits, a tensor of shape [N, H*W*N_anchor, 4]
      - cls: the class rpn logits, a tensor of shape [N, H*W*N_anchor, 2]
    Ouputs
      - boxes: the bbox coordinates (xmin, ymin, xmax, ymax),
               a tensor of shape [N, H*W*N_anchor, 4], the
               decoded [xmin, ymin, xmax, ymax] for each box
      - probs: probability of object, a tensor of shape [N, H*W*N_anchor]
    '''
    imsize = cfg.image_size
    anchors = anchors[np.newaxis, :, :]
    anc_widths = anchors[:,:,2] - anchors[:,:, 0] + 1
    anc_heights = anchors[:,:,3] - anchors[:,:,1] + 1
    anc_ctrx = anchors[:,:,0] + 0.5 * anc_widths
    anc_ctry = anchors[:,:,0] + 0.5 * anc_heights
    
    box_ctrx = boxvar[:,:,0] * 0.1 * anc_widths + anc_ctrx
    box_ctry = boxvar[:,:,1] * 0.1 * anc_heights + anc_ctry
    box_w = anc_widths * np.exp(boxvar[:,:,2] * 0.2)
    box_h = anc_heights * np.exp(boxvar[:,:,3] * 0.2)
    
    boxes = np.zeros(boxvar.shape, dtype=boxvar.dtype)
    boxes[:,:,0] = box_ctrx - 0.5 * box_w
    boxes[:,:,1] = box_ctry - 0.5 * box_h
    boxes[:,:,2] = box_ctrx + 0.5 * box_w - 1
    boxes[:,:,3] = box_ctry + 0.5 * box_h - 1
    boxes = np.maximum(0.0, np.minimum(imsize, boxes))
    
    probs = cls[:,:,1]
    return boxes, probs

def decode_rois(anchors, locs, clses):
    list_probs, list_boxes = [], []
    for i in range(len(anchors)):
        anchor, loc, cls = anchors[i], locs[i], clses[i]
        boxes, probs = tf.py_func(decode_roi, [anchor, loc, cls], [tf.float32, tf.float32, tf.float32])
        list_probs.append(probs)
        list_boxes.append(boxes)

    rois = {}
    roi['prob'] = tf.concat(list_probs, axis=1)
    roi['box'] = tf.concat(list_boxes, axis=1)
    return roi

def refine_roi(boxes, probs, pre_nms_topn, post_nms_topn):
    # filter too small boxes
    widths = normalized_box[:,2] - normalized_box[:,0] 
    heights = normalized_box[:,3] - normalized_box[:,1]
    keep = np.where((widths >= cfg.min_size) & (heights >= cfg.min_size))
    boxes = boxes[keep, :]
    probs = probs[keep]

    # filter with scores
    order = probs.ravel().argsort()[::-1]
    if pre_nms_top_n > 0:
        order = order[:pre_nms_top_n]
    boxes = boxes[order, :]
    probs = probs[order]
    return boxes, probs

def refine_rois(rois, training):
    min_size = cfg.min_size
    nms_thresh = cfg.rpn_nms_thresh
    proposal_count = cfg.proposal_count_train if training else cfg.proposal_count_test
    pre_nms_topn = 12000
    post_nms_topn = 2000
    if not training:
        pre_nms_topn = 6000
        post_nms_topn = 400

    boxes, probs = rois['box'], rois['prob']
    boxes = boxes * np.array(cfg.rpn_bbox_stddev).reshape(1, 1, 4)
    
    N = boxes.shape[0]
    roi_batch = []
    for i in range(N):
        box, prob = tf.gather_nd(boxes, tf.constant(i)), tf.gather_nd(boxes, tf.constant(i))
        box, prob = tf.reshape(box, [-1, 4]), tf.reshape(prob, [-1])
        nonms_box, nonms_probs = tf.py_func(refine_roi, [box, prob, pre_nms_topn, post_nms_topn], [tf.float32, tf.float32])
        normalized_box = nonms_box / cfg.image_size
        indices = tf.images.non_max_suppression(nonmalized_box, nonms_probs, proposal_count, nms_thresh)
        proposals = tf.gather(nonms_box, indices)
        roi_batch.append(proposals)
    return roi_batch

def roi_crop(feats, rois):
    boxes = rois['box'] # shape: None, num_boxes, 4
    probs = rois['prob'] # shape: None, 
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=2)
    w = x2 - x1
    h = y2 - y1
    
    # adaptive features in fpn
    ks = tf.log(tf.sqrt(w*h)/cfg.image_size) / tf.log(tf.constant(2.0))
    ks = 4 + tf.cast(tf.round(k), tf.int32)
    ks = tf.minimum(5, tf.maximum(2, level))
    
    # crop and resize
    out = []
    original_ind = []
    for i, curk in enumerate(range(2, 6)):
        filtered_idx = tf.where(tf.equal(ks, curk))
        original_ind.append(filtered_idx)

        cur_boxes = tf.gather_nd(boxes, filtered_idx)
        batch_idx = tf.cast(filtered_idx[:, 0], tf.int32)
        cur_boxes = tf.stop_gradient(cur_boxes)
        batch_idx = tf.stop_gradient(batch_ind)
        out.append(tf.image.crop_and_resize(feats[i], cur_boxes, batch_idx, cfg.crop_size))

    # encapsulate
    out = tf.concat(out, axis=0)
    original_ind = tf.concat(original_ind, axis=0)

    # rearrange

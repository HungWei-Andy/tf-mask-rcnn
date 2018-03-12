import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
# TODO: argscope for detailed setting in fpn and rpn

def create_anchors(feats, stride, scales, aspect_ratios=[0.5, 1, 2], base_size=16):
    inp_shape = feats.get_shape()
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
    
    anchors = ( stride * tf.reshape(tf.range(width), [-1,1,1,1])
              + stride * tf.reshape(tf.range(height), [1,-1,1,1])
              + base_anchors.reshape(1,1,-1,4) )
    anchors = tf.cast(anchors, tf.float32)
    return anchors

def rpn_logits(feats, ratios):
    out_anchors = []
    out_loc = []
    out_cls = []
    N_batch_tensor = tf.shape(feats[0])[0]
    for i, feat in enumerate(feats):
        ratio = ratios[i]
        
        # create anchors
        anchors = create_anchors(feat, 2**ratio, [2**(ratio-2), 2**(ratio-1), 2**ratio])
        num_anchors = anchors.get_shape().as_list()[-2]
        
        # predict cls, coordinate
        conv_feat = slim.conv2d(feat, 512, 3)
        loc = slim.conv2d(conv_feat, num_anchors*4, 1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                    activation_fn=tf.nn.sigmoid)
        cls = slim.conv2d(conv_feat, num_anchors*2, 1, activation_fn = None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
        cls = tf.nn.softmax(tf.reshape(cls, (-1, 2)))

        # reshape into size(N, -1)
        out_anchors.append(tf.reshape(anchors, (-1, 4))) # shape: [H*W*N_anchor, 4]
        out_loc.append(tf.reshape(loc, (N_batch_tensor, -1, 4))) # shape: [N, H*W*num_anchor, 4]
        out_cls.append(tf.reshape(cls, (N_batch_tensor, -1, 2))) # shape: [N, H*W*num_anchor, 2]
    out_anchors = tf.concat(out_anchors, axis=0)
    out_loc = tf.concat(out_loc, axis=1)
    out_cls = tf.concat(out_cls, axis=1)
    print('anchor shape', out_anchors.get_shape().as_list())
    print('loc shape', out_loc.get_shape().as_list())
    print('cls shape', out_cls.get_shape().as_list())
    return out_anchors, out_loc, out_cls


def decode_roi(anchors, loc, cls, img_tensor):
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
    H, W = img_tensor.get_shape()[1:3]
    H, W = tf.cast(H, tf.float32), tf.cast(W, tf.float32)
    anchors = tf.expand_dims(anchors, axis=1)
    
    anc_widths = anchors[:,:,2] - anchors[:,:, 0] + 1
    anc_heights = anchors[:,:,3] - anchors[:,:,1] + 1
    anc_ctrx = anchors[:,:,0] + 0.5 * anc_widths
    anc_ctry = anchors[:,:,1] + 0.5 * anc_heights
    
    box_ctrx = loc[:,:,0] * 0.1 * anc_widths + anc_ctrx
    box_ctry = loc[:,:,1] * 0.1 * anc_heights + anc_ctry
    box_w = anc_widths * tf.exp(loc[:,:,2] * 0.2)
    box_h = anc_heights * tf.exp(loc[:,:,3] * 0.2)
    
    box_minx = tf.maximum(0.0, box_ctrx - 0.5 * box_w)
    box_miny = tf.maximum(0.0, box_ctry - 0.5 * box_h)
    box_maxx = tf.minimum(W, box_ctrx + 0.5 * box_w - 1)
    box_maxy = tf.minimum(H, box_ctry + 0.5 * box_h - 1)
    boxes = tf.stack([box_minx, box_miny, box_maxx, box_maxy], axis=2)
    
    probs = cls[:,:,1]

    rois = {'anchor': anchors, 'box': boxes, 'prob': probs}
    return rois

def refine_roi(boxes, probs, pre_nms_topn, post_nms_topn):
    image_size = cfg.image_size
    min_size = cfg.min_size

    # filter with scores
    _, order = tf.nn.top_k(probs, pre_nms_topn)
    boxes = tf.gather(boxes, order)
    probs = tf.gather(probs, order)

    # filter too small boxes
    normalized_box = boxes / image_size
    widths = normalized_box[:,2] - normalized_box[:,0] 
    heights = normalized_box[:,3] - normalized_box[:,1]
    keep = tf.logical_and(widths >= min_size, heights >= min_size)
    boxes = tf.boolean_mask(boxes, keep)
    probs = tf.boolean_mask(probs, keep)

    return boxes, probs

def refine_rois(rois, training):
    image_size = cfg.image_size
    min_size = cfg.min_size
    nms_thresh = cfg.rpn_nms_thresh
    proposal_count = cfg.proposal_count_train if training else cfg.proposal_count_infer
    batch_size = cfg.batch_size if training else 1
    box_stddev = cfg.rpn_bbox_stddev

    pre_nms_topn = 12000
    post_nms_topn = 2000
    if not training:
        pre_nms_topn = 6000
        post_nms_topn = 400

    boxes, probs = rois['box'], rois['prob']
    boxes = boxes * box_stddev.reshape(1, 1, 4)
    
    N = boxes.shape[0]
    roi_batch = []
    for i in range(batch_size):
        box, prob = boxes[i], probs[i]
        box, prob = tf.reshape(box, [-1, 4]), tf.reshape(prob, [-1])
        nonms_box, nonms_probs = refine_roi(box, prob, pre_nms_topn, post_nms_topn)

        normalized_box = nonms_box / image_size
        indices = tf.image.non_max_suppression(normalized_box, nonms_probs, proposal_count, nms_thresh)
        proposals = tf.gather(nonms_box, indices)
        padding = proposal_count-tf.shape(proposals)[0]
        proposals = tf.reshape(tf.pad(proposals, [[0, padding], [0,0]]), [proposal_count, 4])
        roi_batch.append(proposals)
    final_proposal = tf.stack(roi_batch, axis=0)
    return final_proposal

def crop_proposals(feats, crop_size, boxes, training):
    crop_channel = cfg.crop_channel
    image_size = cfg.image_size
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=2)
    x1, y1, x2, y2 = x1[:,:,0], y1[:,:,0], x2[:,:,0], y2[:,:,0]
    w = x2 - x1
    h = y2 - y1

    # adaptive features in fpn
    ks = tf.log(tf.sqrt(w*h)/image_size) / tf.log(tf.constant(2.0))
    ks = 4 + tf.cast(tf.round(ks), tf.int32)
    ks = tf.minimum(5, tf.maximum(2, ks))

    # crop and resize
    outputs = []
    original_ind = []
    for i, curk in enumerate(range(2, 6)):
        filtered_ind = tf.where(tf.equal(ks, curk))
        cur_boxes = tf.gather_nd(boxes, filtered_ind)
        batch_ind = tf.cast(filtered_ind[:, 0], tf.int32)
        original_ind.append(batch_ind)

        cur_boxes = tf.stop_gradient(cur_boxes)
        batch_ind = tf.stop_gradient(batch_ind)
       
        out = tf.image.crop_and_resize(feats[i], cur_boxes, batch_ind, [crop_size, crop_size])
        outputs.append(out)

    # encapsulate
    out = tf.concat(out, axis=0)
    original_ind = tf.concat(original_ind, axis=0)

    # re-arrange
    num_total_box = tf.shape(original_ind)[0]
    ind_total_box = tf.range(num_total_box)
    sort_ind = original_ind * num_total_box + ind_total_box
    ind = tf.nn.top_k(sort_ind, k=num_total_box).indices[::-1]
    output = tf.gather(out, ind)
    output = tf.reshape(output, [-1, crop_size, crop_size, crop_channel])
    
    return output


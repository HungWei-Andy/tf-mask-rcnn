import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
# TODO: argscope for detailed setting in fpn and rpn

def create_anchors(feats, stride, scales, aspect_ratios=[0.5, 1, 2], base_size=8):
    feat_size = cfg.image_size / stride
    num_ratios = len(aspect_ratios)
    num_scales = len(scales)
    ctr = 0.5*base_size

    aspr = np.array(aspect_ratios)
    fixed_area = base_size**2
    ratio_wh = np.zeros((num_ratios, 2))
    ratio_wh[:,0] = np.round(np.sqrt(fixed_area/aspr))
    ratio_wh[:,1] = np.round(np.sqrt(fixed_area*aspr))

    scs = np.array(scales).reshape(-1, 1, 1)
    scale_wh = scs * ratio_wh[np.newaxis, :, :]
    scale_wh = scale_wh.reshape(-1, 2)
    
    base_anchors = np.hstack((ctr-0.5*scale_wh, ctr+0.5*scale_wh))
    
    anchors = np.zeros((feat_size, feat_size, num_ratios*num_scales, 4), dtype=np.float32)
    anchors += base_anchors.reshape(1,1,-1,4)
    anchors[:,:,:,[0,2]] += np.arange(feat_size).reshape(-1,1,1,1) * stride
    anchors[:,:,:,[1,3]] += np.arange(feat_size).reshape(-1,1,1,1) * stride
    anchors = np.minimum(cfg.image_size-1, np.maximum(0.0, anchors))
    print('anchors\n', anchors.reshape(-1,4))
    print('base anchors\n', base_anchors)
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    return anchors

def rpn_logits(feats, ratios):
    out_anchors = []
    out_loc = []
    out_cls = []
    for i, feat in enumerate(feats):
        ratio = ratios[i]
        
        # create anchors
        anchors = create_anchors(feat, 2**ratio, [2**(ratio-2), 2**(ratio-1), 2**ratio])
        num_anchors = anchors.get_shape().as_list()[-2]
        
        # predict cls, coordinate
        initializer = tf.truncated_normal_initializer(stddev=0.001)
        conv_feat = slim.conv2d(feat, 512, 3,
                    weights_initializer=initializer)
        loc = slim.conv2d(conv_feat, num_anchors*4, 1, activation_fn = None,
                    weights_initializer=initializer)
        cls = slim.conv2d(conv_feat, num_anchors*2, 1, activation_fn = None,
                    weights_initializer=initializer)

        # reshape into size(N, -18)
        out_anchors.append(tf.reshape(anchors, (-1, 4))) # shape: [H*W*N_anchor, 4]
        out_loc.append(tf.reshape(loc, (cfg.batch_size, -1, 4))) # shape: [N, H*W*num_anchor, 4]
        out_cls.append(tf.reshape(cls, (cfg.batch_size, -1, 2))) # shape: [N, H*W*num_anchor]
    out_anchors = tf.concat(out_anchors, axis=0)
    out_loc = tf.concat(out_loc, axis=1)
    out_cls = tf.concat(out_cls, axis=1)
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
    H, W = 1.*cfg.image_size, 1.*cfg.image_size
    anchors = tf.expand_dims(anchors, 0)

    anc_widths = anchors[:,:,2] - anchors[:,:, 0]
    anc_heights = anchors[:,:,3] - anchors[:,:,1]
    anc_ctrx = anchors[:,:,0] + 0.5 * anc_widths
    anc_ctry = anchors[:,:,1] + 0.5 * anc_heights

    loc = loc * cfg.bbox_stddev.reshape(1,1,4) + cfg.bbox_mean.reshape(1,1,4)
    box_ctrx = loc[:,:,0] * (anc_widths+cfg.eps) + anc_ctrx
    box_ctry = loc[:,:,1] * (anc_heights+cfg.eps) + anc_ctry
    box_w = (anc_widths+cfg.eps) * (tf.exp(loc[:,:,2])-cfg.log_eps)
    box_h = (anc_heights+cfg.eps) * (tf.exp(loc[:,:,3])-cfg.log_eps)

    box_minx = box_ctrx - 0.5 * box_w
    box_miny = box_ctry - 0.5 * box_h
    box_maxx = box_ctrx + 0.5 * box_w
    box_maxy = box_ctry + 0.5 * box_h
    boxes = tf.stack([box_minx, box_miny, box_maxx, box_maxy], axis=2)

    probs = tf.nn.softmax(cls)
    probs = probs[:,:,1]

    rois = {'anchor': anchors, 'box': boxes, 'prob': probs}
    return rois

def refine_roi(boxes, probs, pre_nms_topn):
    image_size = cfg.image_size
    min_size = cfg.min_size

    # filter with scores
    _, order = tf.nn.top_k(probs, tf.minimum(pre_nms_topn, tf.size(probs)))
    boxes = tf.gather(boxes, order)
    probs = tf.gather(probs, order)

    # filter too small boxes
    widths = boxes[:,2] - boxes[:,0] 
    heights = boxes[:,3] - boxes[:,1]
    keep = tf.logical_and(widths >= min_size, heights >= min_size)
    boxes = tf.boolean_mask(boxes, keep)
    probs = tf.boolean_mask(probs, keep)
    return boxes, probs

def refine_rois(rois, training):
    image_size = cfg.image_size
    min_size = cfg.min_size
    nms_thresh = cfg.rpn_nms_thresh
    proposal_count = cfg.proposal_count_infer
    batch_size = 1
    box_mean = cfg.bbox_mean.reshape(1, 1, 4)
    box_stddev = cfg.bbox_stddev.reshape(1, 1, 4)

    pre_nms_topn = 12000
    if not training:
        pre_nms_topn = 6000

    boxes, probs = rois['box'], rois['prob']
    boxes = boxes * box_stddev + box_mean
    
    N = boxes.shape[0]
    roi_batch = []
    for i in range(batch_size):
        box, prob = boxes[i], probs[i]
        box, prob = tf.reshape(box, [-1, 4]), tf.reshape(prob, [-1])
        nonms_box, nonms_probs = refine_roi(box, prob, pre_nms_topn)

        indices = tf.image.non_max_suppression(nonms_box, nonms_probs, proposal_count, nms_thresh)

        proposals = tf.gather(nonms_box, indices)
        padding = proposal_count-tf.shape(proposals)[0]
        proposals = tf.reshape(tf.pad(proposals, [[0, padding], [0,0]]), [proposal_count, 4])
        roi_batch.append(proposals)
    final_proposal = tf.stack(roi_batch, axis=0)
    return final_proposal

def crop_proposals(feats, crop_size, boxes, training):
    crop_channel = feats[0].shape[-1]
    image_size = cfg.image_size
    proposal_count = cfg.rois_per_img if training else cfg.proposal_count_infer
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=2)
    x1, y1, x2, y2 = x1[:,:,0], y1[:,:,0], x2[:,:,0], y2[:,:,0]
    w = x2 - x1
    h = y2 - y1

    if not cfg.use_fpn:
        output = tf.image.crop_and_resize(feats[0], tf.reshape(boxes, (-1,4)), 
                                          tf.range(cfg.batch_size*proposal_count)//proposal_count,
                                          [crop_size, crop_size])
    else:
        # adaptive features in fpn
        ks = tf.log(tf.sqrt(w*h)/(image_size+cfg.eps)+cfg.log_eps) / tf.log(tf.constant(2.0))
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
       
            out = tf.image.crop_and_resize(feats[i], cur_boxes/cfg.image_size, batch_ind, [crop_size, crop_size])
            out = tf.stop_gradient(out)
            outputs.append(out)

        # encapsulate
        out = tf.concat(outputs, axis=0)
        original_ind = tf.concat(original_ind, axis=0)

        # re-arrange
        num_total_box = tf.shape(original_ind)[0]
        ind_total_box = tf.range(num_total_box)
        sort_ind = original_ind * num_total_box + ind_total_box
        ind = tf.nn.top_k(sort_ind, k=num_total_box).indices[::-1]
        output = tf.gather(out, ind)
    output = tf.reshape(output, [-1, crop_size, crop_size, crop_channel])
    return output


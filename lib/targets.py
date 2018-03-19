import tensorflow as tf
import numpy as np
from config import cfg

def compute_area(xmin, xmax, ymin, ymax):
    return ((xmax>xmin)*(xmax-xmin)*(ymax>ymin)*(ymax-ymin)).astype(np.float32)

def bbox_overlaps(boxes, query):
    '''
   boxes: (N, 4) array
    query: (M, 4) array
    RETURN: (N, M) array where ai,j is the distance matrix
    '''
    bxmin, bxmax = np.reshape(boxes[:,0], [-1,1]), np.reshape(boxes[:,2], [-1,1])
    bymin, bymax = np.reshape(boxes[:,1], [-1,1]), np.reshape(boxes[:,3], [-1,1])
    qxmin, qxmax = np.reshape(query[:,0], [1,-1]), np.reshape(query[:,2], [1,-1])
    qymin, qymax = np.reshape(query[:,1], [1,-1]), np.reshape(query[:,3], [1,-1]) 

    area_boxes = compute_area(bxmin, bxmax, bymin, bymax)
    area_query = compute_area(qxmin, qxmax, qymin, qymax)
    union = area_boxes + area_query
    
    ixmin, ixmax = np.maximum(bxmin, qxmin), np.minimum(bxmax, qxmax)
    iymin, iymax = np.maximum(bymin, qymin), np.minimum(bymax, qymax)
    intersection = compute_area(ixmin, ixmax, iymin, iymax)

    overlap = intersection / (union + cfg.eps)
    return overlap

def minmax2ctrwh(boxes):
    widths = np.maximum(0.0, boxes[:,2] - boxes[:,0] + 1)
    heights = np.maximum(0.0, boxes[:,3] - boxes[:,1] + 1)
    ctrx = boxes[:,0] + widths * 0.5
    ctry = boxes[:,1] + heights * 0.5
    return widths, heights, ctrx, ctry

def encode_roi(anchors, boxes):
    '''
    - anchors: (N, 4) tensors
    - boxes: (N, 4) tensors
    RETURN
    - terms: (N, 4) encoded terms
    '''
    anc_w, anc_h, anc_ctrx, anc_ctry = minmax2ctrwh(anchors)
    box_w, box_h, box_ctrx, box_ctry = minmax2ctrwh(boxes)
    tx = (box_ctrx - anc_ctrx) / (anc_w + cfg.eps)
    ty = (box_ctry - anc_ctry) / (anc_h + cfg.eps)
    tw = np.log(box_w / (anc_w + cfg.eps))
    th = np.log(box_h / (anc_h + cfg.eps))
    return np.stack((tx, ty, tw, th), axis=1)

def rpn_target_one_batch(anchors, gt_boxes):
    '''
    Propose rpn_targt for one batch
    - anchors: (N, 4) array
    - gt_boxes: (M, 4) groundtruths boxes
    RETURN
    p labels: (N,), 1 for positive, 0 for negative, -1 for don't care
    - terms: (N, 4), regression terms for each positive anchors
    '''
    N, M = anchors.shape[0], gt_boxes.shape[0]
    iou = bbox_overlaps(gt_boxes, anchors)
    max_iou =  iou.max(axis=1).reshape(-1, 1)
    max_iou_ind = np.where(max_iou == iou)[1]

    max_gt_ind = iou.argmax(axis=0)
    max_gt_iou = iou[max_gt_ind, range(N)]

    # decide labels
    labels = np.zeros(N, np.int32)-1
    labels[max_gt_iou < cfg.rpn_negative_iou] = 0 # iou < negative_thresh
    labels[max_gt_iou > cfg.rpn_positive_iou] = 1 # iou > postive_thresh
    labels[max_iou_ind] = 1 # maximum iou with each groundtruth

    # filter out too many positive or negative
    pos_inds = np.where(labels == 1)[0]
    neg_inds = np.where(labels == 0)[0]
    num_pos = int(cfg.rpn_pos_ratio * cfg.rpn_batch_size)
    num_neg = cfg.rpn_batch_size - num_pos
    if len(pos_inds) > num_pos:
        disabled_ind = np.random.choice(pos_inds, size=len(pos_inds)-num_pos, replace=False)
        labels[disabled_ind] = -1
    if len(neg_inds) > num_neg:
        disabled_ind = np.random.choice(neg_inds, size=len(neg_inds)-num_neg, replace=False)
        labels[disabled_ind] = -1
    
    # decide regression terms 
    terms = np.zeros((N,4), np.float32)-1
    pos_ind = np.where(labels == 1)[0]
    terms[pos_ind] = encode_roi(anchors[pos_ind], gt_boxes[max_gt_ind[pos_ind]])
    return labels, terms, max_iou_ind.shape

def rpn_targets(anchors, gt_boxes):
    '''
    Return the labels and for all anchors w.r.t each image
    - anchors: (N,4) tensor, the 
    - gt_boxes: a list of (K,2) array, K is the number of gt for each image
    RETURN
    - out_labels: (M,N) target labels tensor for M image, N anchors
    - out_terms: (M,N,4) encoded target terms tensor for M image, N anchors
    '''
    out_labels, out_terms = [], []
    for gt in gt_boxes:
        labels, terms, label_part = tf.py_func(
            rpn_target_one_batch, [anchors, gt], [tf.int32, tf.float32, tf.int64])
        labels = tf.Print(labels, [label_part])
        out_labels.append(labels)
        out_terms.append(terms)
    return tf.stack(out_labels, axis=0), tf.stack(out_terms, axis=0)

def classifier_target_one_batch(rois, gt_boxes, gt_classes, gt_masks):
    '''
    Choose foreground and background sample proposals
    - rois: (N,4) roi bboxes
    - gt_boxes: (M,4) groundtruth boxes
    - gt_classes: (M,) class label for each box
    - gt_masks: (M,H,W) binary label for each groundtruth instance
    RETURN
    - sampled_rois: (rois_per_img, 4) sampled rois
    - labels: (rois_per_img,) class labels for each foreground, 0 for background
    - loc: (fg_per_img, 4) encoded regression targets for each foreground, pad for bg
    - mask: (fg_per_img, mask_output_size, mask_output_size, num_classes) class mask
    '''
    num_rois = cfg.rois_per_img
    num_fg = cfg.fg_per_img
    num_bg = num_rois - num_fg
    
    iou = bbox_overlaps(rois, gt_boxes)
    max_iou_ind = iou.argmax(axis=1)
    max_iou = iou[range(iou.shape[0]), max_iou_ind]
    fg_inds = np.array(np.where(max_iou>cfg.rois_fg_thresh)).reshape(-1)
    bg_inds = np.array(np.where((max_iou>=cfg.rois_bg_thresh_low)&(max_iou<=cfg.rois_bg_thresh_high))).reshape(-1)
    
    if fg_inds.size > 0 and bg_inds.size > 0:
        num_fg = min(num_fg, fg_inds.size)
        fg_inds = np.random.choice(fg_inds, size=num_fg, replace=False)
        num_bg = num_rois - num_fg
        bg_inds = np.random.choice(bg_inds, size=num_bg, replace=num_bg>bg_inds.size)
    elif fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=num_rois, replace=num_rois>fg_inds.size)
        num_fg, num_bg = num_rois, 0
    elif bg_inds.size > 0:    
        bg_inds = np.random.choice(bg_inds, size=num_rois, replace=num_rois>bg_inds.size)
        num_fg, num_bg = 0, num_rois
      
    # rois
    sampled_rois = rois[np.append(fg_inds, bg_inds), :]
    # labels
    fg_gt_inds = max_iou_ind[fg_inds]
    labels = np.append(gt_classes[fg_gt_inds], np.zeros(num_bg)).astype(np.int32)
    # box
    loc = np.zeros((num_fg, 4), np.float32)
    all_masks = np.zeros((num_rois, gt_masks.shape[1], gt_masks.shape[2], cfg.num_classes), np.float32)
    intersection = np.zeros((num_fg, 4), np.float32)
    if num_fg > 0:
      box_pred = sampled_rois[:num_fg, :].reshape(-1,4)
      box_gt = gt_boxes[fg_gt_inds, :].reshape(-1,4)
      loc = encode_roi(box_pred, box_gt)
      # intersection
      intersection = np.hstack((np.maximum(box_pred[:,:2], box_gt[:,:2]),
                              np.minimum(box_pred[:,2:], box_gt[:,2:])))
      for i, fg_ind in enumerate(fg_gt_inds):
        all_masks[i, :, :, gt_classes[fg_ind]] = gt_masks[i, :, :]  

    return sampled_rois, labels, loc, intersection, all_masks, np.arange(num_fg, dtype=np.int32)

def classifier_targets(cand_rois, gt_boxes, gt_classes, gt_masks):
    '''
    Return the esampled rois, their class, mask, encoded regression terms.
    - rois: (batch_size, proposal_count, 4) bounding boxes
    - gt_boxes: list of len(batch_size) where gt_boxes[i] is a (N,4) tensor for each gt
    - gt_classes: list of len(batch_size) where gt_classes[i] is a (N,) tensor for the
                  class label of each groundtruth box
    - gt_masks: list of len(batch_size) where gt_masks[i] is a (N,H,W) tensor for the
                mask of each groundtruth instance
    RETURN
    - rois_ind: (N, proposal_batch_size, 4), sampled proposal bbox
    - cls: (N, proposal_batch_size), class label of each sampled proposal
    - loc: (N, proposal_batch_size, 4), regression terms of each sampled proposal
    - 
    '''
    batch_size = cfg.batch_size
    rois, cls, loc, mask = list(), list(), list(), list()
    for i in range(batch_size):
        gt = gt_boxes[i]
        roi = cand_rois[i,:,:]
        gt_cls = gt_classes[i]
        gt_mask = gt_masks[i]
        
        sampled_rois, sampled_cls, sampled_loc, inter, all_masks, num_fg = tf.py_func(
            classifier_target_one_batch, [roi, gt, gt_cls, gt_mask],
            [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32])
        sampled_mask = tf.image.crop_and_resize(all_masks, inter/cfg.image_size,
                                                num_fg,
                                                [cfg.mask_crop_size*2, cfg.mask_crop_size*2])
        rois.append(sampled_rois)
        cls.append(sampled_cls)
        loc.append(sampled_loc) 
        mask.append(sampled_mask)
    rois, cls, loc, mask = tf.stack(rois,0), tf.stack(cls,0), tf.stack(loc,0), tf.stack(mask,0)
    return rois, cls, loc, mask

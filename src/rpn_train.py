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
    intersection = compute_aerea(ixmin, ixmax, iymin, iymax)

    overlap = intersection / union
    return overlap

def minmax2ctrwh(boxes):
    widths = boxes[:,2] - anchors[:,0] + 1
    heights = boxes[:,3] - anchors[:,1] + 1
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
    tx = 10.0 * (box_ctrx - anc_ctrx) / anc_w
    ty = 10.0 * (box_ctry - anc_ctry) / anc_h
    tw = 5.0 * np.log(box_w / anc_w)
    th = 5.0 * np.log(box_h / anc_h)
    return np.stack((tx, ty, tw, th), axis=1)

def rpn_target_one_batch(anchors, gt_boxes):
    '''
    Propose rpn_targt for one batch
    - anchors: (N, 4) array
    - gt_boxes: (M, 4) groundtruths boxes
    RETURN
    - labels: (N,), 1 for positive, 0 for negative, -1 for don't care
    - terms: (N, 4), regression terms for each positive anchors
    '''
    N, M = anchors.shape[0], gt_boxes.shape[0]
    iou = bbox_overlaps(gt_boxes, anchors)
    max_anchor_ind = iou.argmax(axis=1)
    max_gt_ind = iou.argmax(axis=0)
    max_gt_iou = iou[max_gt_ind, range(N)]

    # decide labels
    labels = np.zeros(N)-1
    labels[max_anchor_ind] = 1 # maximum iou with each groundtruth
    labels[max_gt_iou > cfg.rpn_positive_iou] = 1 # iou > postive_thresh
    labels[max_gt_iou < cfg.rpn_negative_iou[ = 0 # iou < negative_thresh

    # filter out too many positive or negative
    pos_inds = np.where(labels == 1)[0]
    neg_inds = np.where(labels == 0)[0]
    num_pos = int(cfg.rpn_pos_fraction * cfg.rpn_batchsize)
    num_neg = cfg.rpn_batchsize - num_pos
    if len(pos_inds) > num_pos:
        disabled_ind = np.random.choice(pos_inds, size=num_pos-len(pos_inds), replace=False)
        labels[disabled_ind] = -1
    if len(neg_inds) > num_neg:
        disabled_ind = np.random.choice(neg_inds, size=num_neg-len(neg_inds), replace=False)
        labels[disabled_ind] = -1
    
    # decide regression terms 
    terms = np.zeros((N,4), np.float32)-1
    pos_ind = np.where(labels == 1)[0]
    terms[pos_ind] = encode_roi(anchors[pos_ind], gt_boxes[max_gt_ind[pos_ind]])
    return labels, terms

def rpn_targets(anchors, gt_boxes):
    out_labels, out_terms = [], []
    for gt in gt_boxes:
        labels, terms = tf.py_func(rpn_target_one_batch, [anchors, gt], [tf.int32, tf.float32])
        out_labels.append(labels)
        out_terms.append(terms)
    return tf.stack(out_labels, axis=0), tf.stack(out_terms, axis=0)

def classifier_targets():
    pass

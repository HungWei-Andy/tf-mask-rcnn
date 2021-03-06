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

    ixmin, ixmax = np.maximum(bxmin, qxmin), np.minimum(bxmax, qxmax)
    iymin, iymax = np.maximum(bymin, qymin), np.minimum(bymax, qymax)
    intersection = compute_area(ixmin, ixmax, iymin, iymax)

    area_boxes = compute_area(bxmin, bxmax, bymin, bymax)
    area_query = compute_area(qxmin, qxmax, qymin, qymax)
    union = area_boxes + area_query - intersection
    
    overlap = intersection / (union + cfg.eps)
    return overlap

def minmax2ctrwh(boxes):
    widths = np.maximum(0.0, boxes[:,2] - boxes[:,0])
    heights = np.maximum(0.0, boxes[:,3] - boxes[:,1])
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
    tw = np.log(box_w / (anc_w + cfg.eps) + cfg.log_eps)
    th = np.log(box_h / (anc_h + cfg.eps) + cfg.log_eps)
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
    max_iou_ind =  iou.argmax(axis=1)
    max_iou = iou[range(M), max_iou_ind]

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
    terms = (terms - cfg.bbox_mean.reshape(1,4)) / cfg.bbox_stddev.reshape(1,4)
    terms = terms.astype(np.float32)
    #return labels, terms, (np.where(labels==1)[0]).size, (np.where(labels==0)[0]).size, max_gt_iou, max_iou
    return labels, terms

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
        #labels, terms, num_pos_rpn, num_neg_rpn, gt_iou, box_iou = tf.py_func(
        #    rpn_target_one_batch, [anchors, gt], [tf.int32, tf.float32, tf.int64, tf.int64, tf.float32, tf.float32])

        labels, terms = tf.py_func(rpn_target_one_batch, [anchors, gt], [tf.int32, tf.float32])
        #################################### DEBUG ##########################################
        # num_pos += num_pos_rpn                                                            #
        # num_neg += num_neg_rpn                                                            #
        # terms = tf.Print(terms, [tf.convert_to_tensor('max iou'), tf.reduce_max(gt_iou)]) #
        # terms = tf.Print(terms, [tf.convert_to_tensor('iou'), gt_iou])                    #
        # terms = tf.Print(terms, [tf.convert_to_tensor('# gt'), tf.size(box_iou)])         #
        # terms = tf.Print(terms, [tf.convert_to_tensor('pos num'), num_pos_rpn])           #
        # terms = tf.Print(terms, [tf.convert_to_tensor('neg num'), num_neg_rpn])           #
        #####################################################################################
        out_labels.append(labels)
        out_terms.append(terms)
    out_labels, out_terms = tf.stack(out_labels, axis=0), tf.stack(out_terms, axis=0)
    out_labels, out_terms = tf.stop_gradient(out_labels), tf.stop_gradient(out_terms)
    return out_labels, out_terms

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

    fg_inds = np.where(max_iou>cfg.rois_fg_thresh)[0]
    bg_inds = np.where((max_iou>=cfg.rois_bg_thresh_low)&(max_iou<=cfg.rois_bg_thresh_high))[0]
    fg_inds_ori = fg_inds
    bg_inds_ori = bg_inds

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
    intersection = np.zeros((num_fg, 4), np.float32)
    if num_fg > 0:
      box_pred = sampled_rois[:num_fg, :]
      box_gt = gt_boxes[fg_gt_inds, :]
      loc = encode_roi(box_pred, box_gt)
      # intersection
      intersection = np.hstack((np.maximum(box_pred[:,:2], box_gt[:,:2]),
                              np.minimum(box_pred[:,2:], box_gt[:,2:])))
    fg_gt_inds = fg_gt_inds.astype(np.int32)
    return (sampled_rois, labels, loc, intersection, fg_gt_inds, np.array(num_bg).astype(np.int32), 
           bg_inds_ori.astype(np.int32), fg_inds_ori.astype(np.int32), iou.max())

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
    - rois: (N, proposal_batch_size, 4), sampled proposal bbox
    - cls: (N, proposal_batch_size), class label of each sampled proposal
    - loc: (N, proposal_batch_size, 4), regression terms of each sampled proposal
    - mask: (N, proposal_batch_size, mask_size, mask_size, num_classes), gt masks 
    '''
    batch_size = cfg.batch_size
    cand_rois = tf.maximum(0.0, tf.minimum(cfg.image_size-1., cand_rois))
    rois, cls, loc, mask = list(), list(), list(), list()
    for i in range(batch_size):
        gt = gt_boxes[i]
        roi = cand_rois[i,:,:]
        gt_cls = gt_classes[i]
        gt_mask = gt_masks[i]
        
        sampled_rois, sampled_cls, sampled_loc, inter, fg_gt_inds, num_bg, bg_inds, fg_inds, iou = tf.py_func(
            classifier_target_one_batch, [roi, gt, gt_cls, gt_mask],
            [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32])

        ########################################## DEBUG ############################################
        # all_masks = tf.Print(all_masks, [tf.convert_to_tensor('second stage targets completed')]) #
        fg_gt_inds = tf.Print(fg_gt_inds, [tf.shape(gt), tf.size(sampled_cls), tf.size(fg_gt_inds), num_bg, tf.size(bg_inds), tf.size(fg_inds), tf.shape(roi), iou])
        #############################################################################################
        
        sampled_mask = tf.image.crop_and_resize(gt_mask, inter/cfg.image_size,
                                                fg_gt_inds,
                                                [cfg.mask_crop_size*2, cfg.mask_crop_size*2])
        
        # padding
        num_fg = tf.size(fg_gt_inds)
        num_bg = cfg.rois_per_img - num_fg
        sampled_loc = tf.pad(sampled_loc, [[0,num_bg], [0,0]])
        sampled_mask = tf.pad(sampled_mask, [[0,num_bg], [0,0], [0,0], [0,0]])
        ############################################# DEBUG ############################################   
        # sampled_mask = tf.Print(sampled_mask, [tf.convert_to_tensor('mask ground truth completed')]) #
        ################################################################################################

        rois.append(sampled_rois)
        cls.append(sampled_cls)
        loc.append(sampled_loc) 
        mask.append(sampled_mask)
    rois = tf.stack(rois, 0)
    cls = tf.stack(cls, 0)
    loc = tf.stack(loc, 0)
    mask = tf.stack(mask, 0)
    rois = (rois - cfg.bbox_mean.reshape(1,1,4)) / cfg.bbox_stddev.reshape(1,1,4)
    rois, cls = tf.stop_gradient(rois), tf.stop_gradient(cls)
    loc, mask = tf.stop_gradient(loc), tf.stop_gradient(mask)
    return rois, cls, loc, mask

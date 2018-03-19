import tensorflow as tf
from config import cfg

def smooth_l1_loss(dist):
    dist = tf.reshape(dist, [-1])
    dist = tf.abs(dist)
    inside_ind = tf.where(dist<1)
    outside_ind = tf.where(dist>=1)
    loss_inside = 0.5*tf.reduce_sum(tf.square(tf.gather(dist, inside_ind)))
    loss_outside = tf.reduce_sum(dist - 0.5)
    return loss_inside + loss_outside

def compute_rpn_loss(cls, loc, gt_cls, gt_loc, delta_loc, loss):
    '''
    compute total rpn loss
    - cls: (batch_size, total_anchors, 2) tensor, the class probability
    - loc: (batch_size, total_anchors, 4) tensor, the encoded reg terms
    - gt_cls: (batch_size, total_anchors) groundtruth label tensor,
              1 for positive, 0 for negative, -1 for don't care
    - gt_loc: (batch_size, total_anchors, 4) groundtruth reg terms
    return
    - loss: a scalar tensor for rpn loss
    '''
    batch_size, total_anchors = cls.get_shape().as_list()[:2]
    
    valid_ind = tf.where(tf.greater(gt_cls, -1))
    pos_ind = tf.where(tf.equal(gt_cls, 1))

    cls = tf.gather_nd(cls, valid_ind)
    gt_cls = tf.gather_nd(gt_cls, valid_ind)
    loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=cls)
    loss_cls = tf.reduce_mean(loss_cls) * total_anchors
    loss['rpn_cls'] = loss_cls

    loc = tf.gather(loc, pos_ind)
    gt_loc = tf.gather(gt_loc, pos_ind)
    loss_loc = smooth_l1_loss(loc - gt_loc)
    loss_loc = loss_loc * batch_size
    loss['rpn_loc'] = (loss_loc, loc, gt_loc)

    loss['rpn'] = loss_cls + loss_loc * delta_loc

def compute_cls_loss(cls, loc, mask, gt_cls, gt_loc, gt_mask, loss):
    num_classes = cfg.num_classes
    crop_size = cfg.mask_crop_size
    mask_pn = 4*crop_size**2

    cls, gt_cls = tf.reshape(cls, (-1,num_classes)), tf.reshape(gt_cls, [-1])
    loc, gt_loc = tf.reshape(loc, (-1,4)), tf.reshape(gt_loc, (-1,4))
    mask, gt_mask = tf.reshape(mask, (-1,mask_pn,num_classes)), tf.reshape(gt_mask, (-1,mask_pn,num_classes))

    valid_ind = tf.where(gt_cls > 0)
    loc, maak = tf.gather(loc, valid_ind), tf.gather(mask, valid_ind)
    
    loss['cls'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=cls))
    loss['loc'] = smooth_l1_loss(loc - gt_loc)
    loss['mask'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_mask, logits=mask))
    loss['classifier'] = loss['cls'] + loss['loc'] + loss['mask']

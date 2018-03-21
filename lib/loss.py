import tensorflow as tf
from config import cfg


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
    valid_ind = tf.stop_gradient(tf.where(tf.greater(gt_cls, -1)))
    pos_ind = tf.stop_gradient(tf.where(tf.greater(gt_cls, 0)))

    cls = tf.gather_nd(cls, valid_ind)
    gt_cls = tf.gather_nd(gt_cls, valid_ind)
    loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=cls)
    loss_cls = tf.reduce_mean(loss_cls)
    loss['rpn_cls'] = loss_cls

    loc = tf.gather_nd(loc, pos_ind)
    gt_loc = tf.gather_nd(gt_loc, pos_ind)
    loss_loc = tf.reduce_mean(tf.losses.huber_loss(gt_loc, loc)) * 4 # product feature dim
    loss['rpn_loc'] = loss_loc

    loss['rpn'] = loss_cls + loss_loc# * delta_loc

def compute_cls_loss(cls, loc, mask, gt_cls, gt_loc, gt_mask, loss):
    num_classes = cfg.num_classes
    crop_size = cfg.mask_crop_size
    mask_pn = 4*crop_size**2

    fg_ind = tf.stop_gradient(tf.where(gt_cls > 0))
    loc = tf.gather_nd(loc, fg_ind)
    mask = tf.gather_nd(mask, fg_ind)

    loc, gt_loc = tf.reshape(loc, (-1,4)), tf.reshape(gt_loc, (-1,4))
    mask, gt_mask = tf.reshape(mask, (-1,num_classes)), tf.reshape(gt_mask, (-1,num_classes))
     
    loss['cls'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=cls))
    loss['loc'] = tf.reduce_mean(tf.losses.huber_loss(loc, gt_loc)) * 4
    loss['mask'] = tf.reduce_mean(-gt_mask * tf.log(mask+cfg.log_eps))
    loss['classifier'] = loss['cls'] + loss['loc'] + loss['mask']

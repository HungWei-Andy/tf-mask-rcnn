import tensorflow as tf

def smooth_l1_loss(dist):
    dist = tf.reshape(dist, [-1])
    dist = tf.abs(dist)
    inside_ind = tf.where(dist<1)
    outside_ind = tf.where(dist>=1)
    loss_inside = 0.5*tf.reduce_sum(tf.square(tf.gather(dist, inside_ind)))
    loss_outside = tf.reduce_sum(dist - 0.5)
    return loss_inside + loss_outside

def compute_rpn_loss(cls, loc, gt_cls, gt_loc, delta_loc=10):
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
    batch_size, total_anchors = cls.get_shape()[:2]
    
    cls = tf.reshape(cls, (-1,2))
    loc = tf.reshape(loc, (-1,4))
    gt_cls = tf.reshape(gt_clos, (-1,2))
    gt_loc = tf.reshape(gt_loc, (-1,4))

    valid_ind = tf.where(gt_cls >= 0)
    pos_ind = tf.where(gt_cls == 1)

    cls = tf.gather(cls, valid_ind)
    gt_cls = tf.gather(gt_cls, valid_ind)
    loss_cls = tf.nn.sparse_softamx_cross_entropy_with_logits(labels=gt_cls, logits=cls)
    loss_cls = loss_cls * total_anchors

    loc = tf.gather(loc, pos_ind)
    gt_loc = tf.gather(gt_loc, pos_ind)
    loss_loc = smooth_l1_loss(loc - gt_loc)
    loss_loc = loss_loc * batch_size

    loss = loss_cls + loss_loc * delta_loc
    return loss

def compute_cls_loss(cls, loc, mask, gt_cls, gt_loc, gt_mask):
    num_classes = cfg.num_classes
    cls, gt_cls = tf.reshape(cls, (-1,2)), tf.reshape(gt_cls, [-1])
    loc, gt_loc = tf.reshape(loc, (-1,4)), tf.reshape(gt_loc, (-1,4))
    mask, gt_mask = tf.reshape(mask, (-1,num_classes)), tf.reshape(gt_mask, (-1,num_classes))

    loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=cls)
    loss_loc = smooth_l1_loss(loc - gt_loc)
    loss_mask = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_mask, logits=mask)
    loss = loss_cls + loss_loc + loss_mask
    return loss

import tensorflow as tf
from config import cfg

def mixture_conv_bn_relu(X, outChannel, kernel, training):
    feat = tf.layers.conv2d(X, outChannel, kernel, padding='same', use_bias=False)
    feat = tf.layers.batch_normalization(feat, training=training)
    feat = tf.nn.relu(feat)
    return feat

def classifier(X, training):
    num_classes = cfg.num_classes
    crop_size = cfg.crop_size
    proposal_count = cfg.proposal_count_train if training else cfg.proposal_count_infer

    # feature mixture
    feat = mixture_conv_bn_relu(X, 1024, crop_size, training)
    feat = mixture_conv_bn_relu(feat, 2048, 1, training)    

    # predict
    class_logits = tf.layers.conv2d(feat, num_classes, 1)
    class_probs = tf.nn.softmax(class_logits)
    bbox_logits = tf.layers.conv2d(feat, num_classes*4, 1, use_bias=False)

    # rshape to [batch, proposal_count, N_cls]
    class_logits = tf.reshape(class_logits, [-1, proposal_count, num_classes])
    class_probs = tf.reshape(class_probs, [-1, proposal_count, num_classes])
    bbox_logits = tf.reshape(bbox_logits, [-1, proposal_count, num_classes, 4])

    return class_logits, class_probs, bbox_logits 

def mask_classifier(X, training):
    num_classes = cfg.num_classes
    crop_size = cfg.mask_crop_size
    proposal_count = cfg.proposal_count_train if training else cf.proposal_count_infer

    feat = mixture_conv_bn_relu(X, 256, 3, training)
    feat = mixture_conv_bn_relu(feat, 256, 3, training)
    feat = mixture_conv_bn_relu(feat, 256, 3, training)
    feat = mixture_conv_bn_relu(feat, 256, 3, training)
    feat = tf.layers.conv2d_transpose(feat, 256, 2, strides=2, padding='same', activation=tf.nn.relu)
    mask = tf.layers.conv2d(feat, num_classes, 1, activation=tf.sigmoid)
    mask = tf.reshape(mask, [-1, proposal_count, crop_size*2, crop_size*2, num_classes])
    return mask


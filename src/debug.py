import tensorflow as tf
from networks import mask_rcnn, resnet50

X = tf.placeholder(tf.float32, [None, 224, 224, 3])
region_feats = mask_rcnn(X, resnet50, True)

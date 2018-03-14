import tensorflow as tf
from mask_rcnn import mask_rcnn, resnet50

X = tf.placeholder(tf.float32, [None, 400, 400, 3])
a, b, c, d = mask_rcnn(X, resnet50, True)
#print(a.get_shape(), b.get_shape(), c.get_shape(), d.get_shape())

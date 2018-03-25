import numpy as np
import tensorflow as tf
from resnet import ResNet50
from mobilenet_v1 import mobilenet_v1_050
from mobilenet

class FeatExtractor(object):
    def __call__(self, X, training):
        raise NotImplementedError
    def load(self, sess, path):
        raise NotImplementedError

class resnet50(FeatExtractor):
    def __call__(self, X, training):
        self.net = ResNet50(istrain=training)
        y = self.net(X)
        output_layers = [
            self.net.conv2,
            self.net.conv3,
            self.net.conv4,
            self.net.conv5
        ]
        shrink_ratios = [2, 3, 4, 5]
        return output_layers, shrink_ratios 

    def load(self, sess, path):
        self.net.load(sess, path)

class mobilenet50(FeatExtractor):
    def __call__(self, X, training):
        net, all_feats = mobilenet_v1_050(X, is_training=training, num_classes=None)
        conv1 = all_feats['Conv2d_1_pointwise']
        conv2 = all_feats['Conv2d_3_pointwise']
        conv3 = all_feats['Conv2d_5_pointwise']
        conv4 = all_feats['Conv2d_11_pointwise']
        conv5 = all_feats['Conv2d_13_depthwise']
        output_layers = [conv2, conv3, conv4, conv5]
        shrink_ratios = [2, 3, 4, 5]
        return output_layers, shrink_ratios

    def load(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path)

def feat_extractor_maker(label):
    if label == 'resnet50':
        return resnet50()
    elif label == 'mobilenet050':
        return mobilenet050()
    else:
        raise NotImplementedError

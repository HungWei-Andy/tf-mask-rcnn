import tensorflow as tf
from nets.network import Network
import numpy as np
# !! The default data format used here is NHWC !!
# TODO: scope

def conv_bn(X, inChannel, outChannel, kernel, istrain, stride=1, name=None):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[kernel, kernel, inChannel, outChannel],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    padding = int((kernel-1)/2)
    out = tf.pad(X, [[0,0], [padding,padding], [padding,padding], [0, 0]])
    out = tf.nn.conv2d(out, weights, [1, stride, stride, 1], 'VALID', name=name)
    out = tf.layers.batch_normalization(out, training=istrain, name=name)
    return out

def conv_bn_relu(X, inChannel, outChannel, kernel, istrain, stride=1, name=None):
    out = conv_bn(X, inChannel, outChannel, kernel, istrain, stride=stride, name=name)
    out = tf.nn.relu(out)
    return out

def fc(X, inChannel, outChannel, name=None):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[inChannel, outChannel],
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases  = tf.get_variable('biases', shape=[outChannel], initializer=tf.zeros_initializer)
    return tf.nn.xw_plus_b(X, weights, biases)

def residual_conv(X, inChannel, interChannel, outputChannel, transition, istrain=False, scope=None):
    with tf.variable_scope(scope):
        if transition:
            init_stride = 2
        else:
            init_stride = 1

        if inChannel == outputChannel:
            skip_out = X
        else:
            skip_out = conv_bn(X, inChannel, outputChannel, 1, istrain, stride=init_stride, name='shortcut')
    
        conv_out = conv_bn_relu(X, inChannel, interChannel, 1, istrain, stride=init_stride, name='a')
        conv_out = conv_bn_relu(conv_out, interChannel, interChannel, 3, istrain, name='b')
        conv_out = conv_bn(conv_out, interChannel, outputChannel, 1, istrain, name='c')
        out = tf.nn.relu(skip_out + conv_out)
    return out

def residual_block(X, inChannel, interChannel, outputChannel, numLayers, transition=True, istrain=False, scope=None):
    with tf.variable_scope(scope):
        out = residual_conv(X, inChannel, interChannel, outputChannel, transition=transition, istrain=istrain, scope='block1')
        for i in range(numLayers - 1):
            out = residual_conv(out, outputChannel, interChannel, outputChannel, transition=False, istrain=istrain, scope='block%d'%(i+2))
    return out

class ResNet(Network):
    def __init__(self, scope='resnet', istrain=False, reuse=False):
        super(ResNet, self).__init__(scope)
        self.reuse = reuse
        self.istrain = istrain

    def _build_resnet(self, numBlock1, numBlock2, numBlock3, numBlock4):
        self.conv1 = conv_bn_relu(self.input, 3, 64, 7, istrain=self.istrain, stride=2, name='scale1')
        self.pool1 = tf.layers.max_pooling2d(self.conv1, 3, 2, padding='same')
        self.conv2 = residual_block(self.pool1, 64, 64, 256, numBlock1, transition=False, istrain=self.istrain, scope='scale2') 
        self.conv3 = residual_block(self.conv2, 256, 128, 512, numBlock2, istrain=self.istrain, scope='scale3')
        self.conv4 = residual_block(self.conv3, 512, 256, 1024, numBlock3, istrain=self.istrain, scope='scale4')
        self.conv5 = residual_block(self.conv4, 1024, 512, 2048, numBlock4, istrain=self.istrain, scope='scale5')
        self.pool5 = tf.layers.average_pooling2d(self.conv5, 7, 1)
        self.pool5_flat = tf.layers.flatten(self.pool5)
        self.scores = fc(self.pool5_flat, 2048, 1000, name='fc')
        return self.scores

class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 4, 6, 3)

class ResNet101(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet101, self).__init__(*args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 4, 23, 3)

class ResNet152(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet152, self).__init__(*args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 8, 36, 3)

def count_total_variables():
    return np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()]) 

if __name__ == '__main__':
    from os.path import join, abspath, dirname
    pretrain_root = abspath(join(dirname(__file__), '../../models/resnet/'))

    resnet50 = ResNet50('resnet50', True)
    resnet101 = ResNet101('resnet101', True)
    resnet152 = ResNet152('resnet152', True)
    
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(X)
    num_param = count_total_variables()
    print('our resnet50 has parameters: %d'%num_param)

    tf.reset_default_graph()
    tf.train.import_meta_graph(join(pretrain_root, 'ResNet-L50.meta'))
    num_param = count_total_variables()
    print('pretrained resnet50 parameters: %d'%num_param)

    tf.reset_default_graph()	
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet101(X)
    num_param = count_total_variables()
    print('our resnet101 has parameters: %d'%num_param)

    tf.reset_default_graph()
    tf.train.import_meta_graph(join(pretrain_root, 'ResNet-L101.meta'))
    num_param = count_total_variables()
    print('pretrained resnet101 parameters: %d'%num_param)
 
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet152(X)
    num_param = count_total_variables()
    print('our resnet152 has parameters: %d'%num_param)

    tf.reset_default_graph()
    tf.train.import_meta_graph(join(pretrain_root, 'ResNet-L152.meta'))
    num_param = count_total_variables()
    print('pretrained resnet152 parameters: %d'%num_param)
 

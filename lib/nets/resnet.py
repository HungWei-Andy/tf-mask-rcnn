import tensorflow as tf
from nets.network import Network
import numpy as np
# !! The default data format used here is NHWC !!
# TODO: scope

def conv_bn(X, inChannel, outChannel, kernel, istrain, stride=1, name=None):
    out = tf.layers.conv2d(X, outChannel, kernel, stride, 'same', use_bias=False, name=name)
    out = tf.layers.batch_normalization(out, training=istrain,
               name=name.replace('res', 'bn').replace('conv1', 'bn_conv1'))
    return out

def conv_bn_relu(X, inChannel, outChannel, kernel, istrain, stride=1, use_bias=False, name=None):
    out = conv_bn(X, inChannel, outChannel, kernel, istrain, stride=stride, name=name)
    out = tf.nn.relu(out)
    return out

def residual_conv(X, inChannel, interChannel, outputChannel, transition, istrain=False, name=None):
    if transition:
        init_stride = 2
    else:
        init_stride = 1

    if inChannel == outputChannel:
        skip_out = X
    else:
        skip_out = conv_bn(X, inChannel, outputChannel, 1, istrain,
                           stride=init_stride, name=name+'_branch1')
  
    conv_out = conv_bn_relu(X, inChannel, interChannel, 1, istrain, name=name+'_branch2a')
    conv_out = conv_bn_relu(conv_out, interChannel, interChannel, 3, istrain,
                            stride=init_stride, name=name+'_branch2b')
    conv_out = conv_bn(conv_out, interChannel, outputChannel, 1, istrain, name=name+'_branch2c')
    out = tf.nn.relu(skip_out + conv_out)
    return out

def residual_block(X, inChannel, interChannel, outputChannel, numLayers,
                   transition=True, istrain=False, number_name=True, name=None):
    if number_name and numLayers > 3:
        names = [name+'a'] + [name+'b'+str(i+1) for i in range(numLayers-1)]
    else:
        names = [name+chr(ord('a')+i) for i in range(numLayers)]

    out = residual_conv(X, inChannel, interChannel, outputChannel,
                        transition=transition, istrain=istrain, name=names[0])
    for i in range(numLayers - 1):
        out = residual_conv(out, outputChannel, interChannel, outputChannel,
                            transition=False, istrain=istrain, name=names[i+1])
    return out

class ResNet(Network):
    def __init__(self, scope=None, istrain=False, reuse=False):
        super(ResNet, self).__init__(scope)
        self.reuse = reuse
        self.istrain = istrain

    def _build_resnet(self, numBlock1, numBlock2, numBlock3, numBlock4, build_classifier=False):
        number_name = (self._scope != 'resnet50')
        self.conv1 = conv_bn_relu(self.input, 3, 64, 7, istrain=self.istrain, stride=2, name='conv1')
        self.pool1 = tf.layers.max_pooling2d(self.conv1, 3, 2, padding='same')    
        self.conv2 = residual_block(self.pool1, 64, 64, 256, numBlock1, transition=False,
                                    istrain=self.istrain, number_name=number_name, name='res2') 
        self.conv3 = residual_block(self.conv2, 256, 128, 512, numBlock2,
                                    istrain=self.istrain, number_name=number_name, name='res3')
        self.conv4 = residual_block(self.conv3, 512, 256, 1024, numBlock3,
                                    istrain=self.istrain, number_name=number_name, name='res4')
        self.conv5 = residual_block(self.conv4, 1024, 512, 2048, numBlock4,
                                    istrain=self.istrain, number_name=number_name, name='res5')
 
        out = self.conv5

        if build_classifier:
            self.pool5 = tf.layers.average_pooling2d(self.conv5, 7, 1)
            self.pool5_flat = tf.layers.flatten(self.pool5)
            self.scores = tf.layers.dense(self.pool5_flat, 1000, name='fc1000')
            out = self.scores
        return out

    def find_key_name(self, var):
        key, name = var.name.split('/')[-2:]
        if 'kernel' in name or 'weight' in name:
            name = 'weights'
        elif 'bias' in name:
            name = 'biases'
        elif 'mean' in name:
            name = 'mean'
        elif 'variance' in name:
            name = 'variance'
        elif 'gamma' in name:
            name = 'scale'
        elif 'beta' in name:
            name = 'offset'
        else:
            raise Exception('%s is not defined in official resnet deploy.txt'%name)
        return key, name

    def load(self, sess, pretrained_file):
        data = np.load(pretrained_file).item()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)
        for var in variables:
            key, name = self.find_key_name(var)
            sess.run(var.assign(data[key][name]))

class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__('resnet50', *args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 4, 6, 3)

class ResNet101(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet101, self).__init__('resnet101', *args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 4, 23, 3)

class ResNet152(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet152, self).__init__('resnet152', *args, **kwargs)
    def _build_network(self):
        return self._build_resnet(3, 8, 36, 3)



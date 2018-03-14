import tensorflow as tf
from nets.resnet import ResNet50, ResNet101, ResNet152
import numpy as np

def count_total_variables(verbose=False):
    if verbose:
        for var in tf.global_variables():
            print(var.name, var.get_shape().as_list())
    return np.sum([np.prod(var.get_shape().as_list()) for var in tf.global_variables()]) 

def count_total_npvars(data):
    num_params = 0
    for key in sorted(data.keys()):
        for name in sorted(data[key].keys()):
            num_params += np.prod(data[key][name].shape)
    return num_params

def check_shape(net):
    for var in tf.global_variables():
        key, name = net.find_key_name(var)
        var_shape = tuple(var.get_shape().as_list())
        np_shape = data[key][name].shape
        if var_shape != np_shape:
            print('incorrect var shape')
 
if __name__ == '__main__':
    from os.path import join, abspath, dirname
    pretrain_root = abspath(join(dirname(__file__), '../../model/pretrained_model/ori_resnet/'))

    resnet50 = ResNet50()
    resnet101 = ResNet101()
    resnet152 = ResNet152()

    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(X)
    data = np.load(join(pretrain_root, 'resnet50.npy'), encoding='bytes').item()
    print('our resnet50 has parameters: %d'%count_total_variables())
    print('pretrained resnet50 parameters: %d'%count_total_npvars(data))
    check_shape(resnet50)
 
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet101(X)
    data = np.load(join(pretrain_root, 'resnet101.npy'), encoding='bytes').item()
    print('our resnet101 has parameters: %d'%count_total_variables())
    print('pretrained resnet101 parameters: %d'%count_total_npvars(data))
    check_shape(resnet101)   
 
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet152(X)
    data = np.load(join(pretrain_root, 'resnet152.npy'), encoding='bytes').item()
    print('our resnet152 has parameters: %d'%count_total_variables())
    print('pretrained resnet152 parameters: %d'%count_total_npvars(data))
    check_shape(resnet152)

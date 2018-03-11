import tensorflow as tf
from nets.resnet import ResNet50, ResNet101, ResNet152
# !! The default data format used here is NHWC !!
# TODO: scope

if __name__ == '__main__':
    from os.path import join, abspath, dirname
    pretrain_root = abspath(join(dirname(__file__), '../../model/pretrained_model/ori_resnet/'))

    resnet50 = ResNet50()
    resnet101 = ResNet101()
    resnet152 = ResNet152()

    sess = tf.Session()
    
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(X) # resnet101, resnet152
    sess.run(tf.global_variables_initializer())
    resnet50.load(sess, join(pretrain_root, 'resnet50.npy'))

from nets.resnet import ResNet50
import tensorflow as tf
from os.path import join, dirname, abspath
import cv2
import numpy as np
from os.path import join, dirname, abspath
from os import listdir
from PIL import Image

def load_img(path):
    img = Image.open(path).convert('RGB')

    w, h = img.size
    s = min(h, w)
    newh, neww = int(224.0*h/s), int(224.0*w/s)
    img = img.resize((neww, newh), Image.BILINEAR)
    midw, midh = int(neww/2), int(newh/2)
    img = img.crop((midw-112, midh-112, midw+112, midh+112))
    img = np.array(img, dtype=np.float32)
    img -= np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
    img = img[np.newaxis, :, :, :]
    return img

def read_gt():
    gt_file = abspath(join(dirname(__file__), '../../dataset/ImageNet/val/groundtruth.txt'))
    with open(gt_file) as fout:
        lines = fout.readlines()
        gt = np.array([int(line) for line in lines])
    return gt

def read_imgs():
    img_dir = abspath(join(dirname(__file__), '../../dataset/ImageNet/val'))
    files = listdir(img_dir)
    images = [f for f in files if f.endswith('.jpeg')]
    images = sorted(images)
    images = [join(img_dir, img) for img in images]
    return images

def evaluate_mean(sess):
    for v in tf.global_variables():
        print('name: %s, mean: %f'%(v.name, 100*np.mean(v.eval(sess))))

def main():
    model = ResNet50(istrain=False)
    X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    gt = tf.placeholder(tf.int32, shape=[None])    
    y = model(X)
    for var in tf.global_variables():
        print(var.name, var.shape)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=y)

    sess = tf.Session()
    evaluate_mean(sess)
    sess.run(tf.global_variables_initializer())
    pretrained_file = abspath(join(dirname(__file__), '../../models/resnet/tensorflow/resnet50.py'))
    model.load(sess, pretrained_file)
    evaluate_mean(sess)

    #gts = read_gt()
    #images = read_imgs()
    #num_correct = 0
    #num_images = 0
    #for i in range(len(images)):
    #    image = load_img(images[i])
    #    score, ls = sess.run([y, loss], feed_dict={X:image, gt:np.array([gts[i]])})
    #    print(score.shape)
    #    print(gts[i])
    #    print(ls, score[0, gts[i]-5:gts[i]+5], score.shape)
    #    pred = np.squeeze(score).argmax()
    #    if pred == gts[i]:
    #        num_correct += 1
    #    num_images += 1
    #    print('accuracy: %d/%d, loss: %f, pred: %d, groundtruth: %d'%(num_correct, num_images, ls[0], pred, gts[i]))

if __name__ == '__main__':
    main()

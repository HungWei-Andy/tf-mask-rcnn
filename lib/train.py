import os
from os.path import join, dirname
import random
from skimage import io, color, util, transform
import numpy as np
import tensorflow as tf

import sys
sys.path.append(join(dirname(__file__), 'cocoapi', 'PythonAPI'))
sys.stdout = sys.stderr

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mask_rcnn import mask_rcnn
from config import cfg

class COCOLoader(object):
  def __init__(self, image_dir, shuffle=True):
    self.image_dir = image_dir
    self.coco = COCO(join(dirname(__file__), '..', 'COCO', 'annotations', 'instances_train2014.json'))
    self.imgIds = self.coco.getImgIds()
    self.shuffleIds = range(len(self.imgIds))
    if shuffle:
      random.shuffle(self.shuffleIds)

  def __len__(self):
    return len(self.imgIds)

  def __getitem__(self, image_index):
    imgInd = self.imgIds[self.shuffleIds[image_index]]
    img = self.coco.imgs[imgInd]
    height, width = img['height'], img['width']

    # read image
    image = io.imread(join(self.image_dir, img['file_name']))
    if image.ndim != 3:
      image = color.gray2rgb(image)
    mins = min(height, width)
    maxs = max(height, width)
    newh = cfg.image_size if height == mins else int(1.*height*cfg.image_size/mins)
    neww = cfg.image_size if width == mins else int(1.*width*cfg.image_size/mins)
    image = transform.resize(image, (newh, neww))
    offh = (newh-cfg.image_size) // 2
    offh = (offh, newh-cfg.image_size-offh)
    offw = (neww-cfg.image_size) // 2
    offw = (offw, neww-cfg.image_size-offw)
    image = util.crop(image, (offh, offw,(0,0)))
    image = image - cfg.image_mean
    
    # read annotations
    annIds = self.coco.getAnnIds(imgIds=imgInd)
    anns = self.coco.loadAnns(annIds)
    anns = [ann for ann in anns if not ann['iscrowd']]

    # load image mask, bbox and remove too small masks
    masks = []
    boxes = []
    cats = []
    for ann in anns:
      rle = self.coco.annToRLE(ann)
      m = maskUtils.decode(rle)
      if m.max() == 1:
        m = transform.resize(m, (newh, neww))
        m = util.crop(m, (offh, offw))
        masks.append(m)
        boxes.append(ann['bbox'])
        cats.append(ann['category_id'])
    masks = np.array(masks, np.float32)
    boxes = np.array(boxes, np.float32)
    cats = np.array(cats, np.int32)

    return {'img': image, 'box': boxes, 'mask': masks, 'class': cats}

def extract_batch(data, start_index):
  index = start_index
  num_loaded = 0
  imgs, boxes, masks, classes = [], [], [], []

  while num_loaded < cfg.batch_size:
    ele = data[index]
    index = (index + 1) % len(data) 

    if ele['box'].shape[0] == 0:
        continue
    else:
        imgs.append(ele['img'])
        boxes.append(ele['box'])
        masks.append(ele['mask'])
        classes.append(ele['class'])
        num_loaded += 1

  imgs = np.stack(imgs, axis=0)
  return imgs, boxes, classes, masks, index

def debug():
  image_dir = join(dirname(__file__), '..', 'COCO', 'images', 'train2014')
  data = COCOLoader(image_dir, shuffle=True)
  
  X = tf.placeholder(tf.float32, shape=(cfg.batch_size,
                                        cfg.image_size, cfg.image_size, 3))
  gt_boxes = [tf.placeholder(tf.float32, shape=(None, 4)) for i in range(cfg.batch_size)]
  gt_classes = [tf.placeholder(tf.int32, shape=[None]) for i in range(cfg.batch_size)]
  gt_masks = [tf.placeholder(tf.float32, shape=(None, cfg.image_size, cfg.image_size))
              for i in range(cfg.batch_size)]
  feat, net = mask_rcnn(X, True, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_masks=gt_masks)

  saver = tf.train.Saver(max_to_keep=100) 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    net.load(sess, join(dirname(__file__), '../model/pretrained_model/ori_resnet/resnet50.npy'))

    data_index = 0
    for i in range(cfg.iterations):
      train_img, train_box, train_cls, train_mask, data_index = extract_batch(data, data_index)
      feed_dict = {}
      feed_dict[X] = train_img
      for ind, box_tensor in enumerate(gt_boxes):
        feed_dict[box_tensor] = train_box[ind].reshape(-1, 4)
      for ind, cls_tensor in enumerate(gt_classes):
        feed_dict[cls_tensor] = train_cls[ind]
      for ind, mask_tensor in enumerate(gt_masks):
        feed_dict[mask_tensor] = train_mask[ind].reshape(-1, cfg.image_size, cfg.image_size)
      _ = sess.run(feat, feed_dict = feed_dict)
      #for key in sorted(_.keys()):
      #  if isinstance(_[key], list):
      #    print(key, len(_[key]))
      #    for j in range(len(_[key])):
      #      print(_[key][j].shape)
      #  elif not isinstance(_[key], tuple):
      #    print(key, _[key].shape)
      #print(_[0].shape, _[1].shape)
      print(_)
      print('iteration %d completed'%i)


def train(rpn_only=False):
  image_dir = join(dirname(__file__), '..', 'COCO', 'images', 'train2014')
  data = COCOLoader(image_dir, shuffle=True)
  
  # build mask rcnn network
  X = tf.placeholder(tf.float32, shape=(cfg.batch_size,
                                        cfg.image_size, cfg.image_size, 3))
  gt_boxes = [tf.placeholder(tf.float32, shape=(None, 4)) for i in range(cfg.batch_size)]
  gt_classes = [tf.placeholder(tf.int32, shape=[None]) for i in range(cfg.batch_size)]
  gt_masks = [tf.placeholder(tf.float32, shape=(None, cfg.image_size, cfg.image_size))
              for i in range(cfg.batch_size)]
  loss, net = mask_rcnn(X, True, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_masks=gt_masks)

  # renew for training only rpn
  if rpn_only:
    loss = {'all': loss['rpn'], 'rpn_cls': loss['rpn_cls'], 'rpn_loc': loss['rpn_loc']}

  # create optimization
  optimizer = tf.train.MomentumOptimizer(cfg.lr, cfg.momentum)
  gvs = optimizer.compute_gradients(loss['all'])
  newgvs = list()
  for ind, gv in enumerate(gvs):
    grad, var = gv
    if grad is None:
      print('Ignore: ', grad, var)
      continue
    print(grad, var)

    if 'bias' in var.name.lower():
      grad = grad * 2
    else:
      grad = grad + cfg.weight_decay * var
    newgvs.append((grad, var))
  opt = optimizer.apply_gradients(gvs)
  saver = tf.train.Saver(max_to_keep=100)

  with tf.Session() as sess:
    # add summary
    for key in loss.keys():
      tf.summary.scalar(key, loss[key])
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(cfg.summary_dir + '/train', sess.graph)
  
    # running training 
    sess.run(tf.global_variables_initializer())
    net.load(sess, join(dirname(__file__), '../model/pretrained_model/ori_resnet/resnet50.npy'))

    data_index = 0
    for i in range(cfg.iterations):
      train_img, train_box, train_cls, train_mask, data_index = extract_batch(data, data_index)
      
      feed_dict = {}
      feed_dict[X] = train_img
      for ind, box_tensor in enumerate(gt_boxes):
        feed_dict[box_tensor] = train_box[ind]
      for ind, cls_tensor in enumerate(gt_classes):
        feed_dict[cls_tensor] = train_cls[ind]
      for ind, mask_tensor in enumerate(gt_masks):
        feed_dict[mask_tensor] = train_mask[ind]

      if (i+1) % cfg.print_every == 0:
        summary, loss_val, _ = sess.run([merged_summary, loss, opt], feed_dict = feed_dict)
        print('===== Iterations: %d ====='%(i+1))
        for key in sorted(loss.keys()):
          print('%s loss: %.5f'%(key, loss_val[key]))
      else:
        summary, _ = sess.run([merged_summary, opt], feed_dict = feed_dict)
      train_writer.add_summary(summary, i)
      
      if (i+1) % cfg.save_every == 0:
        saver.save(sess, join(dirname(__file__), '..', 'output'), global_step=(i+1))

if __name__ == '__main__':
  train(rpn_only=True) #debug/train

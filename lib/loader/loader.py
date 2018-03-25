import os
from os.path import join, dirname
import random
from skimage import io, color, util, transform
import numpy as np
import matplotlib.pyplot as plt
import cv2
from config import cfg

class Loader(object):
  def __init__(self):
    self.index = 0
    self.epoch = 0

  def __len__(self):
    raise NotImplementedError

  def load_img(self, image_index):
    raise NotImplementedError

  def load_ann(self, image_index):
    raise NotImplementedError
 
  def __getitem__(self, image_index):
    # read image
    image_path, height, width = self.load_img(image_index)
    image = io.imread(image_path)
    if image.ndim != 3:
      image = color.gray2rgb(image)
    mins = min(height, width)
    maxs = max(height, width)
    scale = 1.*cfg.image_size/mins
    newh = cfg.image_size if height == mins else int(1.*height*cfg.image_size/mins)
    neww = cfg.image_size if width == mins else int(1.*width*cfg.image_size/mins)
    image = transform.resize(image, (newh, neww))
    offh = (newh-cfg.image_size) // 2
    offh = (offh, newh-cfg.image_size-offh)
    offw = (neww-cfg.image_size) // 2
    offw = (offw, neww-cfg.image_size-offw)
    image = util.crop(image, (offh, offw,(0,0)))
 
    # read annotations, load image mask, bbox and remove too small masks
    boxes = []
    masks = []
    cats = []
    anns = self.load_ann(image_index)
    for ann in anns:
      m = ann['mask']
      if m.max() == 1:
        cat = ann['category_id']
        if cat < 1 or cat > 80:
            continue

        box = ann['bbox']
        box = [v*scale for v in box]
        box[0], box[1] = box[0]-offw[0], box[1]-offh[0]
        box[2], box[3] = box[0]+box[2], box[1]+box[3]
        if (box[0] < 0 or box[1] < 0 or 
            box[2] > cfg.image_size-1 or box[2] > cfg.image_size-1):
            continue
        boxes.append(box)
        cats.append(cat)

        if not cfg.rpn_only:
          m = transform.resize(m, (newh, neww))
          m = util.crop(m, (offh, offw))
          mask = np.zeros((cfg.image_size, cfg.image_size, cfg.num_classes))
          mask[:,:,cat] = m
          masks.append(mask)
    
    boxes = np.array(boxes, np.float32)
    cats = np.array(cats, np.int32)
    if not cfg.rpn_only:
      masks = np.array(masks, np.float32)

    #demo_img = image
    #for i in range(boxes.shape[0]):
    #    cv2.rectangle(demo_img, tuple(boxes[i][:2]), tuple(boxes[i][2:]), (0,0,255))
    #plt.imshow(demo_img)
    #plt.show()

    return {'img': image, 'box': boxes, 'mask': masks, 'class': cats}

  def pop(self):
    ele = self[self.index]
    self.index = (self.index+1) % len(self)
    if self.index == 0:
      self.epoch += 1
    return ele

  def batch(self):
    num_loaded = 0
    imgs, boxes, masks, classes = [], [], [], []

    while num_loaded < cfg.batch_size:
      ele = self.pop()

      if ele['box'].shape[0] == 0:
        continue
      else:
        imgs.append(ele['img'])
        boxes.append(ele['box'])
        masks.append(ele['mask'])
        classes.append(ele['class'])
        num_loaded += 1

    imgs = np.stack(imgs, axis=0)
    imgs -= cfg.image_mean.reshape(1,1,1,3)
    return imgs, boxes, classes, masks

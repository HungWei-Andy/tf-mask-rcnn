import numpy as np
import os
from os.path import join, dirname
import random

import sys
sys.path.append(join(dirname(__file__), 'cocoapi', 'PythonAPI'))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from config import cfg
from .loader import Loader

class COCOLoader(Loader):
  def __init__(self, is_train=True, shuffle=True):
    super(COCOLoader, self).__init__()
    coco_dir = join(dirname(__file__), '..', '..', 'COCO')
    if is_train:
      self.image_dir = join(coco_dir, 'images', 'train2014')
      self.coco = COCO(join(coco_dir, 'annotations', 'instances_train2014.json'))
    else:
      self.image_dir = join(coco_dir, 'images', 'val2014')
      self.coco = COCO(join(coco_dir, 'annotations', 'instances_val2014.json'))
    self.imgIds = self.coco.getImgIds()
    self.catIds = self.coco.getCatIds()
    self.catId2label = dict(zip(self.catIds, range(1, len(self.catIds)+1)))
    self.shuffleIds = [i for i in range(len(self.imgIds))]
    if shuffle:
      random.shuffle(self.shuffleIds)

  def __len__(self):
    return len(self.imgIds)

  def load_img(self, image_index):
    imgInd = self.imgIds[self.shuffleIds[image_index]]
    img = self.coco.imgs[imgInd]
    height, width = img['height'], img['width']
    image_path = join(self.image_dir, img['file_name'])
    return image_path, height, width

  def load_ann(self, image_index):
    imgInd = self.imgIds[self.shuffleIds[image_index]]
    annIds = self.coco.getAnnIds(imgIds=imgInd)
    anns = self.coco.loadAnns(annIds)
    anns = [ann for ann in anns if not ann['iscrowd']]
    for i, ann in enumerate(anns):
      ann['gt_id'] = self.catId2label[ann['category_id']]
      if not cfg.rpn_only:
        rle = self.coco.annToRLE(ann)
        anns[i]['mask'] = np.array(maskUtils.decode(rle))
    return anns

import os
from os.path import join, dirname
import random
import skimage

import sys
sys.path.append(join(dirname(__file__), 'cocoapi', 'PythonAPI'))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

class COCOLoader(object):
  def __init__(self, image_dir, shuffle=True):
    self.image_dir = image_dir
    self.coco = COCO(join(dirname(__file__), '..', 'COCO', 'annotations', 'instances_train2014.json'))
    self.imgIds = self.coco.getImgIds()
    self.shuffleIds = range(len(self.imgIds))
    if shuffle:
      random.shuffle(self.index)

  def __len__(self):
    return len(self.imgInds)

  def __getitem__(self, image_index):
    imgInd = self.imgInds[self.shuffleIds[image_index]]
    img = self.coco.imgs[imgInd]
    height, width = img['height'], img['width']

    # read image
    image = skimage.io.imread(join(self.image_dir, img['file_name']))
    if image.ndim != 3:
      image = skimage.color.gray2rgb(image)
    
    # read annotations
    annIds = self.coco.getAnnIds(annIds=imgInd)
    anns = coco.loadAnns(annIds)
    anns = [ann for ann in anns if not ann['iscrowd']]

    # load image mask, bbox and remove too small masks
    masks = []
    boxes = []
    cats = []
    for ann in anns:
      rle = coco.annToRLE(ann, height, width)
      m = maskUtils.decode(rle)
      if m.max() == 1:
        masks.append(m)
        boxes.append(ann['bbox'])
        cats.append(ann['category_id'])

    return {'img': image, 'box': boxes, 'mask': masks, 'class': cats}

def train():
  image_dir = join(dirname(__file__), '..', 'COCO', 'images', 'train2014')
  data = COCOLoader(image_dir, shuffle=True)


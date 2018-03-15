# A reference toward https://github.com/matterport/Mask_RCNN/blob/master/coco.py

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval 
from pycocotools import mask as mask_utils

class COCODataset(object):
  def __init__(self, root, subset, year):
    self.coco = COCO('{}/annotations/instances_{}{}.json'.format(root, subset, year))

  def __len__(self):

  def __getitem__(self):
    

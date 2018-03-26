import os
from os.path import join, dirname
import random
import numpy as np
import tensorflow as tf
import sys

sys.stdout = sys.stderr
np.random.seed(0)
random.seed(0)

from loader.COCOLoader import COCOLoader
from mask_rcnn import mask_rcnn
from config import cfg
from targets import bbox_overlaps

def decode_roi_test(anchors, loc):
    '''
    Inputs
      - anchors: anchor boxes, an array of shape [N, H*W*N_anchor, 4]
      - loc: the location rpn logits, an array of shape [N, H*W*N_anchor, 4]
    Ouputs
      - boxes: the bbox coordinates (xmin, ymin, xmax, ymax),
               a tensor of shape [N, H*W*N_anchor, 4], the
               decoded [xmin, ymin, xmax, ymax] for each box
    '''
    H, W = 1.*cfg.image_size, 1.*cfg.image_size

    anc_widths = anchors[:,:,2] - anchors[:,:, 0]
    anc_heights = anchors[:,:,3] - anchors[:,:,1]
    anc_ctrx = anchors[:,:,0] + 0.5 * anc_widths
    anc_ctry = anchors[:,:,1] + 0.5 * anc_heights

    loc = loc * cfg.bbox_stddev.reshape(1,1,4) + cfg.bbox_mean.reshape(1,1,4)
    box_ctrx = loc[:,:,0] * (anc_widths+cfg.eps) + anc_ctrx
    box_ctry = loc[:,:,1] * (anc_heights+cfg.eps) + anc_ctry
    box_w = (anc_widths+cfg.eps) * (np.exp(loc[:,:,2])-cfg.log_eps)
    box_h = (anc_heights+cfg.eps) * (np.exp(loc[:,:,3])-cfg.log_eps)

    box_minx = np.maximum(0.0, np.minimum(W-1, box_ctrx - 0.5 * box_w))
    box_miny = np.maximum(0.0, np.minimum(H-1, box_ctry - 0.5 * box_h))
    box_maxx = np.maximum(0.0, np.minimum(W-1, box_ctrx + 0.5 * box_w))
    box_maxy = np.maximum(0.0, np.minimum(H-1, box_ctry + 0.5 * box_h))
    boxes = np.stack([box_minx, box_miny, box_maxx, box_maxy], axis=2)

    return boxes

def test(rpn_only=False):
  cfg.batch_size = 1
  root = join(dirname(__file__), '..')
  coco = COCOLoader(is_train=False, shuffle=False)
 
  # build mask rcnn network
  X = tf.placeholder(tf.float32, shape=(cfg.batch_size,
                                        cfg.image_size, cfg.image_size, 3))
  prediction_tensor = mask_rcnn(X, False, cfg.network)

  saver = tf.train.Saver() 
  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  with tf.Session(config=gpu_config) as sess:
    # restore
    sess.run(tf.global_variables_initializer())
    last_train_file = tf.train.latest_checkpoint(join(root, cfg.output_dir))
    if last_train_file is not None:
        print('restoring %s'%last_train_file)
        saver.restore(sess, last_train_file)

    # run testing
    num_gt, num_true_gt = np.zeros(cfg.num_classes), np.zeros(cfg.num_classes)
    num_pos, num_true_pos = np.zeros(cfg.num_classes), np.zeros(cfg.num_classes)
    for i in range(len(coco) // cfg.batch_size):
      train_img, train_box, train_cls, train_mask = coco.batch()
      
      feed_dict = {}
      proposals, probs, bbox_terms, masks = sess.run(prediction_tensor, feed_dict={X:train_img})

      if (i+1) % cfg.print_every == 0:
        print('%d/%d Done'%(i+1, len(coco)))

      # decode bboxes
      boxes = decode_roi_test(proposals, bbox_terms)
      masks = np.where(masks>0.5, 1, 0)

      # pack the results
      for batch_i in range(cfg.batch_size):
        pred, pred_label = [], []
        for box_i in range(proposals[batch_i].shape[0]):
          prop = proposals[batch_i, box_i]
          if prop[0]==0 and prop[1]==0 and prop[2]==0 and prop[3]==0:
            print('invalid proposal')
            continue

          prob = probs[batch_i, box_i]
          print(prob)
          label = prob.argmax()
          if label == 0:
            print('background')
            continue
          
          pred.append(boxes[batch_i, box_i])
          pred_label.append(label)          
        
        gt = train_box[batch_i]
        gt_label = train_cls[batch_i]
        pred = np.array(pred).reshape(-1, 4)
        pred_label = np.array(pred_label)
        print(pred.shape,gt.shape)    
        if pred.shape[0] == 0:
          continue
 
        iou = bbox_overlaps(gt, pred)

        num_gt[gt_label] += 1
        num_true_gt[gt_label] += iou.max(axis=1) > cfg.eval_iou_thresh
        num_pos[pred_label] += 1
        num_true_pos[pred_label] += iou.max(axis=0) > cfg.eval_iou_thresh

    print('mAP:%.4f'%((num_true_pos[1:] / num_pos[1:]).mean()))
    print('mRec:%.4f'%((num_true_gt[1:] / num_gt[1:]).mean()))

if __name__ == '__main__':
  test() #debug/train

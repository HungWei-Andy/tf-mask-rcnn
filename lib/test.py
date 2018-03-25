import os
from os.path import join, dirname
import random
import numpy as np
import tensorflow as tf

np.random.seed(0)
random.seed(0)

from loader import COCOLoader, extract_batch
from mask_rcnn import mask_rcnn
from config import cfg

def test(rpn_only=False):
  coco = COCOLoader(is_train=False, shuffle=False)
  
  # build mask rcnn network
  X = tf.placeholder(tf.float32, shape=(cfg.batch_size,
                                        cfg.image_size, cfg.image_size, 3))
  gt_boxes = [tf.placeholder(tf.float32, shape=(None, 4)) for i in range(cfg.batch_size)]
  gt_classes = [tf.placeholder(tf.int32, shape=[None]) for i in range(cfg.batch_size)]
  gt_masks = [tf.placeholder(tf.float32, shape=(None, cfg.image_size, cfg.image_size, cfg.num_classes))
              for i in range(cfg.batch_size)]
  loss, net = mask_rcnn(X, False, cfg.network, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_masks=gt_masks)
  #
  raise NotImplementedError

  saver = tf.train.Saver(max_to_keep=100) 
  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  with tf.Session(config=gpu_config) as sess:
    # restore
    sess.run(tf.global_variables_initializer())
    last_train_file = tf.train.latest_checkpoint(join(dirname(__file__), '../output'))
    if last_train_file is not None:
        start_iter = int(last_train_file.split('-')[-1])
        print('restoring %s'%last_train_file)
        saver.restore(sess, last_train_file)

    # run testing
    for i in range(start_iter, cfg.iterations):
      train_img, train_box, train_cls, train_mask = coco.batch()
      
      feed_dict = {}
      feed_dict[X] = train_img
      for ind, box_tensor in enumerate(gt_boxes):
        feed_dict[box_tensor] = train_box[ind]
      for ind, cls_tensor in enumerate(gt_classes):
        feed_dict[cls_tensor] = train_cls[ind]
      if not cfg.rpn_only:
        for ind, mask_tensor in enumerate(gt_masks):
          feed_dict[mask_tensor] = train_mask[ind]

      if (i+1) % cfg.print_every == 0:
        #lr, summary, loss_val, _ = sess.run([learning_rate, merged_summary, loss, opt], feed_dict = feed_dict)
        lr, loss_val, _ = sess.run([learning_rate, loss, opt], feed_dict=feed_dict)
        print('===== Iterations: %d ====='%(i+1))
        for key in sorted(loss.keys()):
          print('%s loss: %.5f'%(key, loss_val[key]))
        print('lr: %.8f'%lr)
      else:
        _ = sess.run(opt, feed_dict=feed_dict)
        #summary, _ = sess.run([merged_summary, opt], feed_dict = feed_dict)
      #train_writer.add_summary(summary, i)
      
      if (i+1) % cfg.save_every == 0:
        if rpn_only:
          model_name = 'rpn'
        else:
          model_name = 'model'
        saver.save(sess, join(dirname(__file__), '..', 'output', model_name), global_step=(i+1))

if __name__ == '__main__':
  train(rpn_only=cfg.rpn_only) #debug/train

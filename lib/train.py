import os
from os.path import join, dirname, isdir
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

def train(rpn_only=False):
  print(cfg.__dict__)
  root = join(dirname(__file__), '..')
  if not isdir(join(root, cfg.output_dir)):
    os.makedirs(join(root, cfg.output_dir))

  coco = COCOLoader(is_train=True, shuffle=True)
  
  # build mask rcnn network
  X = tf.placeholder(tf.float32, shape=(cfg.batch_size,
                                        cfg.image_size, cfg.image_size, 3))
  gt_boxes = [tf.placeholder(tf.float32, shape=(None, 4)) for i in range(cfg.batch_size)]
  gt_classes = [tf.placeholder(tf.int32, shape=[None]) for i in range(cfg.batch_size)]
  gt_masks = [tf.placeholder(tf.float32, shape=(None, cfg.image_size, cfg.image_size, cfg.num_classes))
              for i in range(cfg.batch_size)]
  loss, net = mask_rcnn(X, True, cfg.network, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_masks=gt_masks)

  # renew for training only rpn
  if rpn_only:
    loss = {'all': loss['rpn'], 'rpn_cls': loss['rpn_cls'], 'rpn_loc': loss['rpn_loc']}

  # learning rate
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(cfg.lr, global_step, cfg.decay_step, cfg.decay_rate, staircase=True)

  # create optimization
  if cfg.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer()
  elif cfg.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.momentum)
  else:
    raise NotImplementedError

  gvs = optimizer.compute_gradients(loss['all'])
  newgvs = list()
  for ind, gv in enumerate(gvs):
    grad, var = gv
    if grad is None or cfg.conv1_label in var.name:
      print('Ignore: ', grad, var)
      continue
    if cfg.cls_only and 'CLS' not in var.name:
      print('Ignore: ', grad, var)
      continue
    print(grad, var)

    #if 'bias' in var.name.lower() or 'beta' in var.name.lower():
    #  grad = grad * 2
    #else:
    #  grad = grad + cfg.weight_decay * var
    newgvs.append((grad, var))
  with tf.variable_scope('OPT') as scope:
    opt = optimizer.apply_gradients(gvs, global_step=global_step)

  saver = tf.train.Saver(max_to_keep=100) 
  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  with tf.Session(config=gpu_config) as sess:
    # add summary
    #for key in loss.keys():
    #  tf.summary.scalar(key, loss[key])
    #merged_summary = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(cfg.summary_dir + '/train', sess.graph)
  
    # running training 
    sess.run(tf.global_variables_initializer())
    net.load(sess, join(root, cfg.pretrained_model))

    # restore
    start_iter = 0
    last_train_file = tf.train.latest_checkpoint(join(root, cfg.output_dir))
    if last_train_file is not None:
        start_iter = int(last_train_file.split('-')[-1])
        print('restoring %s'%last_train_file)
        saver.restore(sess, last_train_file)
        coco.index = cfg.batch_size * start_iter % len(coco)

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
        saver.save(sess, join(root, cfg.output_dir, cfg.model_name), global_step=(i+1))

if __name__ == '__main__':
  train(rpn_only=cfg.rpn_only) #debug/train

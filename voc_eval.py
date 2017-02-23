# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for VOC 2007.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import voc
import json

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'ADEF/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'ADEF/train/vgg16-model.ckpt-50w',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_class', 95,
                            """The number of classes in dataset""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 2835,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.flags.DEFINE_string("class_list", 'ADEF/class_list.txt',
                       "Txt file contain class names")
tf.app.flags.DEFINE_string('test_pattern', 'ADEF/TFR/test-?????-of-00002',
                           """The TF record file pattern for test set""")


def get_metrics(logits, labels, sess, num_class, thresh=0.5):
  true_condiction = num_class*[0]
  pred_condiction = num_class*[0]
  true_positive = num_class*[0]

  assert logits.get_shape() == labels.get_shape(),\
    'Logits and labels length unequal!'

  [logits_, labels_] = sess.run([logits, labels])
  for i in xrange(0, len(logits_)):
    for j in xrange(0, num_class):
      if labels_[i][j] != 0:
        true_condiction[j] += 1
        if logits_[i][j] >= thresh:
          true_positive[j] += 1
      if logits_[i][j] >= thresh:
        pred_condiction[j] += 1

  return true_condiction, pred_condiction, true_positive


def get_logit_label(logits, labels, sess):
  [logits_batch, labels_batch] = sess.run([logits, labels])
  return logits_batch, labels_batch


def get_bool_result(logits, labels, sess, num_class, thresh=0.5):
  logits_bool = num_class * [False]
  labels_bool = num_class * [False]

  assert logits.get_shape() == labels.get_shape(),\
    'Logits and labels length unequal!'

  [logits_, labels_] = sess.run([logits, labels])
  for i in xrange(0, len(logits_)):
    for j in xrange(0, num_class):
      if labels_[i][j] != 0:
        labels_bool[j] = True
      if logits_[i][j] >= thresh:
        logits_bool[j] += 1

  return logits_bool, labels_bool


def get_average_precision(logit, label):
  index = np.argsort(logit)[::-1]
  total_obj = sum(label)
  hit= 0.
  average_precision = 0.
  # print('Total objects: %d' % total_obj)
  for i in xrange(len(logit)):
    if label[index[i]] != 0:
      hit += 1
      average_precision += hit/(i+1)
      if hit == total_obj:
        # print('threshold: %.3f, number %d' % (logit[index[i]], i))
        break

  return average_precision/total_obj


def eval_once(saver, summary_writer, logits, labels, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    mean accuracy percentage: mAP.
    summary_op: Summary op.
  """
  with tf.Session() as sess:

    saver.restore(sess, FLAGS.checkpoint_dir)
    num_class = FLAGS.num_class

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      pre = []
      gt = []

      step = 0
      while step < num_iter and not coord.should_stop():
        step += 1
        pre_batch, gt_batch = get_logit_label(logits, labels, sess)
        pre.extend(pre_batch.tolist())
        gt.extend(gt_batch.tolist())
        if not step % 20:
          print('Collecting logits and labels for batch %d' % step)

      print ('---Collect all predicted labels and ground true labels--------')
      result = {'ground_true': gt, 'predicted': pre}
      pre = np.array(pre)
      gt = np.array(gt)
      average_precisions = []
      class_list = [l.rstrip('\n') for l in open(FLAGS.class_list, 'r')]
      for i in xrange(num_class):
        ap_per = get_average_precision(pre[:, i], gt[:, i])
        print('Class name: p=%.3f' % (ap_per))
        average_precisions.append(ap_per)

      result['mAP'] = average_precisions
      with open('result.json', 'w') as json_file:
        json.dump(result, json_file)

      # tp = num_class * [0]
      # tc = num_class * [0]
      # pc = num_class * [0]
      #
      # step = 0
      # while step < num_iter and not coord.should_stop():
      #   tc_batch, pc_batch, tp_batch = get_metrics(logits, labels, sess, 20)
      #   tc = np.sum([tc, tc_batch], axis=0)
      #   pc = np.sum([pc, pc_batch], axis=0)
      #   tp = np.sum([tp, tp_batch], axis=0)
      #   if not step % 20:
      #     print('Mean average precision after batch %d' % step)
      #   step += 1
      #
      # print ('-------------------------------------------------------')
      # recall = np.divide(tp, tc, dtype=float)
      # precision = np.divide(tp, pc, dtype=float)
      # for i in xrange(20):
      #  print ('Recall @0.5: %.3f , Precision @0.5: %.3f' % (recall[i], precision[i]))



      # mAP = []
      # update = []
      # for i in xrange(0, FLAGS.num_class):
      #  mAP_pc, update_pc = tf.contrib.metrics.streaming_sparse_average_precision_at_k(
      #    logits[:, i], labels[:, i], 10)
      #  mAP.append(mAP_pc)
      #  update.append(update_pc)
      # sess.run(tf.initialize_local_variables())
      # step = 0
      # while step < num_iter and not coord.should_stop():
      #   step += 1
      #   sess.run(update)
      #   if not step % 20:
      #    print('Mean average precision after batch %d' % step)
      #
      #  #print('Final Mean average precision: %f' % mAP.eval())
      # result = sess.run(mAP)
      # print ('---------------------------------------------')

      # for e in xrange(0, len(result)):
      #  print ('Mean average precision: %.3f' % result[e])

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval VOC 07 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels from ADEF dataset for testing.
    images, labels = voc.inputs(is_training=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = voc.vgg_inference(images, is_training=False)
    labels = tf.cast(labels, dtype=tf.int64)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        voc.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, logits, labels, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()

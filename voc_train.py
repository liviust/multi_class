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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import voc
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'data/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('ckpt_file', 'data/vgg_16.ckpt',
                           """Pre-trained checkpoint file""")
tf.app.flags.DEFINE_string('train_pattern',
                           # 'ADEF/TFR/train-?????-of-00016',
                           'data/tf/train-?????-of-00002',
                           """The TF record file pattern for train set""")
tf.app.flags.DEFINE_integer('num_class', 20,
                            'The number of class.')
tf.app.flags.DEFINE_string('test_pattern', 'ADEF/TFR/test-?????-of-00002',
                           """The TF record file pattern for test set""")


def main(argv=None):

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  """Train VOC multi-label for a number of steps."""
  # Build the TF graph
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels from TFR files.
    images, labels = voc.inputs(is_training=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = voc.vgg_fc_inference(images)

    # Calculate loss.
    loss = voc.vgg_fc_loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = voc.train(loss, global_step)

    # Create a saver.
    variables_to_save = slim.get_variables(scope='vgg_16', suffix='ExponentialMovingAverage')
    saver = tf.train.Saver(variables_to_save)

    # Create a restorer.
    restore_variables = slim.get_model_variables()
    restorer = tf.train.Saver(restore_variables[:-6])

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    sess.run(init)

    # Restore tensors from checkpoint file for pre-trained model
    restorer.restore(sess, FLAGS.ckpt_file)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'vgg16-model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
  tf.app.run()

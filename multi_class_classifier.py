# Write by Julius@lenovo 2016.12.26

""" Inference labels from an image
    Give a jpeg image, run the multi-label classifier
    and print the class names
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import voc
import matplotlib.pyplot as plt
import os.path as osp

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "ADEF/train/vgg16-model.ckpt-50w",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files",
                       "ADEF/test_1/028.jpg",
                       # '/home/julius/Data/Flowers_53/jpg6/image_03964.jpg',
                       # '/home/julius/Data/ADE20K_2016_07_26/images/validation/a/art_studio/ADE_val_00001038.jpg',
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("class_list", 'ADEF/class_list.txt',
                       "Txt file contain class names")
tf.flags.DEFINE_string('threshold_file', 'ADEF/thresholds.txt',
                       'Txt file store thresholds for each class')


def preprocess_image(encoded_image, height=224, width=224):
  """ Transform input image into CNN import format
      Decode, Resize, Reshape
  Args:
      encoded_image: Encoded image data
      height: Height for CNN input
      width: Width for CNN input
  Returns:
      image: Tensor with shape [1, height, width, 3]
  """
  # Decode jpeg image
  image = tf.image.decode_jpeg(encoded_image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)


  # Resize image without crop
  image = tf.image.resize_images(image,
                                 size=[height, width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)

  # Reshape image from 3 dim to 4 dim
  image = tf.reshape(image, [-1, 224, 224, 3])
  return image


def display_result(result):
  img = plt.imread(FLAGS.input_files)
  gt = [l.rstrip('\n') for l in open('ADEF/result/ground_true.txt', 'r')]
  length = []
  length.extend(tf.gfile.Glob('ADEF/result/*.jpg'))
  gt = gt[len(length)-1].split(' ')
  plt.figure()
  plt.imshow(img)
  plt.title('GT: {} \n EST: {}'.format(gt, result))
  plt.savefig(osp.join('ADEF/result/', osp.basename(FLAGS.input_files)))


def predict_classes(predict, class_list_path, threshold_file):
  # Read the class name and thresholds from txt files
  class_list = [l.rstrip('\n') for l in open(class_list_path, 'r')]
  thresholds = [float(l.rstrip('\r\n')) for l in open(threshold_file, 'r')]

  result = []
  # Predict result with thresh and print
  for i, e in enumerate(predict):
    if e > thresholds[i]:
      print('Class name: %s, p=%.3f' % (class_list[i+1], predict[i]))
      result.append(class_list[i+1])
  display_result(result)


def main(_):
  # Build the inference graph.
  with tf.Graph().as_default() as g:
      with tf.gfile.GFile(FLAGS.input_files, "r") as f:
        encoded_image = f.read()

      # Test
      config = tf.ConfigProto(device_count={'GPU': 0})
      # sess = tf.Session(config=config)
      sess = tf.InteractiveSession(config=config)

      # Preprocess image
      image = preprocess_image(encoded_image)


      # Get the predict logit from image
      logit = voc.vgg_predict(image)

      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(
          voc.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)

      saver.restore(sess, FLAGS.checkpoint_path)
      predict = sess.run(logit)

      predict_classes(predict, FLAGS.class_list, FLAGS.threshold_file)


if __name__ == '__main__':
    tf.app.run()
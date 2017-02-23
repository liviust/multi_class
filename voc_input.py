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

"""Routine for decoding the VOC 2007 TF Record format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = [224, 256]
# HEIGHTS = [224, 256]
# WIDTHS = [224, 256]
# Global constants describing the data set.
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 23385
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2835

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5001
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4952



def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])


# ---------------------Julius-----------------------------------------------
def voc_read(filename_queue, num_class):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  # Parse the TF record example
  context, _ = tf.parse_single_sequence_example(
      serialized_example,
      context_features={
          'image/data': tf.FixedLenFeature([], dtype=tf.string),
          'image/label_fix': tf.FixedLenFeature([num_class], dtype=tf.int64)
      }
  )
  encoded_image = context['image/data']
  label = tf.cast(context['image/label_fix'], tf.float32)
  return encoded_image, label


def distort_image(image, thread_id):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color", values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image


def process_image(encoded_image,
                  is_training,
                  # height=HEIGHTS,
                  # width=WIDTHS,
                  resize_height=346,
                  resize_width=346,
                  # thread_id=0,
                  image_format="jpeg"):
  """ Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  # def image_summary(name, image):
  #   if not thread_id:
  #     tf.image_summary(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  # if resize_height:
  #   image = tf.image.resize_images(image,
  #                                  size=[resize_height, resize_width],
  #                                  method=tf.image.ResizeMethod.BILINEAR)
  #
  # # Crop to final dimensions.
  # if is_training:
  #   image = tf.random_crop(image, [height, width, 3])
  # else:
  #   # Central crop, assuming resize_height > height, resize_width > width.
  #   image = tf.image.resize_image_with_crop_or_pad(image, height, width)


  # Multi-scale processing, choice
  image_size = random.choice(IMAGE_SIZE)
  height = image_size
  width = image_size
  # Resize image without crop
  image = tf.image.resize_images(image,
                                 size=[height, width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  # Randomly distort the image.
  if is_training:
    image = distort_image(image, 0)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)
  return image
# -----------------------------------------------------------------------------


def distorted_inputs(batch_size, pattern, num_class, is_training=True):
  """Construct distorted input for VOC training using the Reader ops.

  Args:
    data_dir: Path to the TF record data directory.
    batch_size: Number of images per batch.
    pattern: TF record file pattern
    num_class: Integer. Identity the number of classes in dataset

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 2D tensor of [batch_size, num_class] size.
  """

  filenames = []
  for p in pattern.split(","):
    filenames.extend(tf.gfile.Glob(p))
  if not filenames:
    tf.logging.fatal("Found no input files matching %s", pattern)
    raise ValueError('Please supply TF record files')
  else:
    # tf.logging.info("Prefetching values from %d files matching %s",
    #                 len(filenames), pattern)
    print('Prefetching values from %d files matching %s' % (len(filenames), pattern))

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  encoded_image, label = voc_read(filename_queue, num_class)

  # Preprocess images
  print ('Preprocessing images, this will take a few minutes .. heiheihei...')
  image = process_image(encoded_image, is_training=is_training)

  # Shuffle the examples and collect them into batch_size batches.
  # (Internally uses a RandomShuffleQueue.)
  # We run this in eight threads to avoid being a bottleneck.
  if is_training:
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
  else:
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=5*batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, labels


# def eval_inputs(batch_size, pattern, num_class):
#   """Construct evalation input for VOC evaluation using the Reader ops.
#
#   Args:
#     batch_size: Number of images per batch.
#     pattern: TF record file pattern
#     num_class: Integer. Identity the number of classes in dataset
#
#   Returns:
#     images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#     labels: Labels. 2D tensor of [batch_size, num_class] size.
#   """
#   filenames = []
#   for p in pattern.split(","):
#       filenames.extend(tf.gfile.Glob(p))
#   if not filenames:
#       tf.logging.fatal("Found no input files matching %s", pattern)
#       raise ValueError('Please supply TF record files')
#   else:
#       tf.logging.info("Prefetching values from %d files matching %s",
#                       len(filenames), pattern)
#
#   # Create a queue that produces the filenames to read.
#   filename_queue = tf.train.string_input_producer(filenames)
#
#   # Read examples from files in the filename queue.
#   encoded_image, label = voc_read(filename_queue, num_class)
#
#   # Preprocess images
#   print('Preprocessing images, this will take a few minutes .. heiheihei...')
#   image = process_image(encoded_image, is_training=False)
#
#   # Shuffle the examples and collect them into batch_size batches.
#   # (Internally uses a RandomShuffleQueue.)
#   # We run this in two threads to avoid being a bottleneck.
#   images, labels = tf.train.shuffle_batch(
#       [image, label], batch_size=batch_size, num_threads=2,
#       capacity=1000 + 3 * batch_size,
#       # Ensures a minimum amount of shuffling of examples.
#       min_after_dequeue=1000)
#
#   # Display the training images in the visualizer.
#   tf.image_summary('images', images)
#
#   return images, labels

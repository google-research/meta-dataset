# coding=utf-8
# Copyright 2020 The Meta-Dataset Authors.
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

# Lint as: python2, python3
"""Tools for preparing datasets for integration in the benchmark.

Specifically, the DatasetConverter class is used to perform the conversion of a
dataset to the format necessary for its addition in the benchmark. This involves
creating a DatasetSpecification for the dataset in question, and creating (and
storing) a tf.record for every one of its classes.

Some subclasses make use of a "split file", which is a JSON file file that
stores a dictionary whose keys are 'train', 'valid', and 'test' and whose values
indicate the corresponding classes assigned to these splits. Note that not all
datasets require a split file. For example it may be the case that a dataset
indicates the intended assignment of classes to splits via their structure (e.g.
all train classes live in a 'train' folder etc).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import collections
import io
import itertools
import json
import operator
import os
import traceback

from absl import logging
from meta_dataset.data import dataset_spec as ds_spec
from meta_dataset.data import imagenet_specification
from meta_dataset.data import learning_spec
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy.io import loadmat
import six
from six.moves import range
import six.moves.cPickle as pkl
import tensorflow.compat.v1 as tf

# Datasets in the same order as reported in the article.
# 'ilsvrc_2012_data_root' is already defined in imagenet_specification.py.
tf.flags.DEFINE_string(
    'ilsvrc_2012_num_leaf_images_path', '',
    'A path used as a cache for a dict mapping the WordNet id of each Synset '
    'of a ILSVRC 2012 class to its number of images. If empty, it defaults to '
    '"ilsvrc_2012/num_leaf_images.json" inside records_root.')

tf.flags.DEFINE_string(
    'omniglot_data_root',
    '',
    'Path to the root of the omniglot data.')

tf.flags.DEFINE_string(
    'aircraft_data_root',
    '',
    'Path to the root of the FGVC-Aircraft Benchmark.')

tf.flags.DEFINE_string(
    'cu_birds_data_root',
    '',
    'Path to the root of the CU-Birds dataset.')

tf.flags.DEFINE_string(
    'dtd_data_root',
    '',
    'Path to the root of the Describable Textures Dataset.')

tf.flags.DEFINE_string(
    'quickdraw_data_root',
    '',
    'Path to the root of the quickdraw data.')

tf.flags.DEFINE_string(
    'fungi_data_root',
    '',
    'Path to the root of the fungi data.')

tf.flags.DEFINE_string(
    'vgg_flower_data_root',
    '',
    'Path to the root of the VGG Flower data.')

tf.flags.DEFINE_string(
    'traffic_sign_data_root',
    '',
    'Path to the root of the Traffic Sign dataset.')

tf.flags.DEFINE_string(
    'mscoco_data_root',
    '',
    'Path to the root of the MSCOCO images and annotations. The root directory '
    'should have a subdirectory `train2017` and an annotation JSON file '
    '`instances_train2017.json`. Both can be downloaded from MSCOCO website: '
    'http://cocodataset.org/#download and unzipped into the root directory.')

# Diagnostics-only dataset.
tf.flags.DEFINE_string(
    'mini_imagenet_data_root',
    '',
    'Path to the root of the MiniImageNet data.')

# Output flags.
tf.flags.DEFINE_string(
    'records_root', '',
    'The root directory storing all tf.Records of datasets.')

tf.flags.DEFINE_string('splits_root', '',
                       'The root directory storing the splits of datasets.')

FLAGS = tf.flags.FLAGS
DEFAULT_FILE_PATTERN = '{}.tfrecords'
TRAIN_TEST_FILE_PATTERN = '{}_{}.tfrecords'
AUX_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
VGGFLOWER_LABELS_PATH = os.path.join(AUX_DATA_PATH,
                                     'VggFlower_labels.txt')
TRAFFICSIGN_LABELS_PATH = os.path.join(AUX_DATA_PATH, 'TrafficSign_labels.txt')


def make_example(features):
  """Creates an Example protocol buffer.

  Create a protocol buffer with an integer feature for the class label, and a
  bytes feature for the input (image or feature)

  Args:
    features: sequence of (key, feature_type, value) tuples. Features to encode
      in the Example. `key` corresponds to the feature name, `feature_type` can
      either be 'int64', 'float32', or 'bytes', and `value` corresponds to the
      feature itself.

  Returns:
    example_serial: A string corresponding to the serialized example.

  """

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  feature_fns = {
      'int64': _int64_feature,
      'float32': _float32_feature,
      'bytes': _bytes_feature
  }

  feature_dict = dict((key, feature_fns[feature_type](value))
                      for key, feature_type, value in features)

  # Create an example protocol buffer.
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  example_serial = example.SerializeToString()
  return example_serial


def write_example(data_bytes,
                  class_label,
                  writer,
                  input_key='image',
                  label_key='label'):
  """Create and write an Example protocol buffer for the given image.

  Create a protocol buffer with an integer feature for the class label, and a
  bytes feature for the image.

  Args:
    data_bytes: bytes, an encoded image representation or serialized feature.
      For images, the usual encoding is JPEG, but could be different
      as long as the DataProvider's record_decoder accepts it.
    class_label: the integer class label of the image.
    writer: a TFRecordWriter
    input_key: String used as key for the input (image of feature).
    label_key: String used as key for the label.
  """
  example = make_example([(input_key, 'bytes', [data_bytes]),
                          (label_key, 'int64', [class_label])])
  writer.write(example)


def gen_rand_split_inds(num_train_classes, num_valid_classes, num_test_classes):
  """Generates a random set of indices corresponding to dataset splits.

  It assumes the indices go from [0, num_classes), where the num_classes =
  num_train_classes + num_val_classes + num_test_classes. The returned indices
  are non-overlapping and cover the entire range.

  Note that in the current implementation, valid_inds and test_inds are sorted,
  but train_inds is in random order.

  Args:
    num_train_classes : int, number of (meta)-training classes.
    num_valid_classes : int, number of (meta)-valid classes.
    num_test_classes : int, number of (meta)-test classes.

  Returns:
    train_inds : np array of training inds.
    valid_inds : np array of valid inds.
    test_inds  : np array of test inds.
  """
  num_trainval_classes = num_train_classes + num_valid_classes
  num_classes = num_trainval_classes + num_test_classes

  # First split into trainval and test splits.
  trainval_inds = np.random.choice(
      num_classes, num_trainval_classes, replace=False)
  test_inds = np.setdiff1d(np.arange(num_classes), trainval_inds)
  # Now further split trainval into train and val.
  train_inds = np.random.choice(trainval_inds, num_train_classes, replace=False)
  valid_inds = np.setdiff1d(trainval_inds, train_inds)

  logging.info(
      'Created splits with %d train, %d validation and %d test classes.',
      len(train_inds), len(valid_inds), len(test_inds))
  return train_inds, valid_inds, test_inds


def write_tfrecord_from_npy_single_channel(class_npy_file, class_label,
                                           output_path):
  """Create and write a tf.record file for the data of a class.

  This assumes that the provided .npy file stores the data of a given class in
  an array of shape [num_images_of_given_class, side**2].
  In the case of the Quickdraw dataset for example, side = 28.
  Each row of that array is interpreted as a single-channel side x side image,
  read into a PIL.Image, converted to RGB and then written into a record.
  Args:
    class_npy_file: the .npy file of the images of class class_label.
    class_label: the label of the class that a Record is being made for.
    output_path: the location to write the Record.

  Returns:
    The number of images in the .npy file for class class_label.
  """

  def load_image(img):
    """Load image img.

    Args:
      img: a 1D numpy array of shape [side**2]

    Returns:
      a PIL Image
    """
    # We make the assumption that the images are square.
    side = int(np.sqrt(img.shape[0]))
    # To load an array as a PIL.Image we must first reshape it to 2D.
    img = Image.fromarray(img.reshape((side, side)))
    img = img.convert('RGB')
    return img

  with tf.io.gfile.GFile(class_npy_file, 'rb') as f:
    imgs = np.load(f)

  # If the values are in the range 0-1, bring them to the range 0-255.
  if imgs.dtype == np.bool:
    imgs = imgs.astype(np.uint8)
    imgs *= 255

  writer = tf.python_io.TFRecordWriter(output_path)
  # Takes a row each time, i.e. a different image (of the same class_label).
  for image in imgs:
    img = load_image(image)
    # Compress to JPEG before writing
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    write_example(buf.getvalue(), class_label, writer)

  writer.close()
  return len(imgs)


def write_tfrecord_from_image_files(class_files,
                                    class_label,
                                    output_path,
                                    invert_img=False,
                                    bboxes=None,
                                    output_format='JPEG',
                                    skip_on_error=False):
  """Create and write a tf.record file for the images corresponding to a class.

  Args:
    class_files: the list of paths to images of class class_label.
    class_label: the label of the class that a record is being made for.
    output_path: the location to write the record.
    invert_img: change black pixels to white ones and vice versa. Used for
      Omniglot for example to change the black-background-white-digit images
      into more conventional-looking white-background-black-digit ones.
    bboxes: list of bounding boxes, one for each filename passed as input. If
      provided, images are cropped to those bounding box values.
    output_format: a string representing a PIL.Image encoding type: how the
      image data is encoded inside the tf.record. This needs to be consistent
      with the record_decoder of the DataProvider that will read the file.
    skip_on_error: whether to skip an image if there is an issue in reading it.
      The default it to crash and report the original exception.

  Returns:
    The number of images written into the records file.
  """

  def load_and_process_image(path, bbox=None):
    """Process the image living at path if necessary.

    If the image does not need any processing (inverting, converting to RGB
    for instance), and is in the desired output_format, then the original
    byte representation is returned.

    If that is not the case, the resulting image is encoded to output_format.

    Args:
      path: the path to an image file (e.g. a .png file).
      bbox: bounding box to crop the image to.

    Returns:
      A bytes representation of the encoded image.
    """
    with tf.io.gfile.GFile(path, 'rb') as f:
      image_bytes = f.read()
    try:
      img = Image.open(io.BytesIO(image_bytes))
    except:
      logging.warn('Failed to open image: %s', path)
      raise

    img_needs_encoding = False

    if img.format != output_format:
      img_needs_encoding = True
    if img.mode != 'RGB':
      img = img.convert('RGB')
      img_needs_encoding = True
    if bbox is not None:
      img = img.crop(bbox)
      img_needs_encoding = True
    if invert_img:
      img = ImageOps.invert(img)
      img_needs_encoding = True

    if img_needs_encoding:
      # Convert the image into output_format
      buf = io.BytesIO()
      img.save(buf, format=output_format)
      buf.seek(0)
      image_bytes = buf.getvalue()
    return image_bytes

  writer = tf.python_io.TFRecordWriter(output_path)
  written_images_count = 0
  for i, path in enumerate(class_files):
    bbox = bboxes[i] if bboxes is not None else None
    try:
      img = load_and_process_image(path, bbox)
    except (IOError, tf.errors.PermissionDeniedError) as e:
      if skip_on_error:
        logging.warn('While trying to load file %s, got error: %s', path, e)
      else:
        raise
    else:
      # This gets executed only if no Exception was raised
      write_example(img, class_label, writer)
      written_images_count += 1

  writer.close()
  return written_images_count


def write_tfrecord_from_directory(class_directory,
                                  class_label,
                                  output_path,
                                  invert_img=False,
                                  files_to_skip=None,
                                  skip_on_error=False):
  """Create and write a tf.record file for the images corresponding to a class.

  Args:
    class_directory: the home of the images of class class_label.
    class_label: the label of the class that a record is being made for.
    output_path: the location to write the record.
    invert_img: change black pixels to white ones and vice versa. Used for
      Omniglot for example to change the black-background-white-digit images
      into more conventional-looking white-background-black-digit ones.
    files_to_skip: a set containing names of files that should be skipped if
      present in class_directory.
    skip_on_error: whether to skip an image if there is an issue in reading it.
      The default it to crash and report the original exception.

  Returns:
    The number of images written into the records file.
  """
  if files_to_skip is None:
    files_to_skip = set()
  class_files = []
  filenames = sorted(tf.io.gfile.listdir(class_directory))
  for filename in filenames:
    if filename in files_to_skip:
      logging.info('skipping file %s', filename)
      continue
    filepath = os.path.join(class_directory, filename)
    if tf.io.gfile.isdir(filepath):
      continue
    class_files.append(filepath)

  written_images_count = write_tfrecord_from_image_files(
      class_files,
      class_label,
      output_path,
      invert_img,
      skip_on_error=skip_on_error)

  if not skip_on_error:
    assert len(class_files) == written_images_count
  return written_images_count


# TODO(goroshin): Make sure to use this function where appropriate.
def encode_image(img, image_format):
  """Get image encoded bytes from numpy array.

     Note: use lossless PNG compression to test perfect reconstruction.
  Args:
    img: A numpy array of uint8 with shape [image_size, image_size, 3].
    image_format: A string describing the image compression format.

  Returns:
    contents: The compressed image serialized to a string of bytes.
  """
  img = Image.fromarray(img)
  buf = io.BytesIO()
  img.save(buf, image_format)
  buf.seek(0)
  img_bytes = buf.getvalue()
  buf.close()
  return img_bytes
class DatasetConverter(object):
  """Converts a dataset to the format required to integrate it in the benchmark.

  In particular, this involves:
  1) Creating a tf.record file for each class of the dataset.
  2) Creating an instance of DatasetSpecification or BiLevelDatasetSpecification
    (as appropriate) for the dataset. This includes information about the
    splits, classes, super-classes if applicable, etc that is required for
    creating episodes from the dataset.

  1) and 2) are accomplished by calling the convert_dataset() method.
  This will create and write the dataset specification and records in
  self.records_path.
  """

  def __init__(self,
               name,
               data_root,
               has_superclasses=False,
               records_path=None,
               split_file=None,
               random_seed=22):
    """Initialize a DatasetConverter.

    Args:
      name: the name of the dataset
      data_root: the root of the dataset
      has_superclasses: Whether the dataset's classes are organized in a two
        level hierarchy of coarse and fine classes. In that case, a
        BiLevelDatasetSpecification will be created.
      records_path: optional path to store the created records. If it's not
        provided, the default path for the dataset will be used.
      split_file: optional path to a file storing the training, validation and
        testing splits of the dataset's classes. If provided, it's a JSON file
        that stores a dictionary whose keys are 'train', 'valid', and 'test' and
        whose values indicate the corresponding classes assigned to these
        splits. Note that not all datasets require a split file. For example it
        may be the case that a dataset indicates the intended assignment of
        classes to splits via their structure (e.g. all train classes live in a
        'train' folder etc).
      random_seed: a random seed used for creating splits (when applicable) in a
        reproducible way.
    """
    self.name = name
    self.data_root = data_root
    self.has_superclasses = has_superclasses
    self.seed = random_seed
    if records_path is None:
      records_path = os.path.join(FLAGS.records_root, name)
    tf.io.gfile.makedirs(records_path)
    self.records_path = records_path

    # Where to write the DatasetSpecification instance.
    self.dataset_spec_path = os.path.join(self.records_path,
                                          'dataset_spec.json')

    self.split_file = split_file
    if self.split_file is None:
      self.split_file = os.path.join(FLAGS.splits_root,
                                     '{}_splits.json'.format(self.name))
      tf.io.gfile.makedirs(FLAGS.splits_root)

    # Sets self.dataset_spec to an initial DatasetSpecification or
    # BiLevelDatasetSpecification.
    self._init_specification()

  def _init_data_specification(self):
    """Sets self.dataset_spec to an initial DatasetSpecification."""
    # Maps each Split to the number of classes assigned to it.
    self.classes_per_split = {
        learning_spec.Split.TRAIN: 0,
        learning_spec.Split.VALID: 0,
        learning_spec.Split.TEST: 0
    }

    self._create_data_spec()

  def _init_bilevel_data_specification(self):
    """Sets self.dataset_spec to an initial BiLevelDatasetSpecification."""
    # Maps each Split to the number of superclasses assigned to it.
    self.superclasses_per_split = {
        learning_spec.Split.TRAIN: 0,
        learning_spec.Split.VALID: 0,
        learning_spec.Split.TEST: 0
    }

    # Maps each superclass id to the number of classes it contains.
    self.classes_per_superclass = collections.defaultdict(int)

    # Maps each superclass id to the name of its class.
    self.superclass_names = {}

    self._create_data_spec()

  def _init_specification(self):
    """Returns an initial DatasetSpecification or BiLevelDatasetSpecification.

    Creates this instance using initial values that need to be overwritten in
    every sub-class implementing the converter for a different dataset. In
    particular, in the case of a DatasetSpecification, each sub-class must
    overwrite the 3 following fields accordingly: classes_per_split,
    images_per_class, and class_names. In the case of its bi-level counterpart,
    each sub-class must overwrite: superclasses_per_split,
    classes_per_superclass, images_per_class, superclass_names, and class_names.
    In both cases, this happens in create_dataset_specification_and_records().
    Note that if other, non-mutable fields are updated, or if these objects are
    replaced with other ones, see self._create_data_spec() to create a new spec.
    """
    # First initialize the fields that are common to both types of data specs.
    # Maps each class id to its number of images.
    self.images_per_class = collections.defaultdict(int)

    # Maps each class id to the name of its class.
    self.class_names = {}

    # Pattern that each class' filenames should adhere to.
    self.file_pattern = DEFAULT_FILE_PATTERN

    if self.has_superclasses:
      self._init_bilevel_data_specification()
    else:
      self._init_data_specification()

  def _create_data_spec(self):
    """Create a new [BiLevel]DatasetSpecification given the fields in self.

    Set self.dataset_spec to that new object. After the initial creation,
    this is needed in the case of datasets with example-level splits, since
    file_pattern and images_per_class have to be replaced by new objects.
    """
    if self.has_superclasses:
      self.dataset_spec = ds_spec.BiLevelDatasetSpecification(
          self.name, self.superclasses_per_split, self.classes_per_superclass,
          self.images_per_class, self.superclass_names, self.class_names,
          self.records_path, self.file_pattern)
    else:
      self.dataset_spec = ds_spec.DatasetSpecification(
          self.name, self.classes_per_split, self.images_per_class,
          self.class_names, self.records_path, self.file_pattern)

  def convert_dataset(self):
    """Converts dataset as required to integrate it in the benchmark.

    Wrapper for self.create_dataset_specification_and_records() which does most
    of the work. This method additionally handles writing the finalized
    DatasetSpecification to the designated location.
    """
    self.create_dataset_specification_and_records()

    # Write the DatasetSpecification to the designated location.
    self.write_data_spec()

  def create_dataset_specification_and_records(self):
    """Creates a DatasetSpecification and records for the dataset.

    Specifically, the work that needs to be done here is twofold:
    Firstly, the initial values of the following attributes need to be updated:
    1) self.classes_per_split: a dict mapping each split to the number of
      classes assigned to it
    2) self.images_per_class: a dict mapping each class to its number of images
    3) self.class_names: a dict mapping each class (e.g. 0) to its (string) name
      if available.
    This automatically results to updating self.dataset_spec as required.

    Important note: Must assign class ids in a certain order:
    lowest ones for training classes, then for validation classes and highest
    ones for testing classes.
    The reader data sources operate under this assumption.

    Secondly, a tf.record needs to be created and written for each class. There
    are some general functions at the top of this file that may be useful for
    this (e.g. write_tfrecord_from_npy_single_channel,
    write_tfrecord_from_image_files).
    """
    raise NotImplementedError('Must be implemented in each sub-class.')

  def read_splits(self):
    """Reads the splits for the dataset from self.split_file.

    This will not always be used (as we noted earlier there are datasets that
    define the splits in other ways, e.g. via structure of their directories).

    Returns:
      A splits dictionary mapping each split to a list of class names belonging
      to it, or False upon failure (e.g. the splits do not exist).
    """
    logging.info('Attempting to read splits from %s...', self.split_file)
    if tf.io.gfile.exists(self.split_file):
      with tf.io.gfile.GFile(self.split_file, 'r') as f:
        try:
          splits = json.load(f)
        except json.decoder.JSONDecodeError:
          logging.info('Unsuccessful: file exists, but loading failed. %s',
                       traceback.format_exc())
          return False
        logging.info('Successful.')
        return splits
    else:
      logging.info('Unsuccessful.')
      return False

  def write_data_spec(self):
    """Write the dataset's specification to a JSON file."""
    with tf.io.gfile.GFile(self.dataset_spec_path, 'w') as f:
      # Use 2-space indentation (which also add newlines) for legibility.
      json.dump(self.dataset_spec.to_dict(), f, indent=2)

  def get_splits(self, force_create=False):
    """Returns the class splits.

    If the splits already exist in the designated location, they are simply
    read. Otherwise, they are created. For this, first reset the random seed to
    self.seed for reproducibility, then create the splits and finally writes
    them to the designated location.
    The actual split creation takes place in self.create_splits() which each
    sub-class must override.

    Args:
      force_create: bool. if True, the splits will be created even if they
        already exist.

    Returns:
      splits: a dictionary whose keys are 'train', 'valid', and 'test', and
      whose values are lists of the corresponding class names.
    """
    # Check if the splits already exist.
    if not force_create:
      splits = self.read_splits()
      if splits:
        return splits

    # First, re-set numpy's random seed, for reproducibility.
    np.random.seed(self.seed)

    # Create the dataset-specific splits.
    splits = self.create_splits()

    # Finally, write the splits in the designated location.
    logging.info('Saving new splits for dataset %s at %s...', self.name,
                 self.split_file)
    with tf.io.gfile.GFile(self.split_file, 'w') as f:
      json.dump(splits, f, indent=2)
    logging.info('Done.')

    return splits

  def create_splits(self):
    """Create class splits.

    Specifically, create a dictionary whose keys are 'train', 'valid', and
    'test', and whose values are lists of the corresponding classes.
    """
    raise NotImplementedError('Must be implemented in each sub-class.')


class OmniglotConverter(DatasetConverter):
  """Prepares Omniglot as required for integrating it in the benchmark.

  Omniglot is organized into two high-level directories, referred to as
  the background and evaluation sets, respectively, with the former
  intended for training and the latter for testing. Each of these contains a
  number of sub-directories, corresponding to different alphabets.
  Each alphabet directory in turn has a number of sub-folders, each
  corresponding to a character, which stores 20 images of that character, each
  drawn by a different person.
  We consider each character to be a different class for our purposes.
  The following diagram illustrates this struture.

  omniglot_root
  |- images_background
     |- alphabet
        |- character
           |- images of character
        ...
  |- images_evaluation
    |- alphabet
        |- character
           |- images of character
        ...
  """

  def __init__(self, *args, **kwargs):
    """Initialize an OmniglotConverter."""
    # Make has_superclasses default to True for the Omniglot dataset.
    if 'has_superclasses' not in kwargs:
      kwargs['has_superclasses'] = True
    super(OmniglotConverter, self).__init__(*args, **kwargs)

  def parse_split_data(self, split, alphabets, alphabets_path):
    """Parse the data of the given split.

    Specifically, update self.class_names, self.images_per_class, and
    self.classes_per_split with the information for the given split, and
    create and write records of the classes of the given split.

    Args:
      split: an instance of learning_spec.Split
      alphabets: the list of names of alphabets belonging to split
      alphabets_path: the directory with the folders corresponding to alphabets.
    """
    # Each alphabet is a superclass.
    for alphabet_folder_name in alphabets:
      alphabet_path = os.path.join(alphabets_path, alphabet_folder_name)
      # Each character is a class.
      for char_folder_name in sorted(tf.io.gfile.listdir(alphabet_path)):
        class_path = os.path.join(alphabet_path, char_folder_name)
        class_label = len(self.class_names)
        class_records_path = os.path.join(
            self.records_path,
            self.dataset_spec.file_pattern.format(class_label))
        self.class_names[class_label] = '{}-{}'.format(alphabet_folder_name,
                                                       char_folder_name)
        self.images_per_class[class_label] = len(
            tf.io.gfile.listdir(class_path))

        # Create and write the tf.Record of the examples of this class.
        write_tfrecord_from_directory(
            class_path, class_label, class_records_path, invert_img=True)

        # Add this character to the count of subclasses of this superclass.
        superclass_label = len(self.superclass_names)
        self.classes_per_superclass[superclass_label] += 1

      # Add this alphabet as a superclass.
      self.superclasses_per_split[split] += 1
      self.superclass_names[superclass_label] = alphabet_folder_name

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records().

    We use Lake's original train/test splits as we believe this is a more
    challenging setup and because we like that it's hierarchically structured.
    We also held out a subset of that train split to act as our validation set.
    Specifically, the 5 alphabets from that set with the least number of
    characters were chosen for this purpose.
    """

    # We chose the 5 smallest alphabets (i.e. those with the least characters)
    # out of the 'background' set of alphabets that are intended for train/val
    # We keep the 'evaluation' set of alphabets for testing exclusively
    # The chosen alphabets have 14, 14, 16, 17, and 20 characters, respectively.
    validation_alphabets = [
        'Blackfoot_(Canadian_Aboriginal_Syllabics)',
        'Ojibwe_(Canadian_Aboriginal_Syllabics)',
        'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Tagalog',
        'Alphabet_of_the_Magi'
    ]

    training_alphabets = []
    data_path_trainval = os.path.join(self.data_root, 'images_background')
    for alphabet_name in sorted(tf.io.gfile.listdir(data_path_trainval)):
      if alphabet_name not in validation_alphabets:
        training_alphabets.append(alphabet_name)
    assert len(training_alphabets) + len(validation_alphabets) == 30

    data_path_test = os.path.join(self.data_root, 'images_evaluation')
    test_alphabets = sorted(tf.io.gfile.listdir(data_path_test))
    assert len(test_alphabets) == 20

    self.parse_split_data(learning_spec.Split.TRAIN, training_alphabets,
                          data_path_trainval)
    self.parse_split_data(learning_spec.Split.VALID, validation_alphabets,
                          data_path_trainval)
    self.parse_split_data(learning_spec.Split.TEST, test_alphabets,
                          data_path_test)


class QuickdrawConverter(DatasetConverter):
  """Prepares Quickdraw as required to integrate it in the benchmark."""

  def create_splits(self):
    """Create splits for Quickdraw and store them in the default path."""
    # Quickdraw is stored in a number of .npy files, one for every class
    # with each .npy file storing an array containing the images of that class.
    class_npy_files = sorted(tf.io.gfile.listdir(self.data_root))
    class_names = [fname[:fname.find('.')] for fname in class_npy_files]
    # Sort the class names, for reproducibility.
    class_names.sort()
    num_classes = len(class_npy_files)
    # Split into train, validation and test splits that have 70% / 15% / 15%
    # of the data, respectively.
    num_trainval_classes = int(0.85 * num_classes)
    num_train_classes = int(0.7 * num_classes)
    num_valid_classes = num_trainval_classes - num_train_classes
    num_test_classes = num_classes - num_trainval_classes

    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        num_train_classes, num_valid_classes, num_test_classes)
    splits = {
        'train': [class_names[i] for i in train_inds],
        'valid': [class_names[i] for i in valid_inds],
        'test': [class_names[i] for i in test_inds]
    }
    return splits

  def parse_split_data(self, split, split_class_names):
    """Parse the data of the given split.

    Specifically, update self.class_names, self.images_per_class, and
    self.classes_per_split with the information for the given split, and
    create and write records of the classes of the given split.

    Args:
      split: an instance of learning_spec.Split
      split_class_names: the list of names of classes belonging to split
    """
    for class_name in split_class_names:
      self.classes_per_split[split] += 1
      class_label = len(self.class_names)
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_label))

      # The names of the files in self.data_root for Quickdraw are of the form
      # class_name.npy, for example airplane.npy.
      class_npy_fname = class_name + '.npy'
      self.class_names[class_label] = class_name
      class_path = os.path.join(self.data_root, class_npy_fname)

      # Create and write the tf.Record of the examples of this class.
      num_imgs = write_tfrecord_from_npy_single_channel(class_path, class_label,
                                                        class_records_path)
      self.images_per_class[class_label] = num_imgs

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records.

    If no split file is provided, and the default location for Quickdraw splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    The splits for this dataset are represented as a dictionary mapping each of
    'train', 'valid', and 'test' to a list of class names. For example the value
    associated with the key 'train' may be ['angel', 'clock', ...].
    """

    splits = self.get_splits()
    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.parse_split_data(learning_spec.Split.TRAIN, train_classes)
    self.parse_split_data(learning_spec.Split.VALID, valid_classes)
    self.parse_split_data(learning_spec.Split.TEST, test_classes)


class CUBirdsConverter(DatasetConverter):
  """Prepares CU-Birds dataset as required to integrate it in the benchmark."""
  # There are 200 classes in CU-Birds.
  NUM_TRAIN_CLASSES = 140
  NUM_VALID_CLASSES = 30
  NUM_TEST_CLASSES = 30
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  def create_splits(self):
    """Create splits for CU-Birds and store them in the default path.

    If no split file is provided, and the default location for CU-Birds splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """

    with tf.io.gfile.GFile(os.path.join(self.data_root, 'classes.txt'),
                           'r') as f:
      class_names = []
      for lines in f:
        _, class_name = lines.strip().split(' ')
        class_names.append(class_name)

    err_msg = 'number of classes in dataset does not match split specification'
    assert len(class_names) == self.NUM_TOTAL_CLASSES, err_msg

    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)
    splits = {
        'train': [class_names[i] for i in train_inds],
        'valid': [class_names[i] for i in valid_inds],
        'test': [class_names[i] for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    image_root_folder = os.path.join(self.data_root, 'images')
    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    for class_id, class_label in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_label)
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_label
      class_directory = os.path.join(image_root_folder, class_label)
      self.images_per_class[class_id] = len(
          tf.io.gfile.listdir(class_directory))
      write_tfrecord_from_directory(class_directory, class_id,
                                    class_records_path)


class VGGFlowerConverter(DatasetConverter):
  """Prepares VGG Flower as required to integrate it in the benchmark."""
  # There are 102 classes in the VGG Flower dataset. A 70% / 15% / 15% split
  # between train, validation and test maps to roughly 71 / 15 / 16 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 71
  NUM_VALID_CLASSES = 15
  NUM_TEST_CLASSES = 16
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES
  ID_LEN = 3

  def create_splits(self):
    """Create splits for VGG Flower and store them in the default path.

    If no split file is provided, and the default location for VGG Flower splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class integers.
    """
    # Load class names from the text file
    file_path = VGGFLOWER_LABELS_PATH
    with tf.io.gfile.GFile(file_path) as fd:
      all_lines = fd.read()
    # First line is expected to be a comment.
    class_names = all_lines.splitlines()[1:]
    err_msg = 'number of classes in dataset does not match split specification'
    assert len(class_names) == self.NUM_TOTAL_CLASSES, err_msg

    # Provided class labels are numbers started at 1.
    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)
    format_str = '%%0%dd.%%s' % self.ID_LEN
    splits = {
        'train': [format_str % (i + 1, class_names[i]) for i in train_inds],
        'valid': [format_str % (i + 1, class_names[i]) for i in valid_inds],
        'test': [format_str % (i + 1, class_names[i]) for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""
    splits = self.get_splits()
    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    imagelabels_path = os.path.join(self.data_root, 'imagelabels.mat')
    with tf.io.gfile.GFile(imagelabels_path, 'rb') as f:
      labels = loadmat(f)['labels'][0]
    filepaths = collections.defaultdict(list)
    for i, label in enumerate(labels):
      filepaths[label].append(
          os.path.join(self.data_root, 'jpg', 'image_{:05d}.jpg'.format(i + 1)))

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    # Class IDs are constructed in such a way that
    #   - training class IDs lie in [0, num_train_classes),
    #   - validation class IDs lie in
    #     [num_train_classes, num_train_classes + num_validation_classes), and
    #   - test class IDs lie in
    #     [num_train_classes + num_validation_classes, num_classes).
    for class_id, class_label in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_label)
      # We encode the original ID's in the label.
      original_id = int(class_label[:self.ID_LEN])
      class_paths = filepaths[original_id]
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_label
      self.images_per_class[class_id] = len(class_paths)
      # Create and write the tf.Record of the examples of this class.
      write_tfrecord_from_image_files(class_paths, class_id, class_records_path)


class DTDConverter(DatasetConverter):
  """Prepares DTD as required to integrate it in the benchmark."""
  # There are 47 classes in the Describable Textures Dataset. A 70% / 15% / 15%
  # split between train, validation and test maps to roughly 33 / 7 / 7 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 33
  NUM_VALID_CLASSES = 7
  NUM_TEST_CLASSES = 7

  def create_splits(self):
    """Create splits for DTD and store them in the default path.

    If no split file is provided, and the default location for DTD splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of strings (class names).
    """
    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)
    class_names = sorted(
        tf.io.gfile.listdir(os.path.join(self.data_root, 'images')))
    splits = {
        'train': [class_names[i] for i in train_inds],
        'valid': [class_names[i] for i in valid_inds],
        'test': [class_names[i] for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))

    for class_id, class_name in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_name)
      class_directory = os.path.join(self.data_root, 'images', class_name)
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_name
      # 'waffled' class directory has a leftover '.directory' file.
      files_to_skip = set()
      if class_name == 'waffled':
        files_to_skip.add('.directory')
      self.images_per_class[class_id] = write_tfrecord_from_directory(
          class_directory,
          class_id,
          class_records_path,
          files_to_skip=files_to_skip)


class AircraftConverter(DatasetConverter):
  """Prepares Aircraft as required to integrate it in the benchmark."""
  # There are 100 classes in the Aircraft dataset. A 70% / 15% / 15%
  # split between train, validation and test maps to 70 / 15 / 15
  # classes, respectively.
  NUM_TRAIN_CLASSES = 70
  NUM_VALID_CLASSES = 15
  NUM_TEST_CLASSES = 15

  def create_splits(self):
    """Create splits for Aircraft and store them in the default path.

    If no split file is provided, and the default location for Aircraft splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of strings (class names).
    """
    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)
    # "Variant" refers to the aircraft model variant (e.g., A330-200) and is
    # used as the class name in the dataset.
    variants_path = os.path.join(self.data_root, 'data', 'variants.txt')
    with tf.io.gfile.GFile(variants_path, 'r') as f:
      variants = [line.strip() for line in f.readlines() if line]
    variants = sorted(variants)
    assert len(variants) == (
        self.NUM_TRAIN_CLASSES + self.NUM_VALID_CLASSES + self.NUM_TEST_CLASSES)

    splits = {
        'train': [variants[i] for i in train_inds],
        'valid': [variants[i] for i in valid_inds],
        'test': [variants[i] for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    # Retrieve mapping from filename to bounding box.
    # Cropping to the bounding boxes is important for two reasons:
    # 1) The dataset documentation mentions that "[the] (main) aircraft in each
    #    image is annotated with a tight bounding box [...]", which suggests
    #    that there may be more than one aircraft in some images. Cropping to
    #    the bounding boxes removes ambiguity as to which airplane the label
    #    refers to.
    # 2) Raw images have a 20-pixel border at the bottom with copyright
    #    information which needs to be removed. Cropping to the bounding boxes
    #    has the side-effect that it removes the border.
    bboxes_path = os.path.join(self.data_root, 'data', 'images_box.txt')
    with tf.io.gfile.GFile(bboxes_path, 'r') as f:
      names_to_bboxes = [
          line.split('\n')[0].split(' ') for line in f.readlines()
      ]
      names_to_bboxes = dict(
          (name, map(int, (xmin, ymin, xmax, ymax)))
          for name, xmin, ymin, xmax, ymax in names_to_bboxes)

    # Retrieve mapping from filename to variant
    variant_trainval_path = os.path.join(self.data_root, 'data',
                                         'images_variant_trainval.txt')
    with tf.io.gfile.GFile(variant_trainval_path, 'r') as f:
      names_to_variants = [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]

    variant_test_path = os.path.join(self.data_root, 'data',
                                     'images_variant_test.txt')
    with tf.io.gfile.GFile(variant_test_path, 'r') as f:
      names_to_variants += [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]

    names_to_variants = dict(names_to_variants)

    # Build mapping from variant to filenames. "Variant" refers to the aircraft
    # model variant (e.g., A330-200) and is used as the class name in the
    # dataset. The position of the class name in the concatenated list of
    # training, validation, and test class name constitutes its class ID.
    variants_to_names = collections.defaultdict(list)
    for name, variant in names_to_variants.items():
      variants_to_names[variant].append(name)

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    assert set(variants_to_names.keys()) == set(all_classes)

    for class_id, class_name in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_name)
      class_files = [
          os.path.join(self.data_root, 'data', 'images',
                       '{}.jpg'.format(filename))
          for filename in sorted(variants_to_names[class_name])
      ]
      bboxes = [
          names_to_bboxes[name]
          for name in sorted(variants_to_names[class_name])
      ]
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_name
      self.images_per_class[class_id] = len(class_files)

      write_tfrecord_from_image_files(
          class_files, class_id, class_records_path, bboxes=bboxes)


class TrafficSignConverter(DatasetConverter):
  """Prepares Traffic Sign as required to integrate it in the benchmark."""
  # There are 43 classes in the Traffic Sign dataset, all of which are used for
  # test episodes.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 43
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  def create_splits(self):
    """Create splits for Traffic Sign and store them in the default path.

    If no split file is provided, and the default location for Traffic Sign
    splits does not contain a split file, a
    self.NUM_TRAIN_CLASSES / self.NUM_VALID_CLASSES / self.NUM_TEST_CLASSES
    split is created and stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """
    # Load class names from the text file
    file_path = TRAFFICSIGN_LABELS_PATH
    with tf.io.gfile.GFile(file_path) as fd:
      all_lines = fd.read()
    # First line is expected to be a comment.
    class_names = all_lines.splitlines()[1:]

    err_msg = 'number of classes in dataset does not match split specification'
    assert len(class_names) == self.NUM_TOTAL_CLASSES, err_msg

    splits = {
        'train': [],
        'valid': [],
        'test': [
            '%02d.%s' % (i, class_names[i])
            for i in range(self.NUM_TEST_CLASSES)
        ]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    for class_id, class_label in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_label)
      # The raw dataset file uncompresses to `GTSRB/Final_Training/Images/`.
      # The `Images` subdirectory contains 43 subdirectories (one for each
      # class) whose names are zero-padded, 5-digit strings representing the
      # class number. data_root should be the path to the GTSRB directory.
      class_directory = os.path.join(self.data_root, 'Final_Training', 'Images',
                                     '{:05d}'.format(class_id))
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_label
      # We skip `GT-?????.csv` files, which contain addditional annotations.
      self.images_per_class[class_id] = write_tfrecord_from_directory(
          class_directory,
          class_id,
          class_records_path,
          files_to_skip=set(['GT-{:05d}.csv'.format(class_id)]))


class MSCOCOConverter(DatasetConverter):
  """Prepares MSCOCO as required to integrate it in the benchmark."""

  # There are 80 classes in the MSCOCO dataset. A 0% / 50% / 50% split
  # between train, validation and test maps to roughly 0 / 40 / 40 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 40
  NUM_TEST_CLASSES = 40

  def __init__(self,
               name,
               data_root,
               records_path=None,
               split_file=None,
               image_subdir_name='train2017',
               annotation_json_name='instances_train2017.json',
               box_scale_ratio=1.2):
    self.num_all_classes = (
        self.NUM_TRAIN_CLASSES + self.NUM_VALID_CLASSES + self.NUM_TEST_CLASSES)
    image_dir = os.path.join(data_root, image_subdir_name)
    if not tf.io.gfile.isdir(image_dir):
      raise ValueError('Directory %s does not exist' % image_dir)
    self.image_dir = image_dir

    annotation_path = os.path.join(data_root, annotation_json_name)
    if not tf.io.gfile.exists(annotation_path):
      raise ValueError('Annotation file %s does not exist' % annotation_path)
    with tf.io.gfile.GFile(annotation_path, 'r') as json_file:
      annotations = json.load(json_file)
      instance_annotations = annotations['annotations']
      if not instance_annotations:
        raise ValueError('Instance annotations is empty.')
      self.coco_instance_annotations = instance_annotations
      categories = annotations['categories']
      if len(categories) != self.num_all_classes:
        raise ValueError(
            'Total number of MSCOCO classes %d should be equal to the sum of '
            'train, val, test classes %d.' %
            (len(categories), self.num_all_classes))
      self.coco_categories = categories
    self.coco_name_to_category = {cat['name']: cat for cat in categories}

    if box_scale_ratio < 1.0:
      raise ValueError('Box scale ratio must be greater or equal to 1.0.')
    self.box_scale_ratio = box_scale_ratio

    super(MSCOCOConverter, self).__init__(name, data_root, records_path,
                                          split_file)

  def create_splits(self):
    """Create splits for MSCOCO and store them in the default path.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """
    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)

    splits = {
        'train': [self.coco_categories[i]['name'] for i in train_inds],
        'valid': [self.coco_categories[i]['name'] for i in valid_inds],
        'test': [self.coco_categories[i]['name'] for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""
    splits = self.get_splits()
    self.classes_per_split[learning_spec.Split.TRAIN] = len(splits['train'])
    self.classes_per_split[learning_spec.Split.VALID] = len(splits['valid'])
    self.classes_per_split[learning_spec.Split.TEST] = len(splits['test'])
    all_classes = list(
        itertools.chain(splits['train'], splits['valid'], splits['test']))

    # Map original COCO "id" to class ids that conform to DatasetConverter's
    # contract.
    coco_id_to_class_id = {}
    for class_id, class_name in enumerate(all_classes):
      self.class_names[class_id] = class_name
      category = self.coco_name_to_category[class_name]
      coco_id_to_class_id[category['id']] = class_id

    def get_image_crop_and_class_id(annotation):
      """Gets image crop and its class label."""
      image_id = annotation['image_id']
      image_path = os.path.join(self.image_dir, '%012d.jpg' % image_id)
      # The bounding box is represented as (x_topleft, y_topleft, width, height)
      bbox = annotation['bbox']
      coco_class_id = annotation['category_id']
      class_id = coco_id_to_class_id[coco_class_id]

      with tf.io.gfile.GFile(image_path, 'rb') as f:
        # The image shape is [?, ?, 3] and the type is uint8.
        image = Image.open(f)
        image = image.convert(mode='RGB')
        image_w, image_h = image.size

        def scale_box(bbox, scale_ratio):
          x, y, w, h = bbox
          x = x - 0.5 * w * (scale_ratio - 1.0)
          y = y - 0.5 * h * (scale_ratio - 1.0)
          w = w * scale_ratio
          h = h * scale_ratio
          return [x, y, w, h]

        x, y, w, h = scale_box(bbox, self.box_scale_ratio)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size
        if crop_width <= 0 or crop_height <= 0:
          raise ValueError('crops are not valid.')
      return image_crop, class_id

    class_tf_record_writers = []
    for class_id in range(self.num_all_classes):
      output_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      class_tf_record_writers.append(tf.python_io.TFRecordWriter(output_path))

    for i, annotation in enumerate(self.coco_instance_annotations):
      try:
        image_crop, class_id = get_image_crop_and_class_id(annotation)
      except IOError:
        logging.warning('Image can not be opened and will be skipped.')
        continue
      except ValueError:
        logging.warning('Image can not be cropped and will be skipped.')
        continue

      logging.info('writing image %d/%d', i,
                   len(self.coco_instance_annotations))

      # TODO(manzagop): refactor this, e.g. use write_tfrecord_from_image_files.
      image_crop_bytes = io.BytesIO()
      image_crop.save(image_crop_bytes, format='JPEG')
      image_crop_bytes.seek(0)

      write_example(image_crop_bytes.getvalue(), class_id,
                    class_tf_record_writers[class_id])
      self.images_per_class[class_id] += 1

    for writer in class_tf_record_writers:
      writer.close()


class ImageNetConverter(DatasetConverter):
  """Prepares ImageNet for integration in the benchmark.

  Different from most datasets that are getting converted here, for
  ImageNet we define a HierarchicalDatasetSpecification which has different
  attributes from a standard DatasetSpecification.

  Only the "training" split of the original ImageNet dataset will be used.

  Images that are shared with other datasets (Caltech for instance) are
  skipped, so that examples from the test sets are not inadvertently
  used during training.
  """

  def _create_data_spec(self):
    """Initializes the HierarchicalDatasetSpecification instance for ImageNet.

    See HierarchicalDatasetSpecification for details.
    """
    # Load lists of image names that are duplicates with images in other
    # datasets. They will be skipped from ImageNet.
    self.files_to_skip = set()
    for other_dataset in ('Caltech101', 'Caltech256', 'CUBirds'):
      duplicates_file = os.path.join(
          AUX_DATA_PATH,
          'ImageNet_{}_duplicates.txt'.format(other_dataset))

      with tf.io.gfile.GFile(duplicates_file) as fd:
        duplicates = fd.read()
      lines = duplicates.splitlines()

      for l in lines:
        # Skip comment lines
        l = l.strip()
        if l.startswith('#'):
          continue
        # Lines look like:
        # 'synset/synset_imgnumber.JPEG  # original_file_name.jpg\n'.
        # Extract only the 'synset_imgnumber.JPG' part.
        file_path = l.split('#')[0].strip()
        file_name = os.path.basename(file_path)
        self.files_to_skip.add(file_name)
    ilsvrc_2012_num_leaf_images_path = FLAGS.ilsvrc_2012_num_leaf_images_path
    if not ilsvrc_2012_num_leaf_images_path:
      ilsvrc_2012_num_leaf_images_path = os.path.join(self.records_path,
                                                      'num_leaf_images.json')
    specification = imagenet_specification.create_imagenet_specification(
        learning_spec.Split, self.files_to_skip,
        ilsvrc_2012_num_leaf_images_path)
    split_subgraphs, images_per_class, _, _, _, _ = specification

    # Maps each class id to the name of its class.
    self.class_names = {}

    self.dataset_spec = ds_spec.HierarchicalDatasetSpecification(
        self.name, split_subgraphs, images_per_class, self.class_names,
        self.records_path, '{}.tfrecords')

  def _get_synset_ids(self, split):
    """Returns a list of synset id's of the classes assigned to split."""
    return sorted([
        synset.wn_id for synset in imagenet_specification.get_leaves(
            self.dataset_spec.split_subgraphs[split])
    ])

  def create_dataset_specification_and_records(self):
    """Create Records for the ILSVRC 2012 classes.

    The field that requires modification in this case is only self.class_names.
    """
    # Get a list of synset id's assigned to each split.
    train_synset_ids = self._get_synset_ids(learning_spec.Split.TRAIN)
    valid_synset_ids = self._get_synset_ids(learning_spec.Split.VALID)
    test_synset_ids = self._get_synset_ids(learning_spec.Split.TEST)
    all_synset_ids = train_synset_ids + valid_synset_ids + test_synset_ids

    # It is expected that within self.data_root there is a directory
    # for every ILSVRC 2012 synset, named by that synset's WordNet ID
    # (e.g. n15075141) and containing all images of that synset.
    set_of_directories = set(
        entry for entry in tf.io.gfile.listdir(self.data_root)
        if tf.io.gfile.isdir(os.path.join(self.data_root, entry)))
    assert set_of_directories == set(all_synset_ids), (
        'self.data_root should contain a directory whose name is the WordNet '
        "id of each synset that is a leaf of any split's subgraph.")

    # By construction of all_synset_ids, we are guaranteed to get train synsets
    # before validation synsets, and validation synsets before test synsets.
    # Therefore the assigned class_labels will respect that partial order.
    for class_label, synset_id in enumerate(all_synset_ids):
      self.class_names[class_label] = synset_id
      class_path = os.path.join(self.data_root, synset_id)
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_label))

      # Create and write the tf.Record of the examples of this class.
      # Image files for ImageNet do not necessarily come from a canonical
      # source, so pass 'skip_on_error' to be more resilient and avoid crashes
      write_tfrecord_from_directory(
          class_path,
          class_label,
          class_records_path,
          files_to_skip=self.files_to_skip,
          skip_on_error=True)


class FungiConverter(DatasetConverter):
  """Prepares Fungi as required to integrate it in the benchmark.

  From https://github.com/visipedia/fgvcx_fungi_comp  download:
    -Training and validation images [13GB]
    -Training and validation annotations [2.9MB]
  and untar the files in the directory passed to initializer as data_root.
  """
  NUM_TRAIN_CLASSES = 994
  NUM_VALID_CLASSES = 200
  NUM_TEST_CLASSES = 200

  def create_splits(self):
    """Create splits for Fungi and store them in the default path.

    If no split file is provided, and the default location for Fungi Identity
    splits does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """
    # We ignore the original train and validation splits (the test set cannot be
    # used since it is not labeled).
    with tf.io.gfile.GFile(os.path.join(self.data_root, 'train.json')) as f:
      original_train = json.load(f)
    with tf.io.gfile.GFile(os.path.join(self.data_root, 'val.json')) as f:
      original_val = json.load(f)

    # The categories (classes) for train and validation should be the same.
    assert original_train['categories'] == original_val['categories']
    # Sort by category ID for reproducibility.
    categories = sorted(
        original_train['categories'], key=operator.itemgetter('id'))

    # Assert contiguous range [0:category_number]
    assert ([category['id'] for category in categories
            ] == list(range(len(categories))))

    # Some categories share the same name (see
    # https://github.com/visipedia/fgvcx_fungi_comp/issues/1)
    # so we include the category id in the label.
    labels = [
        '{:04d}.{}'.format(category['id'], category['name'])
        for category in categories
    ]

    train_inds, valid_inds, test_inds = gen_rand_split_inds(
        self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES)
    splits = {
        'train': [labels[i] for i in train_inds],
        'valid': [labels[i] for i in valid_inds],
        'test': [labels[i] for i in test_inds]
    }
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    with tf.io.gfile.GFile(os.path.join(self.data_root, 'train.json')) as f:
      original_train = json.load(f)
    with tf.io.gfile.GFile(os.path.join(self.data_root, 'val.json')) as f:
      original_val = json.load(f)

    image_list = original_train['images'] + original_val['images']
    image_id_dict = {}
    for image in image_list:
      # assert this image_id was not previously added
      assert image['id'] not in image_id_dict
      image_id_dict[image['id']] = image

    # Add a class annotation to every image in image_id_dict.
    annotations = original_train['annotations'] + original_val['annotations']
    for annotation in annotations:
      # assert this images_id was not previously annotated
      assert 'class' not in image_id_dict[annotation['image_id']]
      image_id_dict[annotation['image_id']]['class'] = annotation['category_id']

    # dict where the class is the key.
    class_filepaths = collections.defaultdict(list)
    for image in image_list:
      class_filepaths[image['class']].append(
          os.path.join(self.data_root, image['file_name']))

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    for class_id, class_label in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_label)
      # Extract the "category_id" information from the class label
      category_id = int(class_label[:4])
      # Check that the key is actually in `class_filepaths`, so that an empty
      # list is not accidentally used.
      if category_id not in class_filepaths:
        raise ValueError('class_filepaths does not contain paths to any '
                         'image for category %d. Existing categories are: %s.' %
                         (category_id, class_filepaths.keys()))
      class_paths = class_filepaths[category_id]
      class_records_path = os.path.join(
          self.records_path, self.dataset_spec.file_pattern.format(class_id))
      self.class_names[class_id] = class_label
      self.images_per_class[class_id] = len(class_paths)

      # Create and write the tf.Record of the examples of this class
      write_tfrecord_from_image_files(class_paths, class_id, class_records_path)


class MiniImageNetConverter(DatasetConverter):
  """Prepares MiniImageNet as required to integrate it in the benchmark.

  From https://github.com/renmengye/few-shot-ssl-public download and untar the
  miniImageNet file in the directory passed to init as data_root.
  """
  NUM_TRAIN_CLASSES = 64
  NUM_VALID_CLASSES = 16
  NUM_TEST_CLASSES = 20

  def create_splits(self):
    """Create splits for MiniImageNet and store them in the default path.

    If no split file is provided, and the default location for MiniImageNet
    splits does not contain a split file, splits are created in this function
    according to the Ravi & Larochelle specification and then stored in that
    default location.

    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """
    start_stop = np.cumsum([
        0, self.NUM_TRAIN_CLASSES, self.NUM_VALID_CLASSES, self.NUM_TEST_CLASSES
    ])
    train_inds = list(range(start_stop[0], start_stop[1]))
    valid_inds = list(range(start_stop[1], start_stop[2]))
    test_inds = list(range(start_stop[2], start_stop[3]))
    splits = {'train': train_inds, 'valid': valid_inds, 'test': test_inds}
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[learning_spec.Split.TRAIN] = len(train_classes)
    self.classes_per_split[learning_spec.Split.VALID] = len(valid_classes)
    self.classes_per_split[learning_spec.Split.TEST] = len(test_classes)

    for classes, split in zip([train_classes, valid_classes, test_classes],
                              ['train', 'val', 'test']):
      path = os.path.join(self.data_root,
                          'mini-imagenet-cache-{}.pkl'.format(split))
      with tf.io.gfile.GFile(path, 'rb') as f:
        data = pkl.load(f)
      # We sort class names to make the dataset creation deterministic
      names = sorted(data['class_dict'].keys())
      for class_id, class_name in zip(classes, names):
        logging.info('Creating record class %d', class_id)
        class_records_path = os.path.join(self.records_path,
                                          self.file_pattern.format(class_id))
        self.class_names[class_id] = class_name
        indices = data['class_dict'][class_name]
        self.images_per_class[class_id] = len(indices)

        writer = tf.python_io.TFRecordWriter(class_records_path)
        for image in data['image_data'][indices]:
          img = Image.fromarray(image)
          buf = io.BytesIO()
          img.save(buf, format='JPEG')
          buf.seek(0)
          write_example(buf.getvalue(), class_id, writer)
        writer.close()

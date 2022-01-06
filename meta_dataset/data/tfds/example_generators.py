# coding=utf-8
# Copyright 2022 The Meta-Dataset Authors.
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

"""Meta-Dataset example generators.

The TFDS implementation assumes that the examples are returned in class order,
i.e. all examples of class 0 first, then all examples of class 1, and so on. If
implementing new functions, make sure that the class order property holds.
"""

import collections
import io
import itertools
import json
import os
import re
from typing import Optional

from absl import logging
from etils import epath
import numpy as np
from PIL import ImageOps
import tensorflow_datasets as tfds


def _image_key(image_id, total_num_examples):
  """Returns an image key in the form of a 128-bit integer.

  The gap between keys is chosen as to occupy as much of the space of 128-bit
  numbers as possible, which ensures that the temporary buckets the examples are
  written to are as balanced as possible.

  Args:
    image_id: image ID, in [0, total_num_examples).
    total_num_examples: total number of examples in the data source.

  Returns:
    The image key.
  """
  gap = int(2 ** 128) // total_num_examples
  return image_id * gap


def _load_and_process_image(image_path = None,
                            image_bytes = None,
                            invert_img = False,
                            bbox=None):
  """Loads and processes an image.

  Args:
    image_path: image path. Exactly one of image_path or image_bytes should be
      passed.
    image_bytes: image bytes. Exactly one of image_path or image_bytes should be
      passed.
    invert_img: if True, invert the image.
    bbox: if passed, crop the image using the bounding box.

  Returns:
    Image bytes.
  """
  if (None not in (image_path, image_bytes)) or not (image_path or image_bytes):
    raise ValueError(
        'exactly one of image_path and image_bytes should be passed.')
  if image_path is not None:
    image_bytes = image_path.read_bytes()
  try:
    img = tfds.core.lazy_imports.PIL_Image.open(io.BytesIO(image_bytes))
  except:
    logging.warn('Failed to open image')
    raise

  assert image_bytes is not None
  img_needs_encoding = False

  if img.format != 'JPEG':
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
    # Convert the image into JPEG
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    image_bytes = buf.getvalue()
  return image_bytes


def generate_aircraft_examples(metadata, paths):
  """Generates Aircraft examples."""
  data_path = paths['fgvc-aircraft-2013b'] / 'fgvc-aircraft-2013b/data'

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
  bboxes_path = data_path / 'images_box.txt'
  with bboxes_path.open('r') as f:
    names_to_bboxes = [
        line.split('\n')[0].split(' ') for line in f.readlines()
    ]
  names_to_bboxes = dict(
      (name, tuple(map(int, (xmin, ymin, xmax, ymax))))
      for name, xmin, ymin, xmax, ymax in names_to_bboxes)

  # Retrieve mapping from filename to variant
  variant_trainval_path = data_path / 'images_variant_trainval.txt'
  with variant_trainval_path.open('r') as f:
    names_to_variants = [
        line.split('\n')[0].split(' ', 1) for line in f.readlines()
    ]

  variant_test_path = data_path / 'images_variant_test.txt'
  with variant_test_path.open('r') as f:
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

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for filename in sorted(variants_to_names[class_name]):
      image_path = data_path / f'images/{filename}.jpg'
      bbox = names_to_bboxes[image_path.stem]
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path, bbox=bbox),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_cu_birds_examples(metadata, paths):
  """Generates CUB examples."""
  data_path = paths['CUB_200_2011'] / 'CUB_200_2011/images'

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for image_path in (data_path / class_name).iterdir():
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_dtd_examples(metadata, paths):
  """Generates DTD examples."""
  data_path = paths['dtd-r1.0.1'] / 'dtd/images'

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for image_path in (data_path / class_name).iterdir():
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_fungi_examples(metadata, paths):
  """Generates Fungi examples."""
  original_train = json.loads(
      (paths['train_val_annotations'] / 'train.json').read_text())
  original_val = json.loads(
      (paths['train_val_annotations'] / 'val.json').read_text())

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

  class_filepaths = collections.defaultdict(list)
  for image in image_list:
    class_filepaths[image['class']].append(
        paths['train_val'] / image['file_name'])

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for image_path in class_filepaths[int(class_name.split('.')[0])]:
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path),
          'format': 'JPEG',
          'filename': image['file_name'].split('/')[-1],
          'label': label,
          'class_name': class_name,
      }


def generate_ilsvrc_2012_examples(metadata, paths):
  """Generates ImageNet examples."""
  # Enumerate all files to skip.
  files_to_skip = set()
  for other_dataset in ('Caltech101', 'Caltech256', 'CUBirds'):
    lines = paths[f'{other_dataset}_duplicates'].read_text().splitlines()

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
      files_to_skip.add(file_name)

  name_to_id = {name: id for id, name in enumerate(metadata['class_names'])}

  for archive_name, archive in  tfds.download.iter_archive(
      paths['ILSVRC2012_img_train'], tfds.download.ExtractMethod.TAR_STREAM):

    class_name = archive_name.split('.')[0]
    label = name_to_id[class_name]
    image_ids = list(range(*metadata['class_slices'][label]))

    archive_iterator = tfds.download.iter_archive(
        archive, tfds.download.ExtractMethod.TAR_STREAM)  # pytype: disable=wrong-arg-types  # gen-stub-imports
    filenames_and_images = sorted([
        (filename, extracted_image.read())
        for filename, extracted_image in archive_iterator
        if filename not in files_to_skip
    ])

    assert len(filenames_and_images) == len(image_ids)

    for image_id, (filename, image_bytes) in zip(image_ids,
                                                 filenames_and_images):
      if filename in files_to_skip:
        logging.info('Skipping file %s', filename)
        continue
      yield _image_key(image_id, metadata['total_num_examples']), {
          'image': _load_and_process_image(image_bytes=image_bytes),
          'format': 'JPEG',
          'filename': filename,
          'label': label,
          'class_name': class_name,
      }


def generate_mscoco_examples(metadata, paths, box_scale_ratio=1.2):
  """Generates MSCOCO examples."""
  if box_scale_ratio < 1.0:
    raise ValueError('Box scale ratio must be greater or equal to 1.0.')

  image_dir = paths['train2017'] / 'train2017'
  annotations = json.loads((
      paths['annotations_trainval2017'] /
      'annotations/instances_train2017.json'
  ).read_text())

  class_name_to_category_id = {
      category['name']: category['id']
      for category in annotations['categories']
  }
  coco_id_to_label = {
      class_name_to_category_id[class_name]: label
      for label, class_name in enumerate(metadata['class_names'])
  }

  label_to_annotations = collections.defaultdict(list)
  for annotation in annotations['annotations']:
    category_id = annotation['category_id']
    if category_id in coco_id_to_label:
      label_to_annotations[coco_id_to_label[category_id]].append(annotation)

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for annotation in label_to_annotations[label]:
      image_path = image_dir / f"{annotation['image_id']:012d}.jpg"

      # The bounding box is represented as (x_topleft, y_topleft, width, height)
      bbox = annotation['bbox']
      with image_path.open('rb') as f:
        image = tfds.core.lazy_imports.PIL_Image.open(f)

      # The image shape is [?, ?, 3] and the type is uint8.
      image = image.convert(mode='RGB')
      image_w, image_h = image.size

      x, y, w, h = bbox
      x = x - 0.5 * w * (box_scale_ratio - 1.0)
      y = y - 0.5 * h * (box_scale_ratio - 1.0)
      w = w * box_scale_ratio
      h = h * box_scale_ratio

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
      image = image.crop((xmin, ymin, xmax, ymax))
      crop_width, crop_height = image.size
      if crop_width <= 0 or crop_height <= 0:
        raise ValueError('crops are not valid.')

      buffer = io.BytesIO()
      image.save(buffer, format='JPEG')
      buffer.seek(0)
      image_bytes = buffer.getvalue()

      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': image_bytes,
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_omniglot_examples(metadata, paths):
  """Generates Omniglot examples."""
  alphabet_paths = itertools.chain(
      (paths['images_background'] / 'images_background').iterdir(),
      (paths['images_evaluation'] / 'images_evaluation').iterdir())
  alphabet_paths = {path.stem: path for path in alphabet_paths}

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    match = re.match('(.*)-(character..)', class_name)
    assert match is not None
    character_path = alphabet_paths[match.group(1)] / match.group(2)
    for image_path in character_path.iterdir():
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(
              image_path=image_path, invert_img=True),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_quickdraw_examples(metadata, paths):
  """Generates Quickdraw examples."""
  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    with paths[class_name].open('rb') as f:
      images = np.load(f)

    for i, image in enumerate(images):
      # We make the assumption that the images are square.
      side = int(np.sqrt(image.shape[0]))
      # To load an array as a PIL.Image we must first reshape it to 2D.
      image = tfds.core.lazy_imports.PIL_Image.fromarray(
          image.reshape((side, side))).convert('RGB')
      # Compress to JPEG before writing
      buffer = io.BytesIO()
      image.save(buffer, format='JPEG')
      buffer.seek(0)

      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': buffer,
          'format': 'JPEG',
          'filename': f'{class_name}.npy[{i}]',
          'label': label,
          'class_name': class_name,
      }


def generate_traffic_sign_examples(metadata, paths):
  """Generates VGG Flowers examples."""
  data_path = paths['GTSRB'] / 'GTSRB/Final_Training/Images'

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    image_paths = sorted((data_path / f'{label:05d}').glob('*.ppm'))
    rng = np.random.RandomState(23)
    rng.shuffle(image_paths)
    for image_path in image_paths:
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }


def generate_vgg_flower_examples(metadata, paths):
  """Generates VGG Flowers examples."""
  data_path = paths['102flowers'] / 'jpg'

  with paths['imagelabels'].open('rb') as f:
    class_ids = tfds.core.lazy_imports.scipy.io.loadmat(f)['labels'][0]

  image_paths = collections.defaultdict(list)
  for i, class_id in enumerate(class_ids):
    image_paths[class_id].append(data_path / f'image_{i + 1:05d}.jpg')

  image_ids = itertools.count()

  for label, class_name in enumerate(metadata['class_names']):
    for image_path in image_paths[int(class_name.split('.')[0])]:
      yield _image_key(next(image_ids), metadata['total_num_examples']), {
          'image': _load_and_process_image(image_path=image_path),
          'format': 'JPEG',
          'filename': image_path.name,
          'label': label,
          'class_name': class_name,
      }

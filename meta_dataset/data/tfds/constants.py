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

"""Meta-Dataset-related constant."""

from meta_dataset.data.tfds import example_generators
import tensorflow_datasets as tfds

CITATION = r"""\
@inproceedings{triantafillou2019meta,
  title={{Meta-Dataset}: A Dataset of Datasets for Learning to Learn from Few Examples},
  author={Triantafillou, Eleni and Zhu, Tyler and Dumoulin, Vincent and Lamblin, Pascal and Evci, Utku and Xu, Kelvin and Goroshin, Ross and Gelada, Carles and Swersky, Kevin and Manzagol, Pierre-Antoine and others},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2019}
}
@article{maji2013fine,
  title={Fine-grained visual classification of aircraft},
  author={Maji, Subhransu and Rahtu, Esa and Kannala, Juho and Blaschko, Matthew
          and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:1306.5151},
  year={2013}
}
@article{welinder2010caltech,
  title={Caltech-UCSD birds 200},
  author={Welinder, Peter and Branson, Steve and Mita, Takeshi and Wah,
          Catherine and Schroff, Florian and Belongie, Serge and Perona, Pietro},
  year={2010},
  publisher={California Institute of Technology}
}
@inproceedings{cimpoi2014describing,
  title={Describing textures in the wild},
  author={Cimpoi, Mircea and Maji, Subhransu and Kokkinos, Iasonas and Mohamed,
          Sammy and Vedaldi, Andrea},
  booktitle={CVPR},
  pages={3606--3613},
  year={2014}
}
@misc{fungi,
  author={Brigit Schroeder and Yin Cui},
  title={{FGVCx} Fungi Classification Challenge 2018},
  year={2018},
  howpublished={\url{github.com/visipedia/fgvcx_fungi_comp}},
}
@misc{danishfungal,
  author={Frøslev, T., Heilmann-Clausen, J., Lange, C., Læssøe, T., Petersen,
          J.H., Søchting, U., Jeppesen, T.S., and Vesterholt, J†},
  title={Danish fungal records database},
  year={2016},
  howpublished={\url{www.svampeatlas.dk}}
}
@article{russakovsky2015imagenet,
  title={Imagenet large scale visual recognition challenge},
  author={Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and others},
  journal={International journal of computer vision},
  volume={115},
  number={3},
  pages={211--252},
  year={2015},
}
@inproceedings{lin2014microsoftcc,
  title={Microsoft {COCO}: Common Objects in Context},
  author={Tsung-Yi Lin and M. Maire and Serge J. Belongie and James Hays and P.
          Perona and D. Ramanan and Piotr Doll{\'a}r and C. L. Zitnick},
  booktitle={ECCV},
  year={2014}
}
@article{lake2015human,
  title={Human-level concept learning through probabilistic program induction},
  author={Lake, Brenden M and Salakhutdinov, Ruslan and Tenenbaum, Joshua B},
  journal={Science},
  volume={350},
  number={6266},
  pages={1332--1338},
  year={2015},
}
@misc{jongejan2016quick,
  title={The {Quick}, {Draw}! -- {A.I.} experiment},
  author={Jongejan, Jonas and Rowley, Henry and Kawashima, Takashi and Kim,
          Jongmin and Fox-Gieg, Nick},
  howpublished={\url{quickdraw.withgoogle.com}},
  year={2016}
}
@inproceedings{stallkamp2011german,
  author={Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
  booktitle={IEEE International Joint Conference on Neural Networks},
  title={The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class
         classification competition},
  year={2011},
  pages={1453--1460}
}
@inproceedings{nilsback2008automated,
  title={Automated flower classification over a large number of classes},
  author={Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle={2008 Sixth Indian Conference on Computer Vision, Graphics \& Image
             Processing},
  pages={722--729},
  year={2008},
  organization={IEEE}
}
"""

QUICKDRAW_CLASS_NAMES = (
    'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
    'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe',
    'backpack', 'banana', 'bandage', 'barn', 'baseball bat', 'baseball',
    'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed',
    'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake',
    'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
    'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
    'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator',
    'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
    'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone',
    'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock',
    'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch',
    'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup',
    'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door',
    'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow',
    'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan',
    'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck',
    'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower',
    'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden hose',
    'garden', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar',
    'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog',
    'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse',
    'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass',
    'house plant', 'house', 'hurricane', 'ice cream', 'jacket', 'jail',
    'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern',
    'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning',
    'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map',
    'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
    'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache',
    'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon',
    'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree',
    'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut',
    'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck',
    'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
    'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
    'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
    'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster',
    'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus',
    'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark',
    'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull',
    'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel',
    'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider',
    'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star',
    'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove',
    'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun',
    'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table',
    'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet',
    'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush',
    'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree',
    'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van',
    'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale',
    'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga',
    'zebra', 'zigzag'
)

BUILDER_CONFIGS = [
    dict(name='aircraft',
         dataset_spec_prefixes={'v1': 'aircraft', 'v2': 'aircraft'},
         filenames={
             'fgvc-aircraft-2013b': ('http://www.robots.ox.ac.uk/~vgg/data/'
                                     'fgvc-aircraft/archives/'
                                     'fgvc-aircraft-2013b.tar.gz'),
         },
         manual_filenames={},
         generate_examples_fn=example_generators.generate_aircraft_examples,
         description='The Aircraft data source.'),
    dict(
        name='cu_birds',
        dataset_spec_prefixes={'v1': 'cu_birds', 'v2': 'cu_birds'},
        filenames={
            'CUB_200_2011': tfds.download.Resource(
                url=('https://drive.google.com/uc?export=download&'
                     'id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'),
                extract_method=tfds.download.ExtractMethod.TAR_GZ),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_cu_birds_examples,
        description='The CUB data source.'),
    dict(
        name='dtd',
        dataset_spec_prefixes={'v1': 'dtd', 'v2': 'dtd'},
        filenames={
            'dtd-r1.0.1': ('https://www.robots.ox.ac.uk/~vgg/data/dtd/'
                           'download/dtd-r1.0.1.tar.gz'),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_dtd_examples,
        description='The DTD data source.'),
    dict(
        name='fungi',
        dataset_spec_prefixes={'v1': 'fungi', 'v2': 'fungi'},
        filenames={
            'train_val': ('https://labs.gbif.org/fgvcx/2018/'
                          'fungi_train_val.tgz'),
            'train_val_annotations': ('https://labs.gbif.org/fgvcx/2018/'
                                      'train_val_annotations.tgz'),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_fungi_examples,
        description='The Fungi data source.'),
    dict(
        name='ilsvrc_2012',
        dataset_spec_prefixes={'v1': 'ilsvrc_2012', 'v2': 'ilsvrc_2012_v2'},
        filenames={  # pylint: disable=g-complex-comprehension
            f'{name}_duplicates': (
                'https://raw.githubusercontent.com/google-research/'
                'meta-dataset/main/meta_dataset/dataset_conversion/'
                f'ImageNet_{name}_duplicates.txt')
            for name in ('Caltech101', 'Caltech256', 'CUBirds')
        },
        manual_filenames={
            'ILSVRC2012_img_train': 'ILSVRC2012_img_train.tar',
        },
        generate_examples_fn=example_generators.generate_ilsvrc_2012_examples,
        description='The ILSVRC 2012 data source.'),
    dict(
        name='mscoco',
        dataset_spec_prefixes={'v1': 'mscoco', 'v2': 'mscoco'},
        filenames={
            'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
            'annotations_trainval2017': ('http://images.cocodataset.org/'
                                         'annotations/'
                                         'annotations_trainval2017.zip'),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_mscoco_examples,
        description='The MSCOCO data source.'),
    dict(
        name='omniglot',
        dataset_spec_prefixes={'v1': 'omniglot', 'v2': 'omniglot'},
        filenames={
            'images_background': ('https://github.com/brendenlake/omniglot/'
                                  'raw/master/python/images_background.zip'),
            'images_evaluation': ('https://github.com/brendenlake/omniglot/'
                                  'raw/master/python/images_evaluation.zip'),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_omniglot_examples,
        description='The Omniglot data source.'),
    dict(
        name='quickdraw',
        dataset_spec_prefixes={'v1': 'quickdraw', 'v2': 'quickdraw'},
        filenames={
            name: f'gs://quickdraw_dataset/full/numpy_bitmap/{name}.npy'
            for name in QUICKDRAW_CLASS_NAMES
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_quickdraw_examples,
        description='The QuickDraw data source.'),
    dict(
        name='traffic_sign',
        dataset_spec_prefixes={'v1': 'traffic_sign', 'v2': 'traffic_sign'},
        filenames={
            'GTSRB': ('https://sid.erda.dk/public/archives/'
                      'daaeac0d7ce1152aea9b61d9f1e19370/'
                      'GTSRB_Final_Training_Images.zip'),
        },
        manual_filenames={},
        generate_examples_fn=(
            example_generators.generate_traffic_sign_examples),
        description='The Traffic Sign data source.'),
    dict(
        name='vgg_flower',
        dataset_spec_prefixes={'v1': 'vgg_flower'},
        filenames={
            '102flowers': ('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
                           '102flowers.tgz'),
            'imagelabels': ('http://www.robots.ox.ac.uk/~vgg/data/flowers/'
                            '102/imagelabels.mat'),
        },
        manual_filenames={},
        generate_examples_fn=example_generators.generate_vgg_flower_examples,
        description='The VGG Flowers data source.'),
]

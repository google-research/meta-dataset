# Evaluating on VTAB+MD

This page contains more specific instructions for evaluating on [VTAB+MD](https://arxiv.org/abs/2104.02638).

## Prerequisites

* Clone both [Meta-Dataset](https://github.com/google-research/meta-dataset) and [VTAB](https://github.com/google-research/task_adaptation) repositories.
* Follow VTAB's [installation](https://github.com/google-research/task_adaptation#installation) and [dataset preparation](https://github.com/google-research/task_adaptation#dataset-preparation) instructions.
* Follow Meta-Dataset's [installation](README.md#installation) and [data downloading and conversion](README.md#downloading-and-converting-datasets) instructions, including the MD-v2-specific [instructions](README.md#meta-dataset-v2).

## Loading VTAB-v2 tasks as episodes

VTAB-v2 tasks can be treaded as episodes from the Meta-Dataset codebase's perspective. Since the test sets of VTAB-v2 tasks can sometimes be much larger than Meta-Dataset query set sizes, we will partition each VTAB-v2 task's test set into smaller batches and turn the task into a *sequence* of episodes, each containing the task's entire training set as its support set and one of the partition sets as its query set.

```python
from meta_dataset.data import providers
from meta_dataset.data import read_episodes
import tensorflow as tf

VTAB_v2_NATURAL = [
    "caltech101",
    "cifar(num_classes=100)",
    # While DTD is part of VTAB, it's not part of VTAB-v2 since it conflicts
    # with Meta-Dataset's DTD data source. Flowers102 also conflicts with
    # Meta-Dataset's Flowers102 data source, but the VTAB+MD protocol reserves
    # it for VTAB-v2 rather than MD-v2.
    "oxford_flowers102",
    "oxford_iiit_pet",
    "sun397",
    "svhn"
]
VTAB_v2_SPECIALIZED = [
    "diabetic_retinopathy(config='btgraham-300')",
    "eurosat",
    "resisc45",
    "patch_camelyon",
]
VTAB_v2_STRUCTURED = [
    "clevr(task='closest_object_distance')",
    "clevr(task='count_all')",
    "dmlab",
    "dsprites(predicted_attribute='label_orientation', num_classes=16)",
    "dsprites(predicted_attribute='label_x_position', num_classes=16)",
    "smallnorb(predicted_attribute='label_azimuth')",
    "smallnorb(predicted_attribute='label_elevation')",
    "kitti(task='closest_vehicle_distance')",
]

# `meta_dataset.data.read_episodes.read_vtab_as_episode` expects one of the
# following as its first argument.
VTAB_v2_TASKS = VTAB_v2_NATURAL + VTAB_v2_SPECIALIZED + VTAB_v2_STRUCTURED

for vtab_key in VTAB_v2_TASKS:
  # `n_eval` is the number of episodes that the task is partitioned into.
  # `n_classes` is the task's number of classes.
  support_ds, query_ds, n_eval, n_classes = read_episodes.read_vtab_as_episode(
      vtab_key=vtab_key,
      # VTAB+MD does not prescribe a specific image size, and 224 is used as an
      # example here.
      image_size=224,
      # This defines the largest possible query set size. In other words, the
      # task's test set is partitioned into query sets of size up to
      # `query_size_limit`.
      query_size_limit=128)

  episodes = tf.data.Dataset.zip(
      (support_ds.repeat(), query_ds.repeat())
  ).map(lambda support_data, query_data: providers.Episode(
          support_images=support_data['image'],
          query_images=query_data['image'],
          support_labels=support_data['label'],
          query_labels=query_data['label'],
          support_class_ids=support_data['label'],
          query_class_ids=query_data['label']))

  for episode in episodes:
    # Evaluate on the episode. We assume that the model's behaviour is
    # deterministic across episodes, i.e. training on the same support set
    # should yield the same classifier every time.
    #
    # `episode` is a namedtuple; the support/query images/labels are accessed
    # via its `{support,query}_{image,label}` fields.
    #
    # Accuracies should be aggregated across episodes in order to get the
    # overall accuracy for a given task. Note that the last episode possibly
    # contains fewer query examples, which needs to be taken into account in
    # order for the aggregated accuracy value to be correct.
    pass
```

## Loading MD-v2 episodes as VTAB tasks

MD-v2 episodes of a given data source can be treated as a set of related tasks
from the VTAB codebase's perspective. In order to do so, we first need to cache
MD-v2 episodes on disk.

From the Meta-Dataset repository's root directory, run

```bash
RECORDS=<...>
OUTPUT_DIR=<...>
for DATASET in aircraft cu_birds dtd fungi \
               mscoco omniglot quickdraw traffic_sign; do \
    python -m meta_dataset.data.dump_episodes \
        --dataset_name=${DATASET} \
        --output_dir=${OUTPUT_DIR}/${DATASET} \
        --num_episodes=50 \
        --records_root_dir=${RECORDS} \
        --gin_config=learn/gin/setups/data_config_string.gin \
        --gin_config=learn/gin/setups/variable_way_and_shot.gin \
        --gin_bindings="DataConfig.num_prefetch=64"; \
done
```

Here `RECORDS` is the path to the root directory which contains the converted
datasets, as per Meta-Dataset's instructions.

Existing TF-Hub modules can then be evaluated by running

```bash
for DATASET in aircraft cu_birds dtd fungi \
               mscoco omniglot quickdraw traffic_sign; do
    for EPISODE_ID in {0..49}; do \
        python -m task_adaptation.adapt_and_eval \
            --hub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3  \
            --hub_module_signature image_feature_vector \
            --finetune_layer resnet_v2_50/global_pool \
            --work_dir /tmp/cifar100 \
            --dataset "meta_dataset(dataset='${DATASET}',episode_id=${EPISODE_ID})" \
            --dataset_train_split_name train \
            --batch_size 64 \
            --batch_size_eval 10 \
            --initial_learning_rate 0.01 \
            --decay_steps 300,600,900 \
            --max_steps 1000; \
    done; \
done
```

## Citation

We ask that work citing VTAB+MD also acknowledges the contributors to its underlying data sources. The following sample is provided for convenience:

```
VTAB+MD aggregates data from multiple datasets, namely
CIFAR100~\cite{krizhevsky2009learning}, Caltech101~\cite{fei2004learning},
PatchCamelyon~\cite{veeling2018rotation}, CLEVR~\cite{johnson2017clevr},
DeepMind Lab~\cite{beattie2016deepmind},
EuroSAT~\cite{helber2018introducing,helber2019eurosat},
Flowers102~\cite{nilsback2008automated}, KITTI~\cite{geiger2013vision},
Oxford-IIIT Pet~\cite{parkhi2012cats}, RESISC45~\cite{cheng2017remote},
Diabetic Retinopathy~\cite{kaggle-diabetic-retinopathy},
SVHN~\cite{yuval2011reading}, Sun397~\cite{xiao2010sun,xiao2016sun},
dSprites~\cite{dsprites17}, sNORB~\cite{lecun2004learning},
Omniglot~\cite{lake2015human}, Aircraft~\cite{maji2013fine},
CU Birds~\cite{welinder2010caltech}, DTD~\cite{cimpoi2014describing},
QuickDraw~\cite{jongejan2016quick}, Fungi~\cite{fungi},
Traffic Signs~\cite{stallkamp2011german}, and MSCOCO~\cite{lin2014microsoftcc}.

Fei-Fei Li, Marco Andreetto, and Marc'Aurelio Ranzato collected data for
Caltech101~\cite{fei2004learning}. All data for
PatchCamelyon~\cite{bejnordi2017diagnostic} is released under the CC0 License,
following the license of Camelyon16~\cite{bejnordi2017diagnostic}. All data for
CLEVR~\cite{johnson2017clevr} is released under the Creative Commons CC BY 4.0
license. The DeepMind Lab~\cite{beattie2016deepmind} code is licensed under the
GNU General Public License v2.0. Radhika Desikan, Liz Hodgson, and Kath Steward
provided expert assistance in ground truth labelling the
Flowers102~\cite{nilsback2008automated} data. All KITTI~\cite{geiger2013vision}
data is released under the Creative Commons Attribution-NonCommercial-ShareAlike
3.0 License. All Oxfort-IIIT Pet~\cite{parkhi2012cats} data is released under
the Creative Commons Attribution-ShareAlike 4.0 International License. The
copyright remains with the original owners of the images. Diabetic
Retinopathy~\cite{kaggle-diabetic-retinopathy} images were provided by EyePACS.
All dSprites~\cite{dsprites17} images were generated using the LOVE framework,
which is licensed under zlib/libpng license. All sNORB~\cite{lecun2004learning}
data is provided for research purposes and cannot be sold. All
Omniglot~\cite{lake2015human} data is released under the MIT license.
Aircraft~\cite{maji2013fine} images were made available by Mick Bajcar, Aldo
Bidini, Wim Callaert, Tommy Desmet, Thomas Posch, James Richard Covington, Gerry
Stegmeier, Ben Wang, Darren Wilson, and Konstantin von Wedelstaedt. All images
are available exclusively for non-commercial research purposes. The
QuickDraw~\cite{jongejan2016quick} data is made available by Google, Inc. under
the Creative Commons Attribution 4.0 International license. All
Fungi~\cite{fungi} images are sourced from fungi species submitted in the Danish
Svampe Atlas~\cite{danishfungal} and use of the data is subject to Fungi's terms
of use.\footnote{\scriptsize\url{https://github.com/visipedia/fgvcx_fungi_comp\#terms-of-use}}
Traffic Sign~\cite{stallkamp2011german} benefited from the annotation support of
Lukas Caup, Sebastian Houben, Lukas Kubik, Bastian Petzka, Stefan Tenbült, and
Marc Tschentscher. Use of MSCOCO~\cite{lin2014microsoftcc} is subject to its
terms of use.\footnote{\scriptsize\url{https://cocodataset.org/\#termsofuse}}
```

References:

```
@techreport{krizhevsky2009learning,
  author={Alex Krizhevsky},
  title={Learning multiple layers of features from tiny images},
  institution={University of Toronto},
  year={2009}
}

@inproceedings{fei2004learning,
  title={Learning generative visual models from few training examples: An
         incremental bayesian approach tested on 101 object categories},
  author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
  booktitle={CVPR Workshop},
  year={2004},
}

@inproceedings{veeling2018rotation,
  title={Rotation equivariant {CNNs} for digital pathology},
  author={Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and Cohen,
          Taco and Welling, Max},
  booktitle={International Conference on Medical Image Computing and
             Computer-Assisted Intervention},
  year={2018},
}

@article{bejnordi2017diagnostic,
  title={Diagnostic assessment of deep learning algorithms for detection of
         lymph node metastases in women with breast cancer},
  author={Bejnordi, Babak Ehteshami and Veta, Mitko and Van Diest, Paul Johannes
          and Van Ginneken, Bram and Karssemeijer, Nico and Litjens, Geert and
          Van Der Laak, Jeroen AWM and Hermsen, Meyke and Manson, Quirine F and
          Balkenhol, Maschenka and others},
  journal={Jama},
  volume={318},
  number={22},
  pages={2199--2210},
  year={2017},
  publisher={American Medical Association}
}

@inproceedings{johnson2017clevr,
  title={{CLEVR}: A diagnostic dataset for compositional language and elementary
         visual reasoning},
  author={Johnson, Justin and Hariharan, Bharath and Van Der Maaten, Laurens and
          Fei-Fei, Li and Lawrence Zitnick, C and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}

@article{beattie2016deepmind,
  title={Deepmind Lab},
  author={Beattie, Charles and Leibo, Joel Z and Teplyashin, Denis and Ward, Tom
          and Wainwright, Marcus and K{\"u}ttler, Heinrich and Lefrancq, Andrew
          and Green, Simon and Vald{\'e}s, V{\'\i}ctor and Sadik, Amir and
          others},
  journal={arXiv preprint arXiv:1612.03801},
  year={2016}
}

@article{helber2019eurosat,
  title={{EuroSAT}: A novel dataset and deep learning benchmark for land use and
         land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth,
          Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and
           Remote Sensing},
  year={2019},
}

@inproceedings{helber2018introducing,
  title={Introducing {EuroSAT}: A Novel Dataset and Deep Learning Benchmark for
         Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth,
          Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing
             Symposium},
  year={2018},
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

@article{geiger2013vision,
  title={Vision meets robotics: The {KITTI} dataset},
  author={Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun,
          Raquel},
  journal={The International Journal of Robotics Research},
  volume={32},
  number={11},
  pages={1231--1237},
  year={2013},
}

@inproceedings{parkhi2012cats,
  title={Cats and dogs},
  author={Parkhi, Omkar M and Vedaldi, Andrea and Zisserman, Andrew and Jawahar,
          CV},
  booktitle={CVPR},
  year={2012},
}

@article{cheng2017remote,
  title={Remote sensing image scene classification: Benchmark and state of the
         art},
  author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
  journal={Proceedings of the IEEE},
  volume={105},
  number={10},
  pages={1865--1883},
  year={2017},
  publisher={IEEE}
}

@misc{kaggle-diabetic-retinopathy,
  author={Kaggle and EyePacs},
  title={Kaggle Diabetic Retinopathy Detection},
  month={July},
  year={2015},
  url={https://www.kaggle.com/c/diabetic-retinopathy-detection/data}
}

@inproceedings{yuval2011reading,
  title={Reading Digits in Natural Images with Unsupervised Feature Learning},
  author={Yuval Netzer and Tao Wang and Adam Coates and Alessandro Bissacco and
          Bo Wu and Andrew Y. Ng},
  booktitle={NeurIPS Workshop on Deep Learning and Unsupervised Feature Learning}
  year={2011},
}

@inproceedings{xiao2010sun,
  title={Sun database: Large-scale scene recognition from abbey to zoo},
  author={Xiao, Jianxiong and Hays, James and Ehinger, Krista A and Oliva, Aude
          and Torralba, Antonio},
  booktitle={CVPR},
  pages={3485--3492},
  year={2010},
}

@article{xiao2016sun,
  title={Sun database: Exploring a large collection of scene categories},
  author={Xiao, Jianxiong and Ehinger, Krista A and Hays, James and Torralba,
          Antonio and Oliva, Aude},
  journal={International Journal of Computer Vision},
  volume={119},
  number={1},
  pages={3--22},
  year={2016},
}

@misc{dsprites17,
  author={Loic Matthey and Irina Higgins and Demis Hassabis and Alexander
          Lerchner},
  title={dSprites: Disentanglement testing Sprites dataset},
  howpublished={https://github.com/deepmind/dsprites-dataset/},
  year={2017},
}

@inproceedings{lecun2004learning,
  title={Learning methods for generic object recognition with invariance to pose
         and lighting},
  author={LeCun, Yann and Huang, Fu Jie and Bottou, Leon},
  booktitle={CVPR},
  year={2004},
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

@misc{jongejan2016quick,
  title={The {Quick}, {Draw}! -- {A.I.} experiment},
  author={Jongejan, Jonas and Rowley, Henry and Kawashima, Takashi and Kim,
          Jongmin and Fox-Gieg, Nick},
  howpublished={\url{quickdraw.withgoogle.com}},
  year={2016}
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

@inproceedings{stallkamp2011german,
  author={Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
  booktitle={IEEE International Joint Conference on Neural Networks},
  title={The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class
         classification competition},
  year={2011},
  pages={1453--1460}
}

@inproceedings{lin2014microsoftcc,
  title={Microsoft {COCO}: Common Objects in Context},
  author={Tsung-Yi Lin and M. Maire and Serge J. Belongie and James Hays and P.
          Perona and D. Ramanan and Piotr Doll{\'a}r and C. L. Zitnick},
  booktitle={ECCV},
  year={2014}
}
```

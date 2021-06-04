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

In order to run the experiments described in the first version of the arXiv
article, [arxiv.org/abs/1903.03096v1](https://arxiv.org/abs/1903.03096v1),
please use the instructions, code, and configuration files at version
[arxiv_v1](https://github.com/google-research/meta-dataset/tree/arxiv_v1) of
this repository.

We are currently working on updating the instructions, code, and configuration
files to reproduce the results in the second version of the article,
[arxiv.org/abs/1903.03096v2](https://arxiv.org/abs/1903.03096v2). You can follow
the progess in branch
[arxiv_v2_dev](https://github.com/google-research/meta-dataset/tree/arxiv_v2_dev)
of this repository.

# Reproducing best results

This section shows how to launch the training experiments with the
hyperparameters that the search in the paper determined to be best. These values
are provided in the `.gin` files in `meta_dataset/learn/gin/best/`.

We assumes you have already converted all datasets according to
[these instructions](dataset_conversion.md), and that the following environment
variables are defined:

-   `$RECORDS`: root directory that contain the converted datasets.
-   `$EXPROOT`: root directory for the output of experiments. It will contain
    two subdirectories, `summaries/` and `checkpoints/`, each of which will have
    a subdirectory for each experiment.

If any given job gets interrupted, launching the same command again (with the
same checkpoint directory) should restart from the last checkpoint (every 500
training episodes by default).

Summaries can be plotted using
[TensorBoard](https://www.tensorflow.org/tensorboard/) to visualize the
training.

Groups of experiments are expressed here with shell loops and variables, but
they could be dispatched to run in parallel with any infrastructure you have
access to.

Time reported are approximate, and were measured on GCP instances with one P100
GPU, 16vCPU cores (although the CPU usage was far from full), 155 GB memory, and
local SSD storage for records, summaries, and checkpoints.

## Baseline and pre-training on ImageNet

Some of the best meta-learning models are initialized from the weights of a
batch baseline (trained on ImageNet). For this reason, we will start with
training the baseline with several backbones (not only the best one). Since not
all backbone variants are needed for the best models, we will only need to train
3 of them.

```bash
export EXPNAME=baseline_imagenet
for BACKBONE in resnet mamlconvnet mamlresnet
do
  export JOBNAME=${EXPNAME}_${BACKBONE}
  python -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${JOBNAME} \
    --summary_dir=${EXPROOT}/summaries/${JOBNAME} \
    --gin_config=meta_dataset/learn/gin/best/${JOBNAME}.gin \
    --gin_bindings="Trainer.experiment_name='$EXPNAME'"
done
```

Each of the jobs took between 12 and 18 hours to reach 75k steps (episodes).

## Training on ImageNet

### k-NN

The `baseline` ("k-NN") model does not have to be trained again, the `resnet`
variant performed the best. For consistency, we can simply add symbolic links:

```bash
ln -s ${EXPROOT}/checkpoints/baseline_imagenet_resnet ${EXPROOT}/checkpoints/baseline_imagenet
ln -s ${EXPROOT}/summaries/baseline_imagenet_resnet ${EXPROOT}/summaries/baseline_imagenet
```

### Finetune, ProtoNet

The best models for `baselinefinetune` ("Finetune") and `prototypical`
("ProtoNet") on ILSVRC-2012 were not initialized from pre-trained model, so
their respective Gin configuration indicates:

-   `Trainer.pretrained_source = 'scratch'`, and
-   `Trainer.checkpoint_to_restore = ''`

They can be launched right away (in parallel with the pre-training), and their
configuration does not need to be changed.

### Other models

For the other models, the respective best pre-train model is:

-   `matching` ("MatchingNet"): `resnet`
-   `maml` ("fo-MAML"): `mamlconvnet` (`four_layer_convnet_maml`)
-   `maml_init_with_proto` ("Proto-MAML"): `mamlresnet` (`resnet_maml`)

The corresponding `.gin` file indicates `Trainer.pretrained_source =
'imagenet'`, and has a placeholder for `Trainer.checkpoint_to_restore`. The
number of steps for the best checkpoint did not make a measurable difference in
our experience, so you can simply update the base path and keep the number in
"`model_?????.ckpt`". If you would like to perform the selection for the best
number of steps, see [Get the best checkpoint](#get-the-best-checkpoint) section
below, and update the Gin configuration files accordingly.

### Command line

```bash
export SOURCE=imagenet
for MODEL in baselinefinetune prototypical matching maml maml_init_with_proto
do
  export EXPNAME=${MODEL}_${SOURCE}
  python -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
    --gin_bindings="Trainer.experiment_name='$EXPNAME'"
done
```

Note: rather than editing the `.gin` file, it is also possible to specify the
path to the pretrained checkpoint to load on the command-line, adding
`--gin_bindings="Trainer.checkpoint_to_restore='...'"`.

Run time:

-   `baselinefinetune`: ~10 hours
-   `matching` and `prototypical`: ~20 hours
-   `maml`: 2 days
-   `maml_init_with_proto`: manually killed after 4 days, reached only 18k
    updates.

## Evaluation

### Get the best checkpoint

Actual early stopping is not performed, in that we do not actually stop training
early according to validation performance. Instead, checkpoints are recorded
every 500 updates during training, and validation error is saved at these times.

To recover the number of steps of the best performing checkpoints, and write
them to `.txt` (as well as `.pklz`) files named `best_...` inside `$EXPROOT/`:

```bash
export SOURCE=imagenet
for MODEL in baseline baselinefinetune prototypical matching maml maml_init_with_proto
do
  export EXPNAME=${MODEL}_${SOURCE}
  python -m meta_dataset.analysis.select_best_model \
    --all_experiments_root=$EXPROOT \
    --experiment_dir_basenames='' \
    --restrict_to_variants=${EXPNAME} \
    --description=best_${EXPNAME}
done
```

The runtime should be a few seconds per model, less than one minute in total.

In order to perform that analysis on the different baseline models used for
pre-training, the main difference is that `$EXPNAME` should be set to
`baseline_imagenet_${BACKBONE}` when iterating.

### Evaluate performance on all datasets

The `meta_dataset.train` script is called in evaluation mode to evaluate a given
checkpoint on (the meta-test classes of) a given dataset. Depending on the
model, each evaluation can take from a few minutes (`baseline`) to about one
hour (`maml_init_with_proto`).

```bash
export SOURCE=imagenet
for MODEL in baseline baselinefinetune matching prototypical maml maml_init_with_proto
do
  export EXPNAME=${MODEL}_${SOURCE}
  # set BESTNUM to the "best_update_num" field in the corresponding best_....txt
  export BESTNUM=$(grep best_update_num ${EXPROOT}/best_${EXPNAME}.txt | awk '{print $2;}')
  for DATASET in ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco
  do
    python -m meta_dataset.train \
      --is_training=False \
      --records_root_dir=$RECORDS \
      --summary_dir=${EXPROOT}/summaries/${EXPNAME}_eval_$DATASET \
      --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
      --gin_bindings="Trainer.experiment_name='${EXPNAME}'" \
      --gin_bindings="Trainer.checkpoint_to_restore='${EXPROOT}/checkpoints/${EXPNAME}/model_${BESTNUM}.ckpt'" \
      --gin_bindings="benchmark.eval_datasets='$DATASET'"
  done
done
```

Summaries of the evaluation job are generated, which contain much finer-grained
information that the mean and confidence interval that are output by the script
itself. Total time was about 12 hours, but it could be parallelized.

### Other metrics and analyses

TODO: ways, shots, fine-grainedness analyses.

## Training and evaluating on all datasets

The setup is quite similar to training on ILSVRC-2012 only, the command lines
can be easily re-used by setting `$SOURCE` to `all`, with the following
differences:

-   The pre-trained checkpoints are still the ones trained on ImageNet only, so
    we are not retraining the baseline with the different backbones.
-   The best baseline still has to be trained on all the data.
-   The best `prototypical` was a pre-trained `resnet`, so the Gin configuration
    also has to be updated for `prototypical_all`.

Since many more `.tfrecords` files are used than for ImageNet only, make sure
that the limits on the number of files open at the same time is large enough
(`ulimit -n`). We used
[these instructions](http://posidev.com/blog/2009/06/04/set-ulimit-parameters-on-ubuntu/)
to set the limit to 100000, although 10000 is probably sufficient (1024 was too
small). If the limit it too low, the script would crash:

-   For batch training, right away with an explicit error:
    `ResourceExhaustedError [...] Too many open files`,
-   For episodic training, with a more cryptic error like `InvalidArgumentError:
    Feature: image (data type: string) is required but could not be found.`,
    after warnings like `DirectedInterleave selected an exhausted input`.

### Training

```bash
export SOURCE=all
for MODEL in baseline baselinefinetune prototypical matching maml maml_init_with_proto
do
  export EXPNAME=${MODEL}_${SOURCE}
  python -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${EXPNAME} \
    --summary_dir=${EXPROOT}/summaries/${EXPNAME} \
    --gin_config=meta_dataset/learn/gin/best/${EXPNAME}.gin \
    --gin_bindings="Trainer.experiment_name='$EXPNAME'"
done
```

Run time:

-   `baseline` and `baselinefinetune`: ~10 hours
-   `matching` and `prototypical`: ~18 hours
-   `maml`: 2 days
-   `maml_init_with_proto`: manually killed after 4 days, reached only 20k
    updates.

### Evaluation

Getting the best checkpoint and evaluating it would be identical, except for
`export SOURCE=all`. Timing should be similar as well.

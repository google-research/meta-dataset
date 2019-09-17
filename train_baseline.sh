#!/bin/bash
ulimit -n 100000
export EXPNAME=baseline_imagenet
#for BACKBONE in resnet mamlconvnet mamlresnet
for BACKBONE in resnet
do
  export JOBNAME=${EXPNAME}_${BACKBONE}
  python2 -m meta_dataset.train \
    --records_root_dir=$RECORDS \
    --train_checkpoint_dir=${EXPROOT}/checkpoints/${JOBNAME} \
    --summary_dir=${EXPROOT}/summaries/${JOBNAME} \
    --gin_config=meta_dataset/learn/gin/best/${JOBNAME}.gin \
    --gin_bindings="LearnerConfig.experiment_name='$EXPNAME'"
done

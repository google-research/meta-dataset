# Dataset download and conversion

This file contains instructions to download the individual datasets used by
Meta-Dataset, and convert them into a common format (one TFRecord file per
class). See [an overview](../README.md#downloading-and-converting-datasets) for
more context.

## ilsvrc_2012

1.  Download `ilsvrc2012_img_train.tar`, from the
    [ILSVRC2012 website](http://www.image-net.org/challenges/LSVRC/2012/index)
2.  Extract it into `ILSVRC2012_img_train/`, which should contain 1000 files,
    named `n????????.tar` (expected time: \~30 minutes)
3.  Extract each of `ILSVRC2012_img_train/n????????.tar` in its own directory
    (expected time: \~30 minutes), for instance:

    ```bash
    for FILE in *.tar;
    do
      mkdir ${FILE/.tar/};
      cd ${FILE/.tar/};
      tar xvf ../$FILE;
      cd ..;
    done
    ```
4.  Download the following two files into `ILSVRC2012_img_train/`:
    -   http://www.image-net.org/archive/wordnet.is_a.txt
    -   http://www.image-net.org/archive/words.txt
5.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=ilsvrc_2012 \
      --ilsvrc_2012_data_root=$DATASRC/ILSVRC2012_img_train \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
6.  Expect the conversion to take 4 to 12 hours, depending on the filesystem's
    latency and bandwidth.
7.  Find the following outputs in `$RECORDS/ilsvrc_2012/`:
    -   1000 tfrecords files named `[0-999].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))
    -   `num_leaf_images.json`

## omniglot

1.  Download
    [`images_background.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip)
    and
    [`images_evaluation.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip)
2.  Extract them into the same `omniglot/` directory
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=omniglot \
      --omniglot_data_root=$DATASRC/omniglot \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take a few seconds.
5.  Find the following outputs in `$RECORDS/omniglot/`:
    -   1623 tfrecords files named `[0-1622].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## aircraft

1.  Download
    [`fgvc-aircraft-2013b.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
2.  Extract it into `fgvc-aircraft-2013b`
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=aircraft \
      --aircraft_data_root=$DATASRC/fgvc-aircraft-2013b \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take 5 to 10 minutes.
5.  Find the following outputs in `$RECORDS/aircraft/`:
    -   100 tfrecords files named `[0-99].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## cu_birds

1.  Download
    [`CUB_200_2011.tgz`](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
2.  Extract it into `CUB_200_2011/` (and `attributes.txt`)
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=cu_birds \
      --cu_birds_data_root=$DATASRC/CUB_200_2011 \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take around one minute.
5.  Find the following outputs in `$RECORDS/cu_birds/`:
    -   200 tfrecords files named `[0-199].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## dtd

1.  Download
    [`dtd-r1.0.1.tar.gz`](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
2.  Extract it into `dtd/`
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=dtd \
      --dtd_data_root=$DATASRC/dtd \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take a few seconds.
5.  Find the following outputs in `$RECORDS/dtd/`:
    -   47 tfrecords files named `[0-46].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## quickdraw

1.  Download all 345 `.npy` files hosted on
    [Google Cloud](https://console.cloud.google.com/storage/quickdraw_dataset/full/numpy_bitmap)
    -   You can use
        [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install#install)
        to download them to `quickdraw/`:

        ```bash
        gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw
        ```
2.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=quickdraw \
      --quickdraw_data_root=$DATASRC/quickdraw \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
3.  Expect the conversion to take 3 to 4 hours.
4.  Find the following outputs in `$RECORDS/quickdraw/`:
    -   345 tfrecords files named `[0-344].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## fungi

1.  Download
    [`fungi_train_val.tgz`](https://data.deic.dk/public.php?service=files&t=2fd47962a38e2a70570f3be027cea57f&download)
    and
    [`train_val_annotations.tgz`](https://data.deic.dk/public.php?service=files&t=8dc110f312677d2b53003de983b3a26e&download)
2.  Extract them into the same `fungi/` directory. It should contain one
    `images/` directory, as well as `train.json` and `val.json`.
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=fungi \
      --fungi_data_root=$DATASRC/fungi \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take 5 to 15 minutes.
4.  Find the following outputs in `$RECORDS/fungi/`:
    -   1394 tfrecords files named `[0-1393].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## vgg_flower

1.  Download
    [`102flowers.tgz`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
    and
    [`imagelabels.mat`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
2.  Extract `102flowers.tgz`, it will create a `jpg/` sub-directory
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=vgg_flower \
      --vgg_flower_data_root=$DATASRC/vgg_flower \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take about one minute.
5.  Find the following outputs in `$RECORDS/vgg_flower/`:
    -   102 tfrecords files named `[0-101].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## traffic_sign

1.  Download
    [`GTSRB_Final_Training_Images.zip`](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)
    If the link happens to be broken, browse the GTSRB dataset [website](http://benchmark.ini.rub.de) for more information.
2.  Extract it in `$DATASRC`, it will create a `GTSRB/` sub-directory
3.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=traffic_sign \
      --traffic_sign_data_root=$DATASRC/GTSRB \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
4.  Expect the conversion to take about one minute.
4.  Find the following outputs in `$RECORDS/traffic_sign/`:
    -   43 tfrecords files named `[0-42].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## mscoco

1.  Download the 2017 train images and annotations from http://cocodataset.org/:
    -   You can use
        [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install#install)
        to download them to `mscoco/`:

        ```bash
        cd $DATASRC/mscoco/ mkdir train2017
        gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
        gsutil -m cp gs://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip annotations_trainval2017.zip
        ```
    -   Otherwise, you can download
        [`train2017.zip`](http://images.cocodataset.org/zips/train2017.zip) and
        [`annotations_trainval2017.zip`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
        and extract them into `mscoco/`.
2.  Launch the conversion script:

    ```bash
    python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=mscoco \
      --mscoco_data_root=$DATASRC/mscoco \
      --splits_root=$SPLITS \
      --records_root=$RECORDS
    ```
3.  Expect the conversion to take about 4 hours.
4.  Find the following outputs in `$RECORDS/mscoco/`:
    -   80 tfrecords files named `[0-79].tfrecords`
    -   `dataset_spec.json` (see [note 1](#notes))

## Notes

1. A [reference version](
https://github.com/google-research/meta-dataset/tree/master/meta_dataset/dataset_conversion/dataset_specs)
of each of the `dataset_spec.json` files is part of this repository. You can
compare them with the version generated by the conversion process for
troubleshooting.

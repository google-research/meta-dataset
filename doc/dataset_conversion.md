# Dataset download and conversion

## ilsvrc_2012

- Download `ilsvrc2012_img_train.tar`, from the [ILSVRC2012 website](http://www.image-net.org/challenges/LSVRC/2012/index)
- Extract it into `ILSVRC2012_img_train/`, which should contain 1000 files, named `n????????.tar` (expected time: \~30 minutes)
- Extract each of `ILSVRC2012_img_train/n????????.tar` in its own directory (expected time: \~30 minutes), for instance:
  ```bash
  for FILE in *.tar; do mkdir ${FILE/.tar/}; cd ${FILE/.tar/}; tar xvf ../$FILE; cd ..; done
  ```
- Download the following two files into `ILSVRC2012_img_train/`:
  - http://www.image-net.org/archive/wordnet.is_a.txt
  - http://www.image-net.org/archive/words.txt
- The conversion itself should take 4 to 12 hours, depending on the filesystem's latency and bandwidth:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=ilsvrc_2012 \
    --ilsvrc_2012_data_root=$DATASRC/ILSVRC2012_img_train \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/ilsvrc_2012/`:
  - 1000 tfrecords files named `[0-999].tfrecords`
  - `dataset_spec.pkl`
  - `num_leaf_images.pkl`

## omniglot

- Download [`images_background.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip) and [`images_evaluation.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip)
- Extract them into the same `omniglot/` directory
- The conversion should take a few seconds:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=omniglot \
    --omniglot_data_root=$DATASRC/omniglot \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/omniglot/`:
  - 1623 tfrecords files named `[0-1622].tfrecords`
  - `dataset_spec.pkl`

## aircraft

- Download [`fgvc-aircraft-2013b.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
- Extract it into `fgvc-aircraft-2013b`
- The conversion itself should take 5 to 10 minutes:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=aircraft \
    --aircraft_data_root=$DATASRC/fgvc-aircraft-2013b \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/aircraft/`:
  - 100 tfrecords files named `[0-99].tfrecords`
  - `dataset_spec.pkl`

## cu_birds

- Download [`CUB_200_2011.tgz`](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
- Extract it into `CUB_200_2011/` (and `attributes.txt`)
- The conversion itself should take around one minute:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=cu_birds \
    --cu_birds_data_root=$DATASRC/CUB_200_2011 \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/cu_birds/`:
  - 200 tfrecords files named `[0-199].tfrecords`
  - `dataset_spec.pkl`

## dtd

- Download [`dtd-r1.0.1.tar.gz`](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
- Extract it into `dtd/`
- The conversion itself should take a few seconds:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=dtd \
    --dtd_data_root=$DATASRC/dtd \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/dtd/`:
  - 47 tfrecords files named `[0-46].tfrecords`
  - `dataset_spec.pkl`

## quickdraw

- Download all 345 `.npy` files hosted on [Google Cloud](https://console.cloud.google.com/storage/quickdraw_dataset/full/numpy_bitmap)
  - You can use [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install#install) to download them to `quickdraw/`:
    ```bash
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw
    ```
- The conversion itself should take 3 to 4 hours:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=quickdraw \
    --quickdraw=$DATASRC/quickdraw \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/quickdraw/`:
  - 345 tfrecords files named `[0-344].tfrecords`
  - `dataset_spec.pkl`

## fungi

- Download [`fungi_train_val.tgz`](https://data.deic.dk/public.php?service=files&t=2fd47962a38e2a70570f3be027cea57f&download)
  and [`train_val_annotations.tgz`](https://data.deic.dk/public.php?service=files&t=8dc110f312677d2b53003de983b3a26e&download)
- Extract them into the same `fungi/` directory. It should contain one
  `images/` directory, as well as `train.json` and `val.json`.
- The conversion should take 5 to 15 minutes:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=fungi \
    --fungi_data_root=$DATASRC/fungi \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/fungi/`:
  - 1394 tfrecords files named `[0-1393].tfrecords`
  - `dataset_spec.pkl`

## vgg_flower

- Download [`102flowers.tgz`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
  and [`imagelabels.mat`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
- Extract `102flowers.tgz`, it will create a `jpg/` sub-directory
- The conversion should take about one minute:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=vgg_flower \
    --fungi_data_root=$DATASRC/vgg_flower \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/vgg_flower/`:
  - 102 tfrecords files named `[0-101].tfrecords`
  - `dataset_spec.pkl`

## traffic_sign

- Download [`GTSRB_Final_Training_Images.zip`](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
- Extract it in `$DATASRC`, it will create a `GTSRB/` sub-directory
- The conversion should take about one minute:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=traffic_sign \
    --fungi_data_root=$DATASRC/GTSRB \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/traffic_sign/`:
  - 43 tfrecords files named `[0-42].tfrecords`
  - `dataset_spec.pkl`

## mscoco

- Download the 2017 train images and annotations from http://cocodataset.org/:
  - You can use [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install#install) to download them to `mscoco/`:
    ```bash
    cd $DATASRC/mscoco/
    mkdir train2017
    gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
    gsutil -m cp gs://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    ```
  - Otherwise, you can download [`train2017.zip`](http://images.cocodataset.org/zips/train2017.zip) and [`annotations_trainval2017.zip`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and extract them into `mscoco/`.
- The conversion should take about 4 hours:
  ```bash
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=mscoco \
    --mscoco_data_root=$DATASRC/mscoco \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
  ```
- Expected outputs in `$RECORDS/mscoco/`:
  - 80 tfrecords files named `[0-79].tfrecords`
  - `dataset_spec.pkl`

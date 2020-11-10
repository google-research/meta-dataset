# USAGE:
# This script must be run from the root directory of this repository.
# The correct python environment must be active, e.g.:
#
# ```bash
# python3 -m virtualenv py37
# source py37/bin/activate
# pip install -r requirements.txt
# ```
#
# Then, start the processing with:
# `bash prepare_all_datasets.sh`
#
# NOTES:
# - In total, the disk space consumed by this script is:
#   539 GB (./datasets) + 204 GB (./records) = 743 GB.
# - In total, this script produces about 1.69 million files.
# - The total runtime is on the order of 8 hours (varies widely).
# - Each successfully processed dataset will have a flag
#   file in $FLAGS/. To reprocess, delete this flag file and re-run.
#
# CAVEATS:
# - ILSVRC2012 initial tar file must be downloaded separately (see below).
# - Fungi dataset requires `gsutil` to be installed.
# - If a dataset fails processing partway through, it may be necessary
#   to delete that dataset in the folders listed here before retrying.
#
# TODO:
# - Launching each dataset's processing in separate shells might 
#   improve overall runtime (or might not, depending on resource conflicts).
# - Could add some `trap` error handling for cleaning up partially processed folders
# - Could continue with script if one dataset fails

set -exo pipefail
# Setup folders
rootdir=$(pwd)
FLAGS=${rootdir}/.${0/.sh/}
DATASRC=${rootdir}/datasets
SPLITS=${rootdir}/splits
RECORDS=${rootdir}/records
LOGS=${FLAGS}/logs

mkdir -p $FLAGS $DATASRC $SPLITS $RECORDS $LOGS

logfile=${LOGS}/log.$(date "+%F.%H-%M-%S").txt
exec &> >(tee -a $logfile)

###############################################################################
# Helper Functions
###############################################################################

run_python () {
  # $1 is the dataset name (like "traffic_sign")
  # $2 is the folder name (like "GTSRB")
  [[ $1 ]] || { echo "Missing dataset name!" >&2; return 1; }
  [[ $2 ]] || { echo "Missing folder name!" >&2; return 1; }
  python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
    --dataset=$1 \
    --${1}_data_root=$DATASRC/$2 \
    --splits_root=$SPLITS \
    --records_root=$RECORDS
}

process_dataset () {
  # $1 is dataset name (like "traffic_sign")
  # $2 is sub directory name (like "GTSRB")
  # $3 is remote filename 1
  # $4 is remote filename 2 (may or may not be present)
  [[ $# -ge 3 ]] || { echo "Missing arguments!" >&2; return 1; }

  dataset_name=$1
  folder_name=$2
  remote_filename_1=$3
  if [[ $# -eq 4 ]]; then
    remote_filename_2=$4
  fi
  # Make folder for this dataset
  path=$DATASRC/$folder_name
  mkdir -p $path

  # Download the first file
  filename=$(basename $remote_filename_1)
  [[ -z `find $path -type f -iname $filename` ]] && \
    wget -P $path "$remote_filename_1"

  if [[ $remote_filename_1 == *zip ]]; then
    unzip $path/$filename -d $path
    if [[ $dataset_name == "traffic_sign" ]]; then
      mv $DATASRC/$folder_name/$folder_name/* $DATASRC/$folder_name
      rmdir $DATASRC/$folder_name/$folder_name
    fi
  else # tarball
    if [[ $dataset_name == "vgg_flower" ]] || \
       [[ $dataset_name == "ilsvrc_2012" ]] || \
       [[ $dataset_name == "fungi" ]]; then
      tar -xvf "$path/$filename" -C $path
    else
      tar --strip-components=1 -xvf "$path/$filename" -C $path
    fi
  fi

  # Download the second file (if specified)
  if [[ $# -eq 4 ]]; then
    filename=$(basename $remote_filename_2)
    [[ -z `find $path -type f -iname $filename` ]] && \
      wget -P $path "$remote_filename_2"

    if [[ $dataset_name != "vgg_flower" ]]; then
      if [[ $remote_filename_2 == *zip ]]; then
        unzip $path/$filename -d $path
        if [[ $dataset_name == "mscoco" ]]; then
          find $DATASRC/$folder_name/annotations/ -type f | xargs -n100 mv -t $DATASRC/$folder_name
          rmdir $DATASRC/$folder_name/annotations
        fi
      else # tarball
        if [[ $dataset_name == "fungi" ]]; then
          tar -xvf "$path/$filename" -C $path
        else
          tar --strip-components=1 -xvf "$path/$filename" -C $path
        fi
      fi
    fi
  fi

  # Process into records
  run_python $dataset_name $folder_name
}

process_ilsvrc () {
  # $1 is outer folder name
  # $2 is inner folder name
  # $3 is file name
  [[ -z `find $1 -type f -name $3` ]] && { echo "Missing $3!" >&2; exit 1; }
  # TODO - uncomment these lines
  mkdir -p $1/$2
  tar -xvf "$1/$3" -C $1/$2
  for FILE in $1/$2/n*.tar;
  do
    folder=${FILE/.tar/}
    mkdir -p $folder
    tar -xvf $FILE -C $folder
  done
  echo
}


###############################################################################
# Datasets
###############################################################################

# ilsvrc_2012 (ImageNet)
# NOTE - must already have source tar file at $DATASRC/ILSVRC2012_img_train.tar
done_flag=${FLAGS}/ilsvrc_2012_done.flag
dataset_name=ilsvrc_2012
folder_name=ILSVRC2012_img_train
file_name=ILSVRC2012_img_train.tar
if [[ ! -f ${done_flag} ]]; then
  ls $DATASRC/$file_name || { echo "Missing $file_name!" >&2; exit 1; }
  process_ilsvrc $DATASRC $folder_name $file_name
  wget -P $DATASRC/$folder_name http://www.image-net.org/archive/wordnet.is_a.txt
  wget -P $DATASRC/$folder_name http://www.image-net.org/archive/words.txt
  run_python $dataset_name $folder_name
  touch ${done_flag}
fi

# omniglot
done_flag=${FLAGS}/omniglot_done.flag
dataset_name=omniglot
folder_name=omniglot
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip \
    https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
  touch ${done_flag}
fi

# aircraft
done_flag=${FLAGS}/aircraft_done.flag
dataset_name=aircraft
folder_name=fgvc-aircraft-2013b
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
  touch ${done_flag}
fi

# cu_birds
done_flag=${FLAGS}/cu_birds_done.flag
dataset_name=cu_birds
folder_name=CUB_200_2011
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
  touch ${done_flag}
fi

# dtd
done_flag=${FLAGS}/dtd_done.flag
dataset_name=dtd
folder_name=dtd
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
  touch ${done_flag}
fi

# quickdraw
done_flag=${FLAGS}/quickdraw_done.flag
if [[ ! -f ${done_flag} ]]; then
  mkdir -p $DATASRC/quickdraw
  command -v gsutil || { echo "Need to install gsutil!" >&2; exit 1; }
  gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw
  run_python quickdraw quickdraw
  touch ${done_flag}
fi

# fungi
done_flag=${FLAGS}/fungi_done.flag
dataset_name=fungi
folder_name=fungi
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    "https://data.deic.dk/public.php?service=files&t=2fd47962a38e2a70570f3be027cea57f&download" \
    "https://data.deic.dk/public.php?service=files&t=8dc110f312677d2b53003de983b3a26e&download"
  touch ${done_flag}
fi

# vgg_flower
done_flag=${FLAGS}/vgg_flower_done.flag
dataset_name=vgg_flower
folder_name=vgg_flower
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz \
    http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
  touch ${done_flag}
fi

# traffic_sign
done_flag=${FLAGS}/traffic_sign_done.flag
dataset_name=traffic_sign
folder_name=GTSRB
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
  touch ${done_flag}
fi

# mscoco
done_flag=${FLAGS}/mscoco_done.flag
dataset_name=mscoco
folder_name=mscoco
if [[ ! -f ${done_flag} ]]; then
  process_dataset $dataset_name $folder_name \
    http://images.cocodataset.org/zips/train2017.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  touch ${done_flag}
fi

# "One script to process them all, one script to find them.
#  One script to wget them all, and on a large volume grind them."

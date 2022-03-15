#!/bin/bash
MESHFILES=../../mesh_files
DATADIR=data2

if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -d $DATADIR ]; then
    echo "[!] Data files do not exist. Preparing data..."
    # download preprocessed spherical data
	wget --no-check-certificate http://island.me.berkeley.edu/ugscnn/data/2d3ds_pano_small.zip

	# setup data
	unzip 2d3ds_pano_small.zip
	mv 2d3ds_pano_small data
	rm 2d3ds_pano_small.zip
fi

# create log directory
mkdir -p logs

#source activate

python3 train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder data2 \
--fold 3 \
--log_dir logs/log_FCN8s_f16_cv3_rgbd_data2 \
--decay \
--train_stats_freq 5 \
--model FCN8s \
--in_ch rgbd \
--lr 1e-3 \
--feat 16

# FCN8s, UNet


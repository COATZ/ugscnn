#!/bin/bash
MESHFILES=../../mesh_files
DATADIR=data


python3 test.py \
--test-batch-size 16 \
--data_folder data \
--fold 3 \
--ckpt logs/log_unet_f16_cv3_rgbd/checkpoint_latest.pth.tar_UNet_200.pth.tar \
--model UNet \
--in_ch rgbd \
--feat 16

#python3 train.py \
#--batch-size 16 \
#--test-batch-size 16 \
#--epochs 200 \
#--data_folder data \
#--fold 3 \
#--log_dir logs/log_unet_f16_cv3_rgbd \
#--decay \
#--train_stats_freq 5 \
#--model UNet \
#--in_ch rgbd \
#--lr 1e-3 \
#--feat 16

# FCN8s, UNet

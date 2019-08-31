#!/usr/bin/env bash

WEIGHTS_SOURCE_ZIP_PATH="ssd_keras/data/VGG_VOC0712_SSD_300x300_iter_120000.zip"
unzip -o $WEIGHTS_SOURCE_ZIP_PATH -d ssd_keras/data

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python create_subsampled_weights.py
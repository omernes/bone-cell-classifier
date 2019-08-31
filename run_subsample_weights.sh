#!/usr/bin/env bash

WEIGHTS_SOURCE_ZIP_PATH="ssd_keras/data/VGG_VOC0712_SSD_300x300_iter_120000.zip"
unzip $WEIGHTS_SOURCE_ZIP_PATH

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python ssd_keras/create_subsampled_weights.py
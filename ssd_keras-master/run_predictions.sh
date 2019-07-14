#!/usr/bin/env bash

export MODEL_PATH=ssd300_pascal_07+12_epoch-80_loss-4.4898_val_loss-5.6198.h5
export IMAGES_DIR=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/JPEGImages
export ANNOTATIONS_DIR=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/Annotations
export DATASET_FILENAME=datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python predict.py
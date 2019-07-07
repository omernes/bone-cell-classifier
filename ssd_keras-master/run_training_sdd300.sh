#!/usr/bin/env bash

#export IMAGES_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/JPEGImages
#export IMAGES_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/JPEGImages
#export IMAGES_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/JPEGImages
#
#export ANNOT_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/Annotations
#export ANNOT_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/Annotations
#export ANNOT_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/Annotations

export IMAGES_DIR_2007=~/VOCdevkit/VOC2007/JPEGImages
export IMAGES_DIR_2007_TEST=~/VOCdevkit/VOC2007/JPEGImages
export IMAGES_DIR_2012=~/VOCdevkit/VOC2012/JPEGImages

export ANNOT_DIR_2007=~/VOCdevkit/VOC2007/Annotations
export ANNOT_DIR_2007_TEST=~/VOCdevkit/VOC2007/Annotations
export ANNOT_DIR_2012=~/VOCdevkit/VOC2012/Annotations


export IMAGESET_TRAIN_2007=~/VOCdevkit/VOC2007/ImageSets/Main/train.txt
export IMAGESET_TRAIN_2012=~/VOCdevkit/VOC2012/ImageSets/Main/train.txt
export IMAGESET_TRAINVAL_2007=~/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
export IMAGESET_TRAINVAL_2012=~/VOCdevkit/VOC20012/ImageSets/Main/trainval.txt
export IMAGESET_TEST_2007=~/VOCdevkit/VOC2007/ImageSets/Main/test.txt

export CUDA_VISIBLE_DEVICES=3

../venv/bin/python ssd300_training.py
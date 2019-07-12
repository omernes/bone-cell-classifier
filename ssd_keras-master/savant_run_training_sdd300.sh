#!/usr/bin/env bash


export IMAGES_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/JPEGImages
export IMAGES_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/JPEGImages
export IMAGES_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/JPEGImages

export ANNOT_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/Annotations
export ANNOT_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/Annotations
export ANNOT_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/Annotations

export IMAGESET_TRAIN_2007=datasets/VOCdevkit/VOC2007/ImageSets/Main/train.txt
export IMAGESET_TRAIN_2012=datasets/VOCdevkit/VOC2012/ImageSets/Main/train.txt
export IMAGESET_TRAINVAL_2007=datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
export IMAGESET_TRAINVAL_2012=datasets/VOCdevkit/VOC20012/ImageSets/Main/trainval.txt
export IMAGESET_TEST_2007=datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt

#export IMAGES_DIR_2007=~/VOCdevkit/VOC2007/JPEGImages
#export IMAGES_DIR_2007_TEST=~/VOCdevkit/VOC2007/JPEGImages
#export IMAGES_DIR_2012=~/VOCdevkit/VOC2012/JPEGImages
#
#export ANNOT_DIR_2007=~/VOCdevkit/VOC2007/Annotations
#export ANNOT_DIR_2007_TEST=~/VOCdevkit/VOC2007/Annotations
#export ANNOT_DIR_2012=~/VOCdevkit/VOC2012/Annotations


#export IMAGESET_TRAIN_2007=~/VOCdevkit/VOC2007/ImageSets/Main/train.txt
#export IMAGESET_TRAIN_2012=~/VOCdevkit/VOC2012/ImageSets/Main/train.txt
#export IMAGESET_TRAINVAL_2007=~/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
#export IMAGESET_TRAINVAL_2012=~/VOCdevkit/VOC20012/ImageSets/Main/trainval.txt
#export IMAGESET_TEST_2007=~/VOCdevkit/VOC2007/ImageSets/Main/test.txt

export CUDA_VISIBLE_DEVICES=1,2,3

export SAVE_IN_MEMORY=1
export INITIAL_EPOCH=0
export FINAL_EPOCH=40
export STEPS_PER_EPOCH=500
export BATCH_SIZE=32

/home/gamir/edocohen/virtualenv/bone_cell_py36_env/bin/python ssd300_training.py
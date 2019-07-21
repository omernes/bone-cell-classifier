#!/usr/bin/env bash

export IMAGES_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/JPEGImages
export ANNOT_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007TEST/Annotations
export IMAGESET_TEST_2007=~/VOCdevkit/VOC2007/ImageSets/Main/test.txt

export CUDA_VISIBLE_DEVICES=1,2,3,4

export MODEL_PATH=
export RESULTS_PATH=

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python ssd300_evaluation.py
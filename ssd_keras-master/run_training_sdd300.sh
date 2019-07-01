#!/usr/bin/env bash

IMAGES_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/JPEGImages
IMAGES_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/JPEGImages
IMAGES_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/JPEGImages

ANNOT_DIR_2007=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/Annotations
ANNOT_DIR_2007_TEST=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2007/Annotations
ANNOT_DIR_2012=/specific/netapp5_2/gamir/datasets/VOCdevkit/VOC2012/Annotations

CUDA_VISIBLE_DEVICES=3

../venv/bin/python ssd300_training.py
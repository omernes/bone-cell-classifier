#!/usr/bin/env bash

export IMAGES_DIR=/a/home/cc/students/csguests/omernestor/data_xml/images
export ANNOTATIONS_DIR=/a/home/cc/students/csguests/omernestor/data_xml/annotations
export TARGET_DIR=/a/home/cc/students/csguests/omernestor/data_xml

#export IMAGES_DIR=/vol/scratch/bone_cell/data_xml/images
#export ANNOTATIONS_DIR=/vol/scratch/bone_cell/data_xml/annotations
#export TARGET_DIR=/vol/scratch/bone_cell/data_xml

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python split_dataset.py
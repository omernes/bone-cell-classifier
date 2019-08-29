#!/usr/bin/env bash

export IMAGES_DIR=/a/home/cc/students/csguests/omernestor/raw_data
export ANNOTATIONS_DIR=/a/home/cc/students/csguests/omernestor/raw_data

export TARGET_IMAGES_DIR=/a/home/cc/students/csguests/omernestor/data_xml/images
export TARGET_ANNOTATIONS_DIR=/a/home/cc/students/csguests/omernestor/data_xml/annotations

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python create_dataset.py
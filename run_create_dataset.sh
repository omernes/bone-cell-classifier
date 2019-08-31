#!/usr/bin/env bash

export DATASET_DIR=data_raw/raw

export TARGET_DATASET_DIR=data_xml2

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python create_dataset.py

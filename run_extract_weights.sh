#!/usr/bin/env bash

export MODEL_PATH=ssd300_pascal_07+12_epoch-80_loss-4.4898_val_loss-5.6198.h5

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python ssd_keras/extract_weights_from_model.py
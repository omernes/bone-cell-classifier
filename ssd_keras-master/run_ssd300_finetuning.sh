#!/usr/bin/env bash

export IMAGES_DIR=/a/home/cc/students/csguests/omernestor/data_xml/images
export ANNOTATIONS_DIR=/a/home/cc/students/csguests/omernestor/data_xml/annotations

export TRAIN_IMAGESET_FILENAME=/a/home/cc/students/csguests/omernestor/data_xml/train.txt
export VAL_IMAGESET_FILENAME=/a/home/cc/students/csguests/omernestor/data_xml/val.txt
export TEST_IMAGESET_FILENAME=/a/home/cc/students/csguests/omernestor/data_xml/test.txt

export WEIGHTS_PATH=data/VGG_VOC0712_SSD_300x300_iter_120000_subsampled_6_classes.h5
export SAVE_IN_MEMORY=1
export MODEL_PATH="/vol/scratch/bone_cell/ssd300_bone-cell-dataset_epoch-20_loss-3.2703_val_loss-2.7799.h5"
export NUM_CLASSES=5
export BATCH_SIZE=32

export INITIAL_EPOCH=20
export FINAL_EPOCH=100
export CHECKPOINT_PERIOD=1
export STEPS_PER_EPOCH=1000
export MODEL_CHECKPOINT_PATH=/vol/scratch/bone_cell

export CUDA_VISIBLE_DEVICES=4,5,6,7

/home/gamir/edocohen/anaconda3/envs/bone_cell_py37/bin/python ssd300_finetuning.py
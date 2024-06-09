# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @Author :lxztju
# @Github : https://github.com/lxztju
# import pipeline

## Number of classes in the dataset
NUM_CLASSES = 2

# Batch size during training and validation
BATCH_SIZE = 128

# Batch size during testing
tBATCH_SIZE = 16

# Default input image size for the network
INPUT_SIZE = [96, 96]

## Path to the pretrained model
# Download URL: https://download.pytorch.org/models/densenet169-b2777c0a.pth
PRETRAINED_MODEL = r'D:\2024-project\classification\dense169.pth'

## Path to save the trained model weights, default is under trained_model
TRAINED_MODEL = r'D:\2024-project\classification\trained-models\dense169_100.pth'

# Path to the dataset
TRAIN_DATASET_DIR = r'D:\2024-project\classification\data\train'
VAL_DATASET_DIR = r'D:\2024-project\classification\data\valid'
TEST_DATASET_DIR = r'D:\classification\data\test'
# # VAL_DATASET_DIR = r'D:\2024-project\classification\test-processed\circ0034\resized_images'
#
# TEST_DATASET_DIR = r'D:\2024-project\classification\test-processed\circ0034\resized_images'
# TEST_DATASET_DIR = pipeline.target_folder_path

labels_to_classes = {'0': 'no',
                     '1': 'yes'
}

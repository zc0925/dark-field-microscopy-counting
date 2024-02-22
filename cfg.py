# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# import pipeline

##数据集的类别
NUM_CLASSES = 2

#训练,validation时batch的大小
BATCH_SIZE = 128

#测试的batch大小
tBATCH_SIZE=16
#网络默认输入图像的大小
INPUT_SIZE = [96,96]

##预训练模型的存放位置
#下载地址：https://download.pytorch.org/models/densenet169-b2777c0a.pth
PRETRAINED_MODEL = r'D:\2024-project\classification\dense169.pth'

##训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = r'D:\2024-project\classification\trained-models\dense169_100.pth'


#数据集的存放位置
TRAIN_DATASET_DIR = r'D:\2024-project\classification\data\train'
VAL_DATASET_DIR = r'D:\2024-project\classification\data\valid'
TEST_DATASET_DIR=r'D:\classification\data\test'
# # VAL_DATASET_DIR = r'D:\2024-project\classification\test-processed\circ0034\resized_images'
#
# TEST_DATASET_DIR = r'D:\2024-project\classification\test-processed\circ0034\resized_images'
# TEST_DATASET_DIR = pipeline.target_folder_path
labels_to_classes = {'0':'no',
                     '1':'yes'
}

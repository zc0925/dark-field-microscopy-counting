# -*- coding:utf-8 -*-
# @time :2019.09.06
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
import torch
from torchvision import transforms, datasets
import cfg
import torch.utils.data as data
import torchvision.datasets as datasets
import os
from PIL import Image
import glob


# 构建数据提取器，利用dataloader
# 利用torchvision中的transforms进行图像预处理
#cfg为config文件，保存几个方便修改的参数

input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE
test_path=cfg.TEST_DATASET_DIR+'/'
train_transforms = transforms.Compose([
    transforms.Resize([96,96]),
    # transforms.RandomRotation(10),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

val_transforms = transforms.Compose([
    transforms.Resize([96,96]),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor()
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
test_transforms = transforms.Compose([
    transforms.Resize([96,96]),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor()
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])



class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in sorted(glob.glob(test_path + "*.png")):
            images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename)).convert('RGB')
       
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)    
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
##ImageFolder对象可以将一个文件夹下的文件构造成一类
#所以数据集的存储格式为一个类的图片放置到一个文件夹下
#然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数
#设置多线程，来相对提升数据提取的速度

train_dir = cfg.TRAIN_DATASET_DIR
# train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_datasets=ImageFolderWithPaths(train_dir,transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)


val_dir = cfg.VAL_DATASET_DIR
val_datasets = ImageFolderWithPaths(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=0)

test_dir= cfg.TEST_DATASET_DIR
# test_dir = create_the_target_path()
# test_datasets = ImageFolderWithPaths(test_dir, transform=test_transforms)
test_dataloader=data.DataLoader(TestImageFolder(test_dir,transform=test_transforms),batch_size=batch_size,shuffle=False,num_workers=0)


# for images,paths in test_dataloader:
#     print(paths)
##进行数据提取函数的测试
# if __name__ =="__main__":
#
#     for images,paths in test_dataloader:
#         print(paths)
# #    for ims,labs,paths in val_dataloader:
# #        print(ims.shape)

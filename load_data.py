# -*- coding:utf-8 -*-
# @time :2019.09.06
# @IDE : pycharm
# @Author :lxztju
# @Github : https://github.com/lxztju
import torch
from torchvision import transforms, datasets
import cfg
import torch.utils.data as data
import os
from PIL import Image
import glob

# Construct data loader using DataLoader
# Use transforms from torchvision for image preprocessing
# cfg is a config file containing several modifiable parameters

input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE
test_path = cfg.TEST_DATASET_DIR + '/'

train_transforms = transforms.Compose([
    transforms.Resize([96, 96]),
    # transforms.RandomRotation(10),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

val_transforms = transforms.Compose([
    transforms.Resize([96, 96]),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor()
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

test_transforms = transforms.Compose([
    transforms.Resize([96, 96]),
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

    # Override the __getitem__ method. This is the method that DataLoader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

## The ImageFolder object can structure files in a folder into a class
# Thus, the storage format of the dataset is to place the images of a class into a folder
# Then use DataLoader to construct the extractor, which returns a batch of data each time
# In many cases, use the num_worker parameter to set multi-threading to relatively improve the data extraction speed

train_dir = cfg.TRAIN_DATASET_DIR
train_datasets = ImageFolderWithPaths(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)

val_dir = cfg.VAL_DATASET_DIR
val_datasets = ImageFolderWithPaths(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=0)

test_dir = cfg.TEST_DATASET_DIR
test_dataloader = data.DataLoader(TestImageFolder(test_dir, transform=test_transforms), batch_size=batch_size, shuffle=False, num_workers=0)

# Test the data extraction function
# if __name__ == "__main__":
#     for images, paths in test_dataloader:
#         print(paths)
#     for ims, labs, paths in val_dataloader:
#         print(ims.shape)

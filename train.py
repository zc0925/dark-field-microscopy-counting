# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @Author :lxztju
# @Github : https://github.com/lxztju

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim


from load_data import train_dataloader, train_datasets
from models.densenet import densenet169
import cfg

## Command line interaction, setting some basic parameters
parser = argparse.ArgumentParser("Train the densenet")

parser.add_argument('-max', '--max_epoch', default=200,
                    help='maximum epoch for training')

parser.add_argument('-b', '--batch_size', default=8,
                    help='batch size for training')

parser.add_argument('-ng', '--ngpu', default=1,
                    help='use multi gpu to train')

parser.add_argument('-lr', '--learning_rate', default=5e-4,
                    help='initial learning rate for training')

## Directory to save trained models
parser.add_argument('--save_folder', default='trained-models',
                    help='the directory to save trained model ')

args = parser.parse_args()

## Create directory for saving trained model parameters
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

##### Build the network model
model = densenet169(num_classes=cfg.NUM_CLASSES)

# Print the model architecture
print(model)

### Load the pretrained weights
pretrained_path = cfg.PRETRAINED_MODEL

print("Initializing the network ...")
# Load pretrained model parameters
# PyTorch usually stores model parameters in a dictionary format, where keys are the names of each layer, and values are the parameters

state_dict = torch.load(pretrained_path)

### Remove the weights of the fully connected layer,
# As we generally do not use the 1000 classes of imagenet directly, we need to replace the last fully connected layer of the network
# Therefore, we need to save the parameters of the previous layers and reinitialize the last layer
# Define a new dictionary, map the original parameters to the new dictionary according to the network definition
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    # Print the keys of the pretrained model and find that they differ from the defined network keys, so we need to modify the keys accordingly
    # Torchvision uses regular expressions to modify keys, but here we use if statements to filter mismatched keys directly
    ### Correct key mismatches
    if k.split('.')[0] == 'features' and (len(k.split('.')) > 4):
        k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + k.split('.')[-2] + '.' + k.split('.')[-1]
    else:
        pass
    ## Initialize the weights of the last fully connected layer
    if k.split('.')[0] == 'classifier':
        if k.split('.')[-1] == 'weights':
            v = nn.init.kaiming_normal(model.state_dict()[k], mode='fan_out')
        else:
            model.state_dict()[k][...] = 0.0
            v = model.state_dict()[k][...]
    else:
        pass
    ## Get the new pretrained parameters corresponding to the defined network
    new_state_dict[k] = v
## Load the network parameters
model.load_state_dict(new_state_dict)

## Parallel computing with multiple GPUs
if args.ngpu:
    model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
print("Network initialization done")

### Place the model on the GPU for computation
if torch.cuda.is_available():
    model.cuda()

## Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_func = nn.CrossEntropyLoss()
# loss_fn=focal_loss(alpha=0.25,gamma=2,num_classes=2)

args.batch_size = cfg.BATCH_SIZE

# Number of batches per epoch
max_batch = len(train_datasets) // args.batch_size

## Train for max_epoch epochs
for epoch in range(args.max_epoch):
    model.train()  ## Use train() when training, and use eval() when testing
    ## eval() will fix BN and Dropout parameters during testing
    batch = 0

    for batch_images, batch_labels, paths in train_dataloader:
        average_loss = 0
        train_acc = 0
        ## In pytorch 0.4 and later, Variable is merged with tensor, so no need to wrap with Variable
        if torch.cuda.is_available():
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()
        out = model(batch_images)
        loss = loss_func(out, batch_labels)

        average_loss = loss
        prediction = torch.max(out, 1)[1]

        train_correct = (prediction == batch_labels).sum()
        ## train_correct is a longtensor, need to convert to float
        train_acc = (train_correct.float()) / args.batch_size

        optimizer.zero_grad() # Clear gradient information, otherwise it will accumulate in each backpropagation
        loss.backward()  # Backpropagation
        optimizer.step()  ## Update gradients

        batch += 1
        print("Epoch: %d/%d || batch:%d/%d average_loss: %.3f || train_acc: %.2f"
              % (epoch, args.max_epoch, batch, max_batch, average_loss, train_acc))

## Save the model every 10 epochs
    if epoch % 10 == 0 and epoch > 0:
        torch.save(model.state_dict(), args.save_folder + '/' + 'dense169' + '_' + str(epoch) + '.pth')

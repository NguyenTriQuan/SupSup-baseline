# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from genericpath import isdir
from PIL import Image
import os
import os.path
import sys
import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import transforms

import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader

import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K
from args import args

class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train):
        super(MiniImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='test'
        root = os.path.join(root, 'miniimagenet')
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = np.array(data_dict['images'])
        self.targets = np.array(data_dict['labels'])
        print('Done')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.targets[i]
        return img, label

class SplitMiniImageNet20Tasks:
    def __init__(self):
        super(SplitMiniImageNet20Tasks, self).__init__()
        np.random.seed(args.seed)
        data={}
        taskcla=[]
        size=[3, 84, 84]
        task_order=shuffle(np.arange(20),random_state=args.seed)
        print('Task order =',task_order+1)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tasknum = args.num_tasks
        # CIFAR100    
        train_set = MiniImageNet('../dat/',train=True)
        test_set = MiniImageNet('../dat/',train=False)

        train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
        test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

        train_data = train_data.permute(0, 3, 1, 2)/255.0
        test_data = test_data.permute(0, 3, 1, 2)/255.0

        n_old = 0
        for t in range(tasknum):
            data[t]={}
            data[t]['name']='mini_imagenet-'+str(task_order[t]+1)
            data[t]['ncla']=5
            #train and valid
            ids = (train_targets//5 == task_order[t])
            images = train_data[ids]
            labels = train_targets[ids]%5 

            # r=np.arange(images.size(0))
            # r=np.array(shuffle(r,random_state=args.seed),dtype=int)
            # nvalid=int(pc_valid*len(r))
            # ivalid=torch.LongTensor(r[:nvalid])
            # itrain=torch.LongTensor(r[nvalid:])
            # data[t]['train_loader'] = DataLoader(TensorDataset(images[itrain], labels[itrain]), batch_size=args.batch_size, shuffle=True)
            # data[t]['valid_loader'] = DataLoader(TensorDataset(images[ivalid], labels[ivalid]), batch_size=args.val_batch_size, shuffle=False)

            data[t]['train_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)

            #test
            ids = (test_targets//5 == task_order[t])
            images = test_data[ids]
            labels = test_targets[ids]%5

            data[t]['test_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.test_batch_size, shuffle=False)

            n_old += 5

        # if args.augment:
        #     data['train_transform'] = torch.nn.Sequential(
        #         K.augmentation.RandomResizedCrop(size=(84, 84), scale=(0.2, 1.0), same_on_batch=False),
        #         K.augmentation.RandomHorizontalFlip(),
        #         K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=False),
        #         K.augmentation.RandomGrayscale(p=0.2),
        #         K.augmentation.Normalize(mean, std),
        #     )
        # else:
        data['train_transform'] = torch.nn.Sequential(
            K.augmentation.Normalize(mean, std),
        )
            
        data['valid_transform'] = torch.nn.Sequential(
            K.augmentation.Normalize(mean, std),
        )
        # Others
        n=0
        for t in range(tasknum):
            taskcla.append((t,data[t]['ncla']))
            n+=data[t]['ncla']
        data['ncla']=n

        self.data = data


    def update_task(self, i):
        self.train_loader = self.data[i]['train_loader']
        self.test_loader = self.data[i]['test_loader']
        self.train_transform = self.data['train_transform']
        self.test_transform = self.data['valid_transform']
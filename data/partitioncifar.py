import numpy as np
import os,sys
import torch

from torchvision import datasets, transforms

import copy

from args import args
import utils

import torch.utils.data as data_utils
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K


def partition_dataset(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10:
    def __init__(self):
        super(PartitionCIFAR10, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_dataset(train_dataset, 2 * i),
                partition_dataset(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print()
            print(f"=> Size of train split {i}: {len(splits[i][0].data)}")
            print(f"=> Size of val split {i}: {len(splits[i][1].data)}")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv2(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10V2:
    def __init__(self):
        super(PartitionCIFAR10V2, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv2(train_dataset, 2 * i),
                partition_datasetv2(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print(len(splits[i][0].data))
            print(len(splits[i][1].data))
            print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv3(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]
    return newdataset


class PartitionCIFAR100V2:
    def __init__(self):
        super(PartitionCIFAR100V2, self).__init__()
        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv3(train_dataset, 5 * i),
                partition_datasetv3(val_dataset, 5 * i),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]




def partition_datasetv4(dataset, perm):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

    newdataset.targets = [
        lperm.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    return newdataset

class RandSplitCIFAR100:
    def __init__(self):
        super(RandSplitCIFAR100, self).__init__()
        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed)
        perm = np.random.permutation(100)
        print(perm)

        splits = [
            (
                partition_datasetv4(train_dataset, perm[5 * i:5 * (i+1)]),
                partition_datasetv4(val_dataset, perm[5 * i:5 * (i+1)]),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")
        [print(perm[5 * i:5 * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]

class SplitCIFAR100_10Tasks:
    def __init__(self):
        super(SplitCIFAR100_10Tasks, self).__init__()
        np.random.seed(args.seed)
        data={}
        taskcla=[]
        size=[3,32,32]
        task_order=shuffle(np.arange(10),random_state=args.seed)
        print('Task order =',task_order+1)

        mean=torch.tensor([x/255 for x in [125.3,123.0,113.9]])
        std=torch.tensor([x/255 for x in [63.0,62.1,66.7]])
        if args.tasknum > 10:
            tasknum = 10
        else:
            tasknum = args.tasknum
        # CIFAR100    
        train_set=datasets.CIFAR100('../dat/',train=True,download=True)
        test_set=datasets.CIFAR100('../dat/',train=False,download=True)

        train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
        test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

        train_data = train_data.permute(0, 3, 1, 2)/255.0
        test_data = test_data.permute(0, 3, 1, 2)/255.0

        n_old = 0
        for t in range(tasknum):
            data[t]={}
            data[t]['name']='cifar100-'+str(task_order[t]+1)
            data[t]['ncla']=10
            #train and valid
            ids = (train_targets//10 == task_order[t])
            images = train_data[ids]
            labels = train_targets[ids]%10 
            if args.cil:
                labels += n_old

            # r=np.arange(images.size(0))
            # r=np.array(shuffle(r,random_state=args.seed),dtype=int)
            # nvalid=int(pc_valid*len(r))
            # ivalid=torch.LongTensor(r[:nvalid])
            # itrain=torch.LongTensor(r[nvalid:])
            # data[t]['train_loader'] = DataLoader(TensorDataset(images[itrain], labels[itrain]), batch_size=args.batch_size, shuffle=True)
            # data[t]['valid_loader'] = DataLoader(TensorDataset(images[ivalid], labels[ivalid]), batch_size=args.val_batch_size, shuffle=False)

            data[t]['train_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)

            #test
            ids = (test_targets//10 == task_order[t])
            images = test_data[ids]
            labels = test_targets[ids]%10
            if args.cil:
                labels += n_old
            data[t]['test_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.val_batch_size, shuffle=False)

            n_old += 10

        # if args.augment:
        #     data['train_transform'] = torch.nn.Sequential(
        #         K.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), same_on_batch=False),
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

class SplitCIFAR10_100_11Tasks:
    def __init__(self):
        super(SplitCIFAR10_100_11Tasks, self).__init__()
        np.random.seed(args.seed)
        data={}
        taskcla=[]
        size=[3,32,32]
        task_order=shuffle(np.arange(10),random_state=args.seed)
        print('Task order =',task_order+1)

        mean=torch.tensor([x/255 for x in [125.3,123.0,113.9]])
        std=torch.tensor([x/255 for x in [63.0,62.1,66.7]])
        if args.tasknum > 11:
            tasknum = 11
        else:
            tasknum = args.tasknum

        # CIFAR10
        train_set = datasets.CIFAR10('../dat/',train=True,download=True)
        test_set = datasets.CIFAR10('../dat/',train=False,download=True)
        train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
        test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

        train_data = train_data.permute(0, 3, 1, 2)/255.0
        test_data = test_data.permute(0, 3, 1, 2)/255.0

        data[0]={}
        data[0]['name']='cifar10'
        data[0]['ncla']=10

        #train and valid
        # r=np.arange(train_data.size(0))
        # r=np.array(shuffle(r,random_state=args.seed),dtype=int)
        # nvalid=int(pc_valid*len(r))
        # ivalid=torch.LongTensor(r[:nvalid])
        # itrain=torch.LongTensor(r[nvalid:])
        # data[0]['train_loader'] = DataLoader(TensorDataset(train_data[itrain], train_targets[itrain]), batch_size=256, shuffle=True)
        # data[0]['valid_loader'] = DataLoader(TensorDataset(train_data[ivalid], train_targets[ivalid]), batch_size=args.val_batch_size, shuffle=False)
        data[0]['train_loader'] = DataLoader(TensorDataset(train_data, train_targets), batch_size=256, shuffle=True)
        #test
        data[0]['test_loader'] = DataLoader(TensorDataset(test_data, test_targets), batch_size=args.val_batch_size, shuffle=False)

        # CIFAR100
        train_set = datasets.CIFAR100('../dat/',train=True,download=True)
        test_set = datasets.CIFAR100('../dat/',train=False,download=True)

        train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
        test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

        train_data = train_data.permute(0, 3, 1, 2)/255.0
        test_data = test_data.permute(0, 3, 1, 2)/255.0

        n_old = 10
        for t in range(1, tasknum):
            data[t]={}
            data[t]['name']='cifar100-'+str(task_order[t-1]+1)
            data[t]['ncla']=10
            #train and valid
            ids = (train_targets//10 == task_order[t-1])
            images = train_data[ids]
            labels = train_targets[ids]%10 
            if args.cil:
                labels += n_old

            # r=np.arange(images.size(0))
            # r=np.array(shuffle(r,random_state=args.seed),dtype=int)
            # nvalid=int(pc_valid*len(r))
            # ivalid=torch.LongTensor(r[:nvalid])
            # itrain=torch.LongTensor(r[nvalid:])
            # data[t]['train_loader'] = DataLoader(TensorDataset(images[itrain], labels[itrain]), batch_size=args.batch_size, shuffle=True)
            # data[t]['valid_loader'] = DataLoader(TensorDataset(images[ivalid], labels[ivalid]), batch_size=args.val_batch_size, shuffle=False)
            data[t]['train_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)
            #test
            ids = (test_targets//10 == task_order[t-1])
            images = test_data[ids]
            labels = test_targets[ids]%10
            if args.cil:
                labels += n_old
            data[t]['test_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.val_batch_size, shuffle=False)

            n_old += 10

        # if args.augment:
        #     data['train_transform'] = torch.nn.Sequential(
        #         K.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), same_on_batch=False),
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
import os, sys
import torch
from torchvision import datasets, transforms

import numpy as np

from args import args

import torch.utils.data as data_utils
from sklearn.utils import shuffle



class MNIST:
    def __init__(self):
        super(MNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        return


class FashionMNIST:
    def __init__(self):
        super(FashionMNIST, self).__init__()

        data_root = os.path.join(args.data, "fashionmnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.FashionMNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        return



class Permute(object):
    def __call__(self, tensor):
        out = tensor.flatten()
        out = out[self.perm]
        return out.view(1, 28, 28)

    def __repr__(self):
        return self.__class__.__name__

class MNISTPerm:
    def __init__(self):
        super(MNISTPerm, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        self.permuter = Permute()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    self.permuter,
                ]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        self.permuter,
                    ]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        np.random.seed(i + args.seed)
        self.permuter.__setattr__("perm", np.random.permutation(784))

class PMNIST:
    def __init__(self):
        super(PMNIST, self).__init__()

        seed = args.seed
        tasknum = args.num_tasks
        np.random.seed(seed)
        data = {}
        taskcla = []
        size = [1, 28, 28]
        # Pre-load
        # MNIST
        mean = torch.Tensor([0.1307])
        std = torch.Tensor([0.3081])
        dat = {}
        dat['train'] = datasets.MNIST(args.data, train=True, download=True)
        dat['test'] = datasets.MNIST(args.data, train=False, download=True)
        
        for i in range(tasknum):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10
            permutation = np.random.permutation(28*28)
            for s in ['train', 'test']:
                if s == 'train':
                    arr = dat[s].train_data.view(dat[s].train_data.shape[0],-1).float()
                    label = torch.LongTensor(dat[s].train_labels)
                else:
                    arr = dat[s].test_data.view(dat[s].test_data.shape[0],-1).float()
                    label = torch.LongTensor(dat[s].test_labels)
                    
                arr = (arr/255 - mean) / std
                data[i][s]={}
                data[i][s]['x'] = arr[:,permutation].view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = label
                
        # Validation
        for t in range(tasknum):
            data[t]['train']['dataset'] = data_utils.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
            data[t]['test']['dataset'] = data_utils.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])

            # data[t]['valid'] = {}
            # data[t]['valid']['x'] = data[t]['train']['x'].clone()
            # data[t]['valid']['y'] = data[t]['train']['y'].clone()
            
            # r=np.arange(data[t]['train']['x'].size(0))
            # r=np.array(shuffle(r,random_state=seed),dtype=int)
            # nvalid=int(pc_valid*len(r))
            # ivalid=torch.LongTensor(r[:10000])
            # itrain=torch.LongTensor(r[10000:])
            # data[t]['valid']={}
            # data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
            # data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
            # data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
            # data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

        # Others
        n = 0
        for t in range(tasknum):
            taskcla.append((t, data[t]['ncla']))
            n += data[t]['ncla']
        data['ncla'] = n

        # Data loading code
        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    data[t]['train']['dataset'] , batch_size=args.batch_size, shuffle=True
                ),
                torch.utils.data.DataLoader(
                    data[t]['train']['dataset'], batch_size=args.test_batch_size, shuffle=True
                ),
                torch.utils.data.DataLoader(
                    data[t]['test']['dataset'], batch_size=args.test_batch_size, shuffle=True
                ),
            )
            for t in range(tasknum)
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
        self.test_loader = self.loaders[i][2]
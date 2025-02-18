# Dataloaders classes to be used with any model

from torch.utils.data import DataLoader
import os
import torch
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import datasets, transforms
from numpy.random import randint


class MNIST_DL():
    def __init__(self, data_path, type):
        self.type = type
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True,device='cuda', transform=None):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == "cuda" else {}
        if transform is None:
            tx = transforms.ToTensor()
        else :
            tx = transform
        datasetC = datasets.MNIST if self.type == 'numbers' else datasets.FashionMNIST
        train = DataLoader(datasetC('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC('../data', train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test


class MNIST_FASHION_DATALOADER():

    def __init__(self, data_path):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=None):
        print('Retrieving data from ', self.data_path)
        if not (os.path.exists(self.data_path + 'train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'train-ms-fashion-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-fashion-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + 'train-ms-mnist-idx.pt')
        t_fashion = torch.load(self.data_path + 'train-ms-fashion-idx.pt')
        s_mnist = torch.load(self.data_path + 'test-ms-mnist-idx.pt')
        s_fashion = torch.load(self.data_path + 'test-ms-fashion-idx.pt')

        # load base datasets
        t1,s1 = MNIST_DL(self.data_path,'numbers').getDataLoaders(batch_size,shuffle,device, transform)
        t2,s2 = MNIST_DL(self.data_path,'fashion').getDataLoaders(batch_size,shuffle,device, transform)


        train_mnist_fashion = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_fashion[i], size=len(t_fashion))
        ])

        test_mnist_fashion = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_fashion[i], size=len(s_fashion))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_fashion, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_fashion, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test



class BasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform = None):

        self.data = data # shape len_data x ch x w x h
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0 # We return 0 as label to have a dataset that is homogeneous with the other datasets

class MultimodalBasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        # data of shape n_mods x len_data x ch x w x h
        self.lenght = len(data[0])
        self.datasets = [BasicDataset(d, transform) for d in data ]

    def __len__(self):
        return self.lenght

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)


class SVHN_DL():

    def __init__(self, data_path = '../data'):
        self.data_path = data_path
        return

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

        train = DataLoader(datasets.SVHN(self.data_path, split='train', download=True, transform=transform),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN(self.data_path, split='test', download=True, transform=transform),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test


class MNIST_SVHN_DL():

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):

        if not (os.path.exists(self.data_path + '/train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/train-ms-svhn-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + '/train-ms-mnist-idx.pt')
        t_svhn = torch.load(self.data_path + '/train-ms-svhn-idx.pt')
        s_mnist = torch.load(self.data_path + '/test-ms-mnist-idx.pt')
        s_svhn = torch.load(self.data_path + '/test-ms-svhn-idx.pt')

        # load base datasets
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)

        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test


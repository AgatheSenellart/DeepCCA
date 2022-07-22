# Dataloaders for OASIS, Mnist and the tensor dataset
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from os import path
from torchvision import transforms, datasets
from torchnet.dataset import TensorDataset
from resample_dataset import ResampleDataset
import pandas as pd

class MRIDataset(Dataset):
    """This dataset include the preprocessed MRI of a list of subjects"""

    def __init__(self, img_dir, data_df, transform=None):
        """
        Args:
            img_dir: (str) path to the images directory.
            data_df: (DataFrame) list of subjects / sessions used.
            transform: Optional, transformations applied to the tensor
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data_df = data_df
        self.data_np = data_df.values
        self.label_code = {"AD": 1, "CN": 0}

        self.size = self[0][0].shape

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) the index of the subject/session whom data is loaded.
        Returns:
            sample: (dict) corresponding data described by the following keys:
                image: (Tensor) MR image
                label: (int) the diagnosis code (0 for CN or 1 for AD)
                participant_id: (str) ID of the participant (format sub-...)
                session_id: (str) ID of the session (format ses-M...)
        """
        print(idx, type(idx))
        print(self.data_df)
        diagnosis = self.data_np[idx,2]
        label = self.label_code[diagnosis]

        participant_id = self.data_np[idx,0]
        session_id = self.data_np[idx, 1]
        filename = participant_id + '_' + session_id + \
                   '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt'

        image = torch.load(path.join(self.img_dir, filename))

        if self.transform:
            image = self.transform(image)


        return image, label

    def train(self):
        """Put all the transforms of the dataset in training mode"""
        self.transform.train()

    def eval(self):
        """Put all the transforms of the dataset in evaluation mode"""
        self.transform.eval()


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
        train = DataLoader(datasetC(self.data_path, train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC(self.data_path, train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test



class MNIST_OASIS_DL():

    def __init__(self, oasis_transform = None, mnist_transform = None):
        self.name = 'mnist_oasis_dl'
        self.oasis_transform = oasis_transform
        self.mnist_transform = mnist_transform


    def getDataLoaders(self,batch_size, shuffle=True, device='cuda'):

        # get the linked indices
        t_mnist = torch.load('data/train-mo-mnist-idx.pt')
        s_mnist = torch.load('data/test-mo-mnist-idx.pt')
        t_oasis = torch.load('data/train-mo-oasis-idx.pt')
        s_oasis = torch.load('data/test-mo-oasis-idx.pt')

        print('coucou', t_oasis)
        print('coucou', s_oasis)
        # Get the base datasets
        print(t_oasis, len(t_oasis))
        t1,s1 = MNIST_DL('home/Code/vaes/mmvae/data', type='numbers').getDataLoaders(batch_size,shuffle,
                                                                                     device, self.mnist_transform)
        oasis_path = '/home/agathe/Code/datasets/OASIS-1_dataset/'
        train_df = pd.read_csv(oasis_path+'tsv_files/lab_1/train.tsv', sep='\t')
        test_df = pd.read_csv(oasis_path + 'tsv_files/lab_1/validation.tsv', sep ='\t')

        print(train_df)
        print(test_df)

        t2 = MRIDataset(oasis_path + 'preprocessed',train_df,self.oasis_transform)
        s2 = MRIDataset(oasis_path + 'preprocessed', test_df, self.oasis_transform)

        # Create the paired dataset

        train_mnist_oasis = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2,lambda d,i : t_oasis[i], size=len(t_oasis))
            # t2
        ])

        test_mnist_oasis = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2, lambda d, i: s_oasis[i], size=len(s_oasis))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_oasis, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_oasis, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test
# drvae/datamodules/datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from typing import Union, Literal
import os


class LabelOffsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, offset: int):
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y + self.offset

class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        """
        Args:
            dataset: The original dataset to wrap.
        """
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        out = self.dataset.__getitem__(idx)
        return out, idx

class VAEDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "~/scratch/datasets/", dataset: Literal["mnist", "fashionmnist", "kmnist", "kfmnist", "gmmrot"] = "mnist", batch_size: int = 128, subset_size: Union[int, None] = None, **kwargs):
        """
        Args:
        - dataset: string, the name of the dataset to use. Can be 'mnist', 'fashionmnist', 'kmnist', or 'kfmnist'.
        - batch_size: int, the batch size to use for training and validation.
        - subset_size: int or None, the number of samples to use from the training set. If None, use the full training set.
        """
        super().__init__()
        self.data_path = os.path.expanduser(data_path)
        self._validate_data_path()

        self.dataset = dataset.lower()
        self.batch_size = batch_size
        self.subset_size = subset_size

        if self.dataset == "gmmrot":
            self.gmm_k = kwargs.get("gmm_k", 8)
            self.gmm_seed = kwargs.get("gmm_seed", 123)
            self.gmm_train = kwargs.get("gmm_train", 60000)
            self.gmm_val = kwargs.get("gmm_val", 10000)
            self.gmm_pad_sigma = kwargs.get("gmm_pad_sigma", 0.02)
            self.gmm_dim = kwargs.get("gmm_dim", 50)
            self.gmm_symmetric_dataset = kwargs.get("gmm_symmetric_dataset", False)
                                            
    def _validate_data_path(self):
        if not os.path.exists(self.data_path):
            raise ValueError(f"data_path does not exist: {self.data_path}")
        if not os.path.isdir(self.data_path):
            raise ValueError(f"data_path is not a directory: {self.data_path}")
        if not os.access(self.data_path, os.R_OK):
            raise ValueError(f"data_path is not readable: {self.data_path}")

    def get_image_shape(self):
        if self.dataset == "mnist":
            return (1, 28, 28)
        elif self.dataset == "fashionmnist":
            return (1, 28, 28)
        elif self.dataset == "kmnist":
            return (1, 28, 28)
        elif self.dataset == "kfmnist":
            return (1, 28, 28)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def prepare_data(self):
        # Download datasets if needed
        if self.dataset == 'mnist':  
            datasets.MNIST(self.data_path, train=True, download=True)
            datasets.MNIST(self.data_path, train=False, download=True)
        elif self.dataset == 'fashionmnist':
            datasets.FashionMNIST(self.data_path, train=True, download=True)
            datasets.FashionMNIST(self.data_path, train=False, download=True)
        elif self.dataset == 'kmnist':
            datasets.KMNIST(self.data_path, train=True, download=True)
            datasets.KMNIST(self.data_path, train=False, download=True)
        elif self.dataset == 'kfmnist':
            datasets.MNIST(self.data_path, train=True, download=True)
            datasets.MNIST(self.data_path, train=False, download=True)
            datasets.FashionMNIST(self.data_path, train=True, download=True)
            datasets.FashionMNIST(self.data_path, train=False, download=True)
            datasets.KMNIST(self.data_path, train=True, download=True)
            datasets.KMNIST(self.data_path, train=False, download=True)
        elif self.dataset == "gmmrot":
            pass
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}. Supported datasets are 'mnist', 'cifar10', 'gaussian', 'celeba', 'freyfaces', 'histopathologygray', 'omniglot', 'fashionmnist', and 'kmnist'.")

    def setup(self, stage: Union[str, None] = None):
        if self.dataset == "mnist":  
            transform = transforms.ToTensor()
            self.train_dataset = datasets.MNIST(self.data_path, train=True, transform=transform)
            self.val_dataset = datasets.MNIST(self.data_path, train=False, transform=transform)
            self.image_shape = (1, 28, 28)
        elif self.dataset == 'fashionmnist':
            transform = transforms.ToTensor()
            self.train_dataset = datasets.FashionMNIST(self.data_path, train=True, transform=transform)
            self.val_dataset = datasets.FashionMNIST(self.data_path, train=False, transform=transform)
            self.image_shape = (1, 28, 28)
        elif self.dataset == 'kmnist':
            transform = transforms.ToTensor()
            self.train_dataset = datasets.KMNIST(self.data_path, train=True, transform=transform)
            self.val_dataset = datasets.KMNIST(self.data_path, train=False, transform=transform)
            self.image_shape = (1, 28, 28)
        elif self.dataset == 'kfmnist':
            transform = transforms.ToTensor()
            mnist = LabelOffsetDataset(datasets.MNIST(self.data_path, train=True, transform=transform), offset=0)
            fashionmnist = LabelOffsetDataset(datasets.FashionMNIST(self.data_path, train=True, transform=transform), offset=10)
            kmnist = LabelOffsetDataset(datasets.KMNIST(self.data_path, train=True, transform=transform), offset=20)
            self.train_dataset = ConcatDataset([mnist, fashionmnist, kmnist])
            # Optionally for validation: use test sets and apply same offsets
            mnist_val = LabelOffsetDataset(datasets.MNIST(self.data_path, train=False, transform=transform), offset=0)
            fashionmnist_val = LabelOffsetDataset(datasets.FashionMNIST(self.data_path, train=False, transform=transform), offset=10)
            kmnist_val = LabelOffsetDataset(datasets.KMNIST(self.data_path, train=False, transform=transform), offset=20)
            self.val_dataset = ConcatDataset([mnist_val, fashionmnist_val, kmnist_val])
            #np.random.seed(42)
            #indices = np.random.permutation(len(self.val_dataset))
            #self.val_dataset = Subset(self.val_dataset, indices)
            self.image_shape = (1, 28, 28)
        elif self.dataset == "gmmrot":
            from ..datamodules.gmm_synthetic import GMMSynth
            gmm_synth = GMMSynth(n_train=self.gmm_train, n_val=self.gmm_val, dim=self.gmm_dim, k=self.gmm_k, seed=self.gmm_seed, pad_sigma=self.gmm_pad_sigma, symmetric_dataset=self.gmm_symmetric_dataset)
            self.train_dataset = gmm_synth.get_dataset('train')
            self.val_dataset = gmm_synth.get_dataset('val')
            self.image_shape = (1, 1, self.gmm_dim)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}. Supported datasets are 'mnist', 'cifar10', and 'gaussian'.")
        
        if self.subset_size is not None:
            np.random.seed(42)
            indices = np.random.choice(len(self.train_dataset), self.subset_size, replace=False)
            self.train_dataset = Subset(self.train_dataset, indices)

        self.train_dataset = IndexDataset(self.train_dataset)
        self.val_dataset = IndexDataset(self.val_dataset)   
    
    def train_dataloader(self, batch_size: int = None, shuffle: bool = True):
        if batch_size is None:
            batch_size = self.batch_size
        workers = os.cpu_count() or 0
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(20, workers))

    def val_dataloader(self, batch_size: int = None, shuffle: bool = False):
        if batch_size is None:
            batch_size = self.batch_size
        workers = os.cpu_count() or 0
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(20, workers))
    
    def test_dataloader(self, batch_size: int = None, shuffle: bool = False):
        return self.val_dataloader(batch_size=batch_size, shuffle=shuffle)
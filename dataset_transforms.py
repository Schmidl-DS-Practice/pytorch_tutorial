import torch
import torchvision
Dataset, DataLoader = torch.util.data.Dataset(), torch.utils.data.DataLoader()
import numpy as np
import math

# dataset = torchvsion.datasets.MNIST(root="./data", transform=torchvision.transforms.ToTensor())

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt(".data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

class MulTranform:

    def __init__(self, factor):

        self.factor = factor

    def __call__(self, sample):

        inputs, target = sample
        inputs *= self.factor

        return inputs, target

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.tranforms.Composed([ToTensor(), MulTranform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

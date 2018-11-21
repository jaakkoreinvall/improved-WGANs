# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def batch_generator(batch_size, phase, datapath):
    if phase == 'train':
        train = True
    elif phase == 'test':
        train = False
    else:
        raise ValueError
    data_transforms = transforms.Compose([transforms.ToTensor()])  
    dataset = datasets.MNIST(datapath, train=train, transform = data_transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    n_batches = len(dataloader)
    iterator = enumerate(dataloader)
    return iterator, n_batches
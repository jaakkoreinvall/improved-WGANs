# -*- coding: utf-8 -*-

import argparse

import batch_generator

parser = argparse.ArgumentParser(description='Wasserstein GAN with Gradient Penalty - MNIST')
parser.add_argument('--batch_size', type=int, default=50, help='batch size (default: 50)')
parser.add_argument('--datapath', required=True, help='path to dataset')
args = parser.parse_args()

batch_size = args.batch_size
datapath = args.datapath

# test batch generator
phase = 'train'
data_generator = batch_generator.batch_generator(batch_size, phase, datapath)
i, sampled_batch = next(data_generator)
print(sampled_batch)

while 1:
    try:
        i, sampled_batch = next(data_generator)
    except StopIteration:
        break
    print(i)
    
# test batch generator
phase = 'test'
data_generator = batch_generator.batch_generator(batch_size, phase, datapath)

while 1:
    try:
        i, sampled_batch = next(data_generator)
    except StopIteration:
        break
    print(i)
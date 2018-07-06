from __future__ import print_function, division

import os
from logging.config import fileConfig

import torch
from torch.utils.data import Dataset

#Custom lazy loader per file
from LazyTextDataset import LazyTextDataset


def load_large_dataset(base_dir):
    # Can make this dynamic, but in this sample 2 parts of the file
    datasets = []
    for f in os.listdir(base_dir):
        full_file_path = os.path.join(os.path.dirname(__file__), base_dir, f)
        datasets.append(LazyTextDataset(full_file_path))


    # Concare and load dataset
    mn_dataset_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets),
                                                    batch_size=1,
                                                    shuffle=True)
    return mn_dataset_loader




if __name__ == '__main__':
    fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))
    for id, labels in load_large_dataset(""):
        print(id, labels)
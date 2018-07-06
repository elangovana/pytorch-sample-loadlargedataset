from __future__ import print_function, division

import linecache

import os
from logging.config import fileConfig

import torch
from torch.utils.data import Dataset
import logging

#Custom lazy loader per file
class LazyTextDataset(Dataset):
    def __init__(self, filename):
        self._filename = filename
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialising for file {}".format(filename))
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        self._logger.debug("Executing __getitem__ file {}, with index {}".format(self._filename, idx))
        csv_line = linecache.getline(self._filename, idx + 1)
        return csv_line.split(",")

    def __len__(self):
        return self._total_data


def load_large_dataset(base_dir):
    # Can make this dynamic, but in this sample 2 parts of the file
    datasets = []
    for f in os.listdir(base_dir):
        full_file_path = os.path.join(os.path.dirname(__file__), base_dir, f)
        datasets.append( LazyTextDataset(full_file_path))


    # Concare and load dataset
    mn_dataset_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets),
                                                    batch_size=1,
                                                    shuffle=True)
    return mn_dataset_loader


def main(base_dir):
    for id, labels in load_large_dataset(base_dir):
        print(id, labels)


if __name__ == '__main__':
    fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))
    main("./data")
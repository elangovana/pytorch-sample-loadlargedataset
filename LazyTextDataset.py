import linecache
import logging

from torch.utils.data import Dataset


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
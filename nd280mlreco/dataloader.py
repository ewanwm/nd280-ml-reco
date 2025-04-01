import typing
from collections.abc import Callable
import os

import torch
from torch_geometric import data as tg_data

class HATDataset(tg_data.Dataset):

    def __init__(
            self,
            root:str,
            raw_filenames:list[str], 
            processed_file_names:list[str],
            transform = None,
            pre_transform=None, 
            pre_filter=None
            ):

        # keep copies of the raw and processed file names
        self.raw_filenames:list[str] = raw_filenames
        self.processed_filenames:list[str] = processed_file_names
        
        super().__init__(root = root, transform = transform, pre_transform = pre_transform, pre_filter = pre_filter)
    
    @property
    def raw_file_names(self) -> list[str]:
        return self.raw_filenames

    @property
    def processed_file_names(self) -> list[str]:
        return self.processed_filenames

    def process(self) -> None:
        idx = 0
        for raw_path in self.raw_filenames:
            # Read data from `raw_path`.
            data = torch.load(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.root, "processed", self.processed_file_names[idx]))
            idx += 1

    def len(self):
        return len(self.processed_filenames)

    def get(self, idx):
        data = torch.load(os.path.join(self.root, "processed", self.processed_file_names[idx]))
        return data


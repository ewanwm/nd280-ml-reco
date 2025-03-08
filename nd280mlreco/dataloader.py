import typing
import os

import torch
from torch_geometric.data import Dataset, Data

class HATDataset(Dataset):
    
    def __init__(self, filenames:list[str], processed_file_path:str):
        # keep copies of the raw and processed file names
        self._raw_filenames:list[str] = list(filenames)
        self._processed_filenames:list[str] = []

        self._processed_file_path:str = processed_file_path

    @property
    def raw_file_names(self) -> list[str]:
        return self._raw_filenames

    @property
    def processed_file_names(self) -> list[str]:
        return self._processed_filenames

    def process(self) -> None:
        raise NotImplementedError()
        
    def len(self):
        raise NotImplementedError()
    
    def get(self, idx):
        data = torch.load(os.path.join(self._processed_file_path, f'data_HAT_{idx}.pt'))
        return data


from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import numpy as np
import torch

class DualAugmentDataSet(Dataset):
    def __init__(self, base_dataset: CIFAR10, transform: Compose):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = ToTensor()(image)

        augment_1 = self.transform(image)
        augment_2 = self.transform(image)
        return augment_1, augment_2, label


class NormalizedDataSet(Dataset):
    def __init__(self, base_dataset: CIFAR10, transform: Compose):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = ToTensor()(image)
        image = self.transform(image)
        return image, label
    
    def get_image_by_index(self, idx):
        return self.base_dataset[idx][0]
    

class NeighborPairDataset(Dataset):
    def __init__(self, base_dataset: NormalizedDataSet, neighbor_indices: torch.Tensor):
        self.base_dataset = base_dataset
        self.neighbor_indices = neighbor_indices

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        neighbor_index = int(self.neighbor_indices[idx][torch.randint(0, len(self.neighbor_indices[idx]), (1,))].item())
        base_image, _ = self.base_dataset[idx]
        neighbor_image, _ = self.base_dataset[neighbor_index]
        return base_image, neighbor_image



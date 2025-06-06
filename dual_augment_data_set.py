from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor


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


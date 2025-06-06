from torchvision.datasets import CIFAR10
from augmentations import train_transform, test_transform
from dual_augment_data_set import DualAugmentDataSet
from torch.utils.data import DataLoader


def create_data(batch_size = 256):
    train_base = CIFAR10(root='./data', train=True, download=True, transform=None)
    test_base = CIFAR10(root='./data', train=False, download=True, transform=None)

    train_dataset = DualAugmentDataSet(train_base, train_transform)
    test_dataset = DualAugmentDataSet(test_base, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

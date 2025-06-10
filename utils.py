from torchvision.datasets import CIFAR10
from augmentations import train_transform, test_transform
from custom_datasets import DualAugmentDataSet, LinearProbingDataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def create_data_for_veicreg(batch_size = 256):
    train_base = CIFAR10(root='./data', train=True, download=True, transform=None)
    test_base = CIFAR10(root='./data', train=False, download=True, transform=None)

    train_dataset = DualAugmentDataSet(train_base, train_transform)
    test_dataset = DualAugmentDataSet(test_base, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_data_for_linear_probing(batch_size = 256):
    trainset = CIFAR10(root='./data', train=True, 
                                       transform=None)
    testset = CIFAR10(root='./data', train=False,
                                      transform=None)
    train_dataset = LinearProbingDataSet(trainset, test_transform)
    test_dataset = LinearProbingDataSet(testset, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def plot_vicreg_losses(train_losses, test_losses, batch_numbers, test_epochs, batches_per_epoch):
    
    test_batch_numbers = [(epoch + 1) * batches_per_epoch for epoch in test_epochs]
    
    loss_components = ['invariance', 'variance', 'covariance']
    titles = ['Invariance Loss', 'Variance Loss', 'Covariance Loss']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (component, title) in enumerate(zip(loss_components, titles)):
        ax = axes[i]
        
        ax.plot(batch_numbers, train_losses[component], 
                label='Training', alpha=0.7, linewidth=0.8)
        
        ax.plot(test_batch_numbers, test_losses[component], 
                'ro-', label='Test', markersize=4, linewidth=2)
        
        ax.set_xlabel('Training Batch')
        ax.set_ylabel(f'{title}')
        ax.set_title(f'{title} vs Training Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for epoch in range(1, len(test_epochs) + 1):
            if epoch * batches_per_epoch <= max(batch_numbers):
                ax.axvline(x=epoch * batches_per_epoch, color='gray', 
                          linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('vicreg_loss_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    for i, (component, title) in enumerate(zip(loss_components, titles)):
        plt.figure(figsize=(10, 6))
        
        plt.plot(batch_numbers, train_losses[component], 
                label='Training', alpha=0.7, linewidth=0.8)
        
        plt.plot(test_batch_numbers, test_losses[component], 
                'ro-', label='Test', markersize=4, linewidth=2)
        
        plt.xlabel('Training Batch')
        plt.ylabel(f'{title}')
        plt.title(f'{title} vs Training Batches')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for epoch in range(1, len(test_epochs) + 1):
            if epoch * batches_per_epoch <= max(batch_numbers):
                plt.axvline(x=epoch * batches_per_epoch, color='gray', 
                           linestyle='--', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(f'vicreg_{component}_loss.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_transform_results(result: np.ndarray, labels: np.ndarray, title: str, x_label: str, y_label: str):
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels == i
        plt.scatter(result[mask, 0], result[mask, 1], 
                   c=[colors[i]], label=class_names[i], 
                   alpha=0.6, s=20)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

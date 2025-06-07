from torchvision.datasets import CIFAR10
from augmentations import train_transform, test_transform
from dual_augment_data_set import DualAugmentDataSet
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


def plot_vicreg_losses(train_losses, test_losses, batch_numbers, test_epochs, batches_per_epoch):
    """Plot training and test losses for each VICReg component"""
    
    # Convert test epochs to batch numbers for plotting
    test_batch_numbers = [(epoch + 1) * batches_per_epoch for epoch in test_epochs]
    
    loss_components = ['invariance', 'variance', 'covariance']
    titles = ['Invariance Loss', 'Variance Loss', 'Covariance Loss']
    
    # Create combined figure with all three components
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (component, title) in enumerate(zip(loss_components, titles)):
        ax = axes[i]
        
        # Plot training losses
        ax.plot(batch_numbers, train_losses[component], 
                label='Training', alpha=0.7, linewidth=0.8)
        
        # Plot test losses
        ax.plot(test_batch_numbers, test_losses[component], 
                'ro-', label='Test', markersize=4, linewidth=2)
        
        ax.set_xlabel('Training Batch')
        ax.set_ylabel(f'{title}')
        ax.set_title(f'{title} vs Training Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines to show epoch boundaries
        for epoch in range(1, len(test_epochs) + 1):
            if epoch * batches_per_epoch <= max(batch_numbers):
                ax.axvline(x=epoch * batches_per_epoch, color='gray', 
                          linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('vicreg_loss_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create separate figures for each component
    for i, (component, title) in enumerate(zip(loss_components, titles)):
        plt.figure(figsize=(10, 6))
        
        # Plot training losses
        plt.plot(batch_numbers, train_losses[component], 
                label='Training', alpha=0.7, linewidth=0.8)
        
        # Plot test losses
        plt.plot(test_batch_numbers, test_losses[component], 
                'ro-', label='Test', markersize=4, linewidth=2)
        
        plt.xlabel('Training Batch')
        plt.ylabel(f'{title}')
        plt.title(f'{title} vs Training Batches')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines to show epoch boundaries
        for epoch in range(1, len(test_epochs) + 1):
            if epoch * batches_per_epoch <= max(batch_numbers):
                plt.axvline(x=epoch * batches_per_epoch, color='gray', 
                           linestyle='--', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(f'vicreg_{component}_loss.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_transform_results(result: np.ndarray, labels: np.ndarray):
    """Plot PCA results colored by CIFAR-10 classes"""
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot for each class
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels == i
        plt.scatter(result[mask, 0], result[mask, 1], 
                   c=[colors[i]], label=class_names[i], 
                   alpha=0.6, s=20)
    
    plt.title('PCA Visualization of VICReg Representations\nCIFAR-10 Test Set')
    plt.xlabel(f'First Principal Component')
    plt.ylabel(f'Second Principal Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

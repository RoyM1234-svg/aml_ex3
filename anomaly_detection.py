import torch
from vicreg_main import VICReg
from torchvision import datasets
from custom_datasets import NormalizedDataSet
from utils import create_normalized_data_loaders, extract_representations
from torch.utils.data import DataLoader
from augmentations import test_transform
import numpy as np
import faiss
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compute_knn_inverse_density(train_representations, test_representations, k=2):
    """
    Compute inverse density scores using kNN with FAISS.
    
    Args:
        train_representations: Reference representations from CIFAR10 training set (torch.Tensor)
        test_representations: Test representations (CIFAR10 + MNIST) (torch.Tensor)
        k: Number of nearest neighbors
        
    Returns:
        inverse_density_scores: Array of inverse density scores (average L2 distance to k neighbors)
    """
    train_representations = train_representations.cpu().numpy().astype(np.float32)
    test_representations = test_representations.cpu().numpy().astype(np.float32)
    
    dimension = train_representations.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(train_representations)
    
    distances, _ = index.search(test_representations, k)
    
    inverse_density_scores = np.mean(distances, axis=1)
    
    return inverse_density_scores


def Q1_helper(model: VICReg,
              device: torch.device,
              model_name: str,
              cifar10_train_loader: DataLoader,
              cifar10_test_loader: DataLoader,
              mnist_test_loader: DataLoader):
    
    encoder = model.encoder

    cifar10_train_representations, _ = extract_representations(encoder, cifar10_train_loader, device)
    cifar10_test_representations, _ = extract_representations(encoder, cifar10_test_loader, device)
    mnist_test_representations, _ = extract_representations(encoder, mnist_test_loader, device)

    test_representations = torch.cat([cifar10_test_representations, mnist_test_representations], dim=0)

    test_labels = np.concatenate([
        np.zeros(len(cifar10_test_representations), dtype=int),
        np.ones(len(mnist_test_representations), dtype=int)
    ])

    inverse_density_scores = compute_knn_inverse_density(
        cifar10_train_representations, 
        test_representations, 
        k=2
    )

    cifar10_scores = inverse_density_scores[:len(cifar10_test_representations)]
    mnist_scores = inverse_density_scores[len(cifar10_test_representations):]

    print(f"\n{model_name} Anomaly Detection Results:")
    print(f"Average inverse density score for CIFAR10 (normal): {cifar10_scores.mean():.4f}")
    print(f"Average inverse density score for MNIST (anomaly): {mnist_scores.mean():.4f}")

    results = {
        'train_representations': cifar10_train_representations,
        'test_representations': test_representations,
        'test_labels': test_labels,
        'inverse_density_scores': inverse_density_scores,
        'cifar10_scores': cifar10_scores,
        'mnist_scores': mnist_scores
    }
    
    return results
    

def Q3(vicreg_results, no_generated_neighbors_results):
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve for VICReg
    labels = vicreg_results['test_labels']
    scores = vicreg_results['inverse_density_scores']
    fpr1, tpr1, _ = roc_curve(labels, scores)
    auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, 'b-', label=f'VICReg (AUC = {auc1:.4f})')
    
    # Plot ROC curve for VICReg No Generated Neighbors
    labels = no_generated_neighbors_results['test_labels']
    scores = no_generated_neighbors_results['inverse_density_scores']
    fpr2, tpr2, _ = roc_curve(labels, scores)
    auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, 'r-', label=f'VICReg No Gen Neighbors (AUC = {auc2:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Anomaly Detection')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    
    # Print AUC values
    print(f"\nAUC Values:")
    print(f"VICReg: {auc1:.4f}")
    print(f"VICReg No Generated Neighbors: {auc2:.4f}")
    print(f"Difference: {auc1 - auc2:.4f}")

def Q1():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    vicreg_model = VICReg(device=device_str)
    vicreg_model.load_state_dict(torch.load('vicreg_model.pth', map_location=device))
    vicreg_model.to(device)

    cifar10_train_loader, cifar10_test_loader = create_normalized_data_loaders(shuffle_train=False)

    mnist_preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.Lambda(lambda x: x.convert('RGB'))
    ])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                               transform=mnist_preprocess)
    mnist_test_dataset = NormalizedDataSet(mnist_test, test_transform)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=256, shuffle=False)

    vicreg_results = Q1_helper(vicreg_model, device, "VICReg", cifar10_train_loader, cifar10_test_loader, mnist_test_loader)

    no_generated_neighbors_model = VICReg(device=device_str)
    no_generated_neighbors_model.load_state_dict(torch.load('vicreg_model_no_generated_neighbors.pth', map_location=device))
    no_generated_neighbors_model.to(device)

    no_generated_neighbors_results = Q1_helper(no_generated_neighbors_model, device, "VICReg (no generated neighbors)", cifar10_train_loader, cifar10_test_loader, mnist_test_loader)

    return vicreg_results, no_generated_neighbors_results
def main():
    vicreg_results, no_generated_neighbors_results = Q1()
    

if __name__ == "__main__":
    main()
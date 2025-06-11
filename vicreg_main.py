import torch
from tqdm import tqdm
from custom_datasets import NeighborPairDataset, NormalizedDataSet
from models import VICReg
from utils import choose_image_from_each_class, create_normalized_data_loaders, create_data_for_veicreg, plot_vicreg_losses, plot_transform_results
from vicreg_loss import VICRegLoss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from typing import Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
from augmentations import test_transform
from torchvision.transforms import ToTensor
from utils import plot_images_with_neighbors
import faiss


def train_model(lambda_=25.0, mu=25.0, nu=1.0, gamma=1.0, epsilon=1e-4):
    """
    lambda_ = 25.0        # invariance loss weight (λ)
    mu = 25.0           # variance loss weight (μ) 
    nu = 1.0            # covariance loss weight (ν)
    gamma = 1.0         # variance threshold (γ)
    epsilon = 1e-4      # numerical stability (ε)
    """
    
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 30  
    projection_dim = 512
    encoder_dim = 128

    betas = (0.9, 0.999)
    weight_decay = 1e-6

    train_loader, test_loader = create_data_for_veicreg(batch_size=batch_size)

    loss_fn = VICRegLoss(lambda_, mu, nu, gamma, epsilon)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = VICReg(encoder_dim, projection_dim, device_str)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=betas, 
        weight_decay=weight_decay
    )

    train_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    test_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    test_epochs = []
    batch_numbers = []
    
    global_batch = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (x1, x2, y) in enumerate(train_bar):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss, inv, var, cov = loss_fn(z1, z2)
            loss.backward()
            optimizer.step()
            
            train_losses['total'].append(loss.item())
            train_losses['invariance'].append(inv.item())
            train_losses['variance'].append(var.item())
            train_losses['covariance'].append(cov.item())
            batch_numbers.append(global_batch)
            global_batch += 1

        test_total, test_inv, test_var, test_cov, test_epoch = evaluate_vicreg_model(model, test_loader, device, loss_fn, epoch)
        test_losses['total'].append(test_total)
        test_losses['invariance'].append(test_inv)
        test_losses['variance'].append(test_var)
        test_losses['covariance'].append(test_cov)
        test_epochs.append(test_epoch)
    
    plot_vicreg_losses(train_losses, test_losses, batch_numbers, test_epochs, len(train_loader))

    return model
    
def evaluate_vicreg_model(model, loader, device, loss_fn, epoch):
    model.eval()

    test_total, test_inv, test_var, test_cov = 0, 0, 0, 0
    
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss, inv, var, cov = loss_fn(z1, z2)
            
            test_total += loss.item()
            test_inv += inv.item()
            test_var += var.item()
            test_cov += cov.item()
    
    # Average test losses
    total = (test_total / len(loader))
    invariance = (test_inv / len(loader))
    variance = (test_var / len(loader))
    covariance = (test_cov / len(loader))
    
    
    print(f"Epoch {epoch+1} - Test Loss: {total:.4f}, "
        f"Test Inv: {invariance:.4f}, "
        f"Test Var: {variance:.4f}, "
        f"Test Cov: {covariance:.4f}")

    return total, invariance, variance, covariance, epoch

def apply_pca(representations: np.ndarray, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(representations)
    return pca_result

def Q2():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    model = VICReg(device=device_str)
    model.load_state_dict(torch.load('vicreg_model.pth', map_location=device))

    encoder = model.encoder
    encoder.to(device)

    _, test_loader = create_data_for_veicreg()

    test_representations, labels = extract_vicreg_representations(encoder, test_loader, device)

    pca_result = apply_pca(test_representations)

    plot_transform_results(pca_result, labels, 'PCA Visualization of VICReg Representations\nCIFAR-10 Test Set', 'First Principal Component', 'Second Principal Component')

    tsne_result = apply_tsne(test_representations)
    plot_transform_results(tsne_result, labels, 't-SNE Visualization of VICReg Representations\nCIFAR-10 Test Set', 'First t-SNE Component', 'Second t-SNE Component')

def extract_vicreg_representations(encoder, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    representations = []
    labels = []
    with torch.no_grad():
        for x, _, label in tqdm(loader):
            x = x.to(device)
            y = encoder(x)
            representations.append(y.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(representations, axis=0), np.concatenate(labels, axis=0)

def apply_tsne(representations):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(representations)
    return tsne_result

def train_linear_probing_model(encoder_dim: int, train_representations, train_labels, test_representations, test_labels, device):
    num_epochs = 30
    lr = 0.01
    momentum = 0.9
    batch_size = 256
    num_classes = 10

    classifier = nn.Linear(encoder_dim, num_classes).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(train_representations, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_representations, test_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        classifier.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for representations, labels in train_bar:
            representations, labels = representations.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(representations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss.item()}")

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for representations, labels in test_loader:
                representations, labels = representations.to(device), labels.to(device)
                outputs = classifier(representations)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Accuracy: {correct / total}")

    return classifier
                       
def Q3():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = VICReg(device=device_str)
    model.load_state_dict(torch.load('vicreg_model.pth', map_location=device))
    model.to(device)

    train_loader, test_loader = create_normalized_data_loaders()

    encoder = model.encoder
    encoder.to(device)

    train_representations, train_labels = extract_representations(encoder, train_loader, device)
    test_representations, test_labels = extract_representations(encoder, test_loader, device)

    train_linear_probing_model(
        encoder.get_encoder_dim(), train_representations, train_labels, test_representations, test_labels, device)
        
def extract_representations(encoder, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    representations = []
    labels = []
    with torch.no_grad():
        for x, label in tqdm(loader):
            x = x.to(device)
            y = encoder(x)
            representations.append(y)
            labels.append(label)
    return torch.cat(representations, dim=0), torch.cat(labels, dim=0)

def train_model_with_no_variance_loss():
    model = train_model(mu=0.0)
    return model

def Q4():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    model = VICReg(device=device_str)
    model.load_state_dict(torch.load('vicreg_model_no_variance_loss.pth', map_location=device))

    encoder = model.encoder
    encoder.to(device)

    _, pca_test_loader = create_data_for_veicreg()

    np_test_representations, np_test_labels = extract_vicreg_representations(encoder, pca_test_loader, device)
    pca_result = apply_pca(np_test_representations)
    plot_transform_results(pca_result, np_test_labels, 
                           'PCA Visualization of VICReg Representations\nCIFAR-10 Test Set with No Variance Loss',
                           'First Principal Component',
                           'Second Principal Component')
    

    train_loader, test_loader = create_normalized_data_loaders()
    train_representations, train_labels = extract_representations(encoder, train_loader, device)
    test_representations, test_labels = extract_representations(encoder, test_loader, device)
    train_linear_probing_model(encoder.get_encoder_dim(),
                                train_representations,
                                train_labels,
                                test_representations,
                                test_labels,
                                device)
    
def train_model_no_generated_neighbors(dataset: NormalizedDataSet,neighbor_indices: torch.Tensor, device: torch.device, device_str: str) -> VICReg:
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 1
    projection_dim = 512
    encoder_dim = 128

    betas = (0.9, 0.999)
    weight_decay = 1e-6

    loss_fn = VICRegLoss(lambda_=25.0, mu=25.0, nu=1.0, gamma=1.0, epsilon=1e-4)

    model = VICReg(encoder_dim, projection_dim, device_str)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=betas, 
        weight_decay=weight_decay
    )

    neighbor_dataset = NeighborPairDataset(dataset, neighbor_indices)
    train_loader = DataLoader(neighbor_dataset, batch_size=batch_size, shuffle=True)
    

    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (base_image, neighbor_image) in enumerate(train_bar):
            base_image, neighbor_image = base_image.to(device), neighbor_image.to(device)
            optimizer.zero_grad()
            _, z1 = model(base_image)
            _, z2 = model(neighbor_image)
            loss, _, _, _ = loss_fn(z1, z2)
            loss.backward()
            optimizer.step()
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item()}")

    return model
            
def Q5():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = VICReg(device=device_str)
    model.load_state_dict(torch.load('vicreg_model.pth', map_location=device))

    encoder = model.encoder
    encoder.to(device)

    train_loader, test_loader = create_normalized_data_loaders()
    train_data_set: NormalizedDataSet = train_loader.dataset # type: ignore

    train_representations, _ = extract_representations(encoder, train_loader, device)
    
    neighbor_indices = create_neighbors_array(train_representations)

    del model

    no_generated_neighbors_model = train_model_no_generated_neighbors(train_data_set, neighbor_indices, device, device_str)
    torch.save(no_generated_neighbors_model.state_dict(), 'vicreg_model_no_generated_neighbors.pth')

    no_generated_neighbors_model = VICReg(device=device_str)
    no_generated_neighbors_model.load_state_dict(torch.load('vicreg_model_no_generated_neighbors.pth', map_location=device))
    no_generated_neighbors_model.to(device)
    
    
    train_representations, train_labels = extract_representations(no_generated_neighbors_model.encoder, train_loader, device)
    test_representations, test_labels = extract_representations(no_generated_neighbors_model.encoder, test_loader, device)
    classifier = train_linear_probing_model(no_generated_neighbors_model.encoder.get_encoder_dim(),
                                train_representations,
                                train_labels,
                                test_representations,
                                test_labels,
                                device)
    torch.save(classifier.state_dict(), 'linear_probing_model_no_generated_neighbors.pth')
    
def create_neighbors_array(representations: torch.Tensor, k: int = 3) -> torch.Tensor:
    representations_np = representations.detach().cpu().numpy().astype('float32')
    index = faiss.IndexFlatL2(representations_np.shape[1])
    index.add(representations_np)
    _, indices = index.search(representations_np, k+1)
    neighbor_indices = indices[:, 1:4] 

    return torch.from_numpy(neighbor_indices)

def Q7():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    images, labels = choose_image_from_each_class()
    vicreg_model = VICReg(device=device_str)
    vicreg_model.load_state_dict(torch.load('vicreg_model.pth', map_location=device))
    vicreg_model.to(device)

    train_loader, _ = create_normalized_data_loaders()
    train_representations, _ = extract_representations(vicreg_model.encoder, train_loader, device)

    images_dataset = torch.stack([test_transform(ToTensor()(image)) for image in images])
    with torch.no_grad():
        images_dataset = images_dataset.to(device)
        images_representations, _ = vicreg_model(images_dataset)

    indices = find_nearest_neighbors(images_representations, train_representations)
    
    nearest_neighbors = []
    train_dataset = train_loader.dataset
    for i in range(len(images)):
        nearest_neighbors.append([train_dataset.get_image_by_index(index) for index in indices[i]]) # type: ignore
    
    
    plot_images_with_neighbors(images, nearest_neighbors)
    



def find_nearest_neighbors(query_representations: torch.Tensor, reference_representations: torch.Tensor, k: int = 5):
    distances = torch.cdist(query_representations, reference_representations)
    _, indices = distances.topk(k, dim=1, largest=False)
    return indices
    
    

    

def main():
    # # train model
    # model = train_model()
    # torch.save(model.state_dict(), 'vicreg_model.pth')

    # Q2()
    # Q3()

    # model = train_model_with_no_variance_loss()
    # torch.save(model.state_dict(), 'vicreg_model_no_variance_loss.pth')

    # Q4()
    # Q5()

    Q7()

    
    
    
    
    

    

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from augmentations import train_transform, test_transform
from models import VICReg
from utils import create_data_for_veicreg


def train_model():
    train_loader, test_loader = create_data_for_veicreg()

    # VICReg Hyperparameters
    lambda_ = 25.0        # invariance loss weight (λ)
    mu = 25.0           # variance loss weight (μ) 
    nu = 1.0            # covariance loss weight (ν)
    gamma = 1.0         # variance threshold (γ)
    epsilon = 1e-4      # numerical stability (ε)

    # Training Hyperparameters
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 30     
    projection_dim = 512
    encoder_dim = 128

    # Optimizer parameters
    betas = (0.9, 0.999)
    weight_decay = 1e-6

    model = VICReg(encoder_dim, projection_dim)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=betas, 
        weight_decay=weight_decay
    )

    loss_fn = torch.nn.MSELoss()

    
    

def main():
    print("hello world")

if __name__ == "__main__":
    main()
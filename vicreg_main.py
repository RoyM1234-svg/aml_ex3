import torch
from tqdm import tqdm
from models import VICReg
from utils import create_data
from vicreg_loss import VICRegLoss


def train_model():
    
    lambda_ = 25.0        # invariance loss weight (λ)
    mu = 25.0           # variance loss weight (μ) 
    nu = 1.0            # covariance loss weight (ν)
    gamma = 1.0         # variance threshold (γ)
    epsilon = 1e-4      # numerical stability (ε)

    
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 1
    projection_dim = 512
    encoder_dim = 128

    betas = (0.9, 0.999)
    weight_decay = 1e-6

    train_loader, test_loader = create_data(batch_size=batch_size)

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

    # train_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    # test_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    
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

    
    

def main():
    train_model()

if __name__ == "__main__":
    main()
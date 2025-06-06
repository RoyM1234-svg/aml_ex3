import torch
from tqdm import tqdm
from models import VICReg
from utils import create_data, plot_vicreg_losses
from vicreg_loss import VICRegLoss


def train_model():
    
    lambda_ = 25.0        # invariance loss weight (λ)
    mu = 25.0           # variance loss weight (μ) 
    nu = 1.0            # covariance loss weight (ν)
    gamma = 1.0         # variance threshold (γ)
    epsilon = 1e-4      # numerical stability (ε)

    
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 3  # Increased epochs for better plotting
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

    train_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    
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

        test_losses, test_epochs = evaluate_model(model, test_loader, device, loss_fn, epoch)
    
    plot_vicreg_losses(train_losses, test_losses, batch_numbers, test_epochs, len(train_loader))
    

def evaluate_model(model, loader, device, loss_fn, epoch):
    test_losses = {'total': [], 'invariance': [], 'variance': [], 'covariance': []}
    test_epochs = []
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
    test_losses['total'].append(test_total / len(loader))
    test_losses['invariance'].append(test_inv / len(loader))
    test_losses['variance'].append(test_var / len(loader))
    test_losses['covariance'].append(test_cov / len(loader))
    test_epochs.append(epoch)
    
    print(f"Epoch {epoch+1} - Test Loss: {test_losses['total'][-1]:.4f}, "
        f"Test Inv: {test_losses['invariance'][-1]:.4f}, "
        f"Test Var: {test_losses['variance'][-1]:.4f}, "
        f"Test Cov: {test_losses['covariance'][-1]:.4f}")

    return test_losses, test_epochs

def main():
    train_model()

if __name__ == "__main__":
    main()
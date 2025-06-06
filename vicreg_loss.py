import torch
import torch.nn as nn

class VICRegLoss(nn.Module):
    def __init__(self, lambda_, mu, nu, gamma, epsilon):
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon

    def invariance_loss(self, z1, z2) -> torch.Tensor:
        return ((z1 - z2) ** 2).mean(dim=1).mean()


    def variance_loss(self, z) -> torch.Tensor:
        variances = torch.var(z, dim=0, unbiased=False)
        stds = torch.sqrt(variances + self.epsilon)
        return torch.clamp(self.gamma - stds, min=0.0).mean()
          

    def covariance_loss(self, z) -> torch.Tensor:
        z_mean = torch.mean(z, dim=0)
        z_centered = z - z_mean
        cov_mat = z_centered.T @ z_centered / (z_centered.shape[0] - 1)
        cov_squared = cov_mat ** 2
        return (cov_squared.sum() - cov_squared.diag().sum()) / cov_mat.shape[0]
        

    def forward(self, z1, z2):
        inv = self.lambda_ * self.invariance_loss(z1, z2)
        var = self.mu * (self.variance_loss(z1) + self.variance_loss(z2))
        cov = self.nu * (self.covariance_loss(z1) + self.covariance_loss(z2))
        loss = inv + var + cov
        return loss, inv, var, cov
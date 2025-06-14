import torch
import torch.nn as nn
from torchvision.models import resnet18


class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.D = D
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)
    
    def get_encoder_dim(self):
        return self.D


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)
    

class VICReg(nn.Module):
    def __init__(self, encoder_dim = 128, projector_dim = 512, device='cuda'):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.projector_dim = projector_dim
        self.encoder = Encoder(encoder_dim, device)
        self.projector = Projector(encoder_dim, projector_dim)

        
    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        return y, z


class LinearProbing(nn.Module):
    def __init__(self, encoder: Encoder, device='cuda', num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.get_encoder_dim(), num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            y = self.encoder(x)
        y = self.classifier(y)
        return y
    
    def get_encoder(self):
        return self.encoder

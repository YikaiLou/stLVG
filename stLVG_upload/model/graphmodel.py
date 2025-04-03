"""
Spatio-Temporal Lightweight Vector Graph Network(stLVG) Implementation

Contains:
- Graph Convolutional Network variants
- GAN components
- Contrastive learning modules
"""

import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graphnet.AGF_Com import CombGaussian_F
from .graphnet.combnet import CombUnweighted

class LGCN(torch.nn.Module):
    r"""
    Lightweight Graph Convolutional Network (LGCN) with layer-wise feature concatenation
    
    Architecture: Z = f_e(A, X) = Concat([X, AX, A^2X, ..., A^KX])W_e
    
    Parameters:
        input_size (int): Dimension of input features
        K (int, optional): Number of propagation layers (default: 8)
    """
    def __init__(self, input_size: int, K: Optional[int] = 8):
        super(LGCN, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        
    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through LGCN layers"""
        x = self.conv1(feature, edge_index)
        return x

class LGCN_mlp(torch.nn.Module):
    r"""
    LGCN with MLP projection head
    
    Parameters:
        input_size (int): Input feature dimension
        output_size (int): Output embedding dimension
        K (int, optional): Number of GCN layers (default: 8)
        hidden_size (int, optional): MLP hidden layer dimension (default: 512)
        dropout (float, optional): Dropout probability (default: 0.2)
    """
    def __init__(self, input_size: int, output_size: int, K: Optional[int] = 8,
                 hidden_size: Optional[int] = 512, dropout: Optional[float] = 0.2):
        super(LGCN_mlp, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """Forward pass with norm return"""
        x, norm = self.conv1(feature, edge_index)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x, norm

class LGCN_mlp_AGF(torch.nn.Module):
    r"""
    LGCN with Azimuthal Gaussian Filter (AGF) and MLP
    
    Parameters:
        input_size (int): Input feature dimension
        output_size (int): Output embedding dimension  
        coordinates (torch.Tensor): Node coordinate matrix
        m (int): Azimuthal harmonic order
        K (int): Number of propagation layers (default: 8)
        hidden_size (int): MLP hidden dimension (default: 512)
        dropout (float): Dropout probability (default: 0.2)
    """
    def __init__(self, input_size: int, output_size: int, coordinates: torch.Tensor, 
                 m: int, K: int = 8, hidden_size: int = 512, dropout: float = 0.2):
        super(LGCN_mlp_AGF, self).__init__()
        self.conv1 = CombGaussian_F(coordinates=coordinates, m=m, K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor, 
               coor: torch.Tensor, m: int) -> torch.Tensor:
        """Forward pass with spatial coordinates"""
        x = self.conv1(feature, edge_index, coor, m)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class LGCN_AGF_norm(torch.nn.Module):
    
    def __init__(self, input_size:int, output_size:int, coordinates, m:int, K: int = 8,
                 hidden_size: int = 512, dropout: float = 0.2):
        super(LGCN_AGF_norm, self).__init__()
        self.conv1 = CombGaussian_F(coordinates= coordinates, m = m, K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor, coor:torch.Tensor, m:int):
        x, norm = self.conv1(feature, edge_index, coor, m) #norm
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x, norm # norm

class WDiscriminator(torch.nn.Module):
    r"""
    Wasserstein GAN Discriminator
    
    Parameters:
        hidden_size (int): Input dimension
        hidden_size2 (int, optional): Hidden layer dimension (default: 512)
    """
    def __init__(self, hidden_size: int, hidden_size2: int = 512):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
        
    def forward(self, input_embd: torch.Tensor) -> torch.Tensor:
        """Forward pass for discrimination"""
        x = F.leaky_relu(self.hidden(input_embd), 0.2)
        x = F.leaky_relu(self.hidden2(x), 0.2)
        return self.output(x)

class transformation(torch.nn.Module):
    r"""
    Linear Transformation Layer
    
    Parameters:
        hidden_size (int): Dimension of input embeddings
    """
    def __init__(self, hidden_size: int = 512):
        super(transformation, self).__init__()
        self.trans = torch.nn.Parameter(torch.eye(hidden_size))
        
    def forward(self, input_embd: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation"""
        return input_embd.mm(self.trans)

class ReconDNN(torch.nn.Module):
    r"""
    Feature Reconstruction Network
    
    Parameters:
        hidden_size (int): Input embedding dimension
        feature_size (int): Original feature dimension
        hidden_size2 (int, optional): Hidden layer dimension (default: 512)
    """
    def __init__(self, hidden_size: int, feature_size: int, hidden_size2: int = 512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, feature_size)
        
    def forward(self, input_embd: torch.Tensor) -> torch.Tensor:
        """Reconstruct original features"""
        return self.output(F.relu(self.hidden(input_embd)))

class Contrast(nn.Module):
    r"""
    Contrastive Learning Module
    
    Parameters:
        hidden_dim (int): Embedding dimension
        tau (float): Temperature parameter
    """
    def __init__(self, hidden_dim: int, tau: float):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Calculate similarity matrix with temperature scaling"""
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        return torch.exp(dot_numerator / dot_denominator / self.tau)

    def forward(self, embd0: torch.Tensor, embd1: torch.Tensor, 
               pos: torch.Tensor) -> tuple:
        """Calculate contrastive loss"""
        embd0_proj = self.proj(embd0)
        embd1_proj = self.proj(embd1)

        matrix = self.sim(embd0_proj, embd1_proj).clone()
        matrix_norm = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8)
        loss = -torch.log((matrix_norm * pos).sum(dim=-1) + 1e-8).mean()
        return embd0_proj, embd1_proj, loss

# Utility Components
class notrans(torch.nn.Module):
    """Identity Transformation Layer"""
    def __init__(self):
        super(notrans, self).__init__()
        
    def forward(self, input_embd: torch.Tensor) -> torch.Tensor:
        """Return input unchanged"""
        return input_embd
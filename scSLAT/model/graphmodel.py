r"""
Graph and GAN networks in SLAT
"""
import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scSLAT.model.graphconv.CBank import CombGaussian
from scSLAT.model.graphconv.comBanksy import CombBanksy
from scSLAT.model.graphconv.AGF_Com import CombGaussian_F

from .graphconv import CombUnweighted

        
class LGCN(torch.nn.Module):
    r"""
    Lightweight GCN which remove nonlinear functions and concatenate the embeddings of each layer:
        (:math:`Z = f_{e}(A, X) = Concat( [X, A_{X}, A_{2X}, ..., A_{KX}])W_{e}`)
    
    Parameters
    ----------
    input_size
        input dim
    K
        LGCN layers
    """
    def __init__(self, input_size:int, K:Optional[int]=8):
        super(LGCN, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor):
        x = self.conv1(feature, edge_index)
        return x


class LGCN_mlp(torch.nn.Module):
    r"""
    Add one hidden layer MLP in LGCN
    
    Parameters
    ----------
    input_size
        input dim
    output_size
        output dim
    K
        LGCN layers
    hidden_size
        hidden size of MLP
    dropout
        dropout ratio
    """
    def __init__(self, input_size:int, output_size:int, K:Optional[int] = 8,
                 hidden_size:Optional[int] = 512, dropout:Optional[int]=0.2):
        super(LGCN_mlp, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor):
        x, norm = self.conv1(feature, edge_index) #norm
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x ,norm #norm



class LGCN_mlp2(torch.nn.Module):
    
    def __init__(self, input_size:int, output_size:int, coordinates, K:int=8,
                 hidden_size:int = 512, dropout:float = 0.2):
        super(LGCN_mlp2, self).__init__()
        # self.coordinates = coordinates
        self.conv1 = CombGaussian(coordinates= self.coordinates , K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor, coor:torch.Tensor):
        x = self.conv1(feature, edge_index, coor)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class LGCN_mlp_AGF(torch.nn.Module):
    
    def __init__(self, input_size:int, output_size:int, coordinates, m:int, K: int = 8,
                 hidden_size: int = 512, dropout: float = 0.2):
        super(LGCN_mlp_AGF, self).__init__()
        self.conv1 = CombGaussian_F(coordinates= coordinates, m = m, K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

        
    def forward(self, feature:torch.Tensor, edge_index:torch.Tensor, coor:torch.Tensor, m:int):
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
# class DimensionalityReductionModel(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(DimensionalityReductionModel, self).__init__()
#         self.encoder = torch.nn.Linear(input_dim, hidden_dim)
#         self.decoder = torch.nn.Linear(hidden_dim, output_dim)
#         self.activation = torch.nn.ReLU()

#     def forward(self, x):
#         encoded = self.activation(self.encoder(x))
#         decoded = self.decoder(encoded)
#         return encoded, decoded

class DimensionalityReductionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DimensionalityReductionModel, self).__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        return encoded
    

class WDiscriminator(torch.nn.Module):
    r"""
    WGAN Discriminator
    
    Parameters
    ----------
    hidden_size
        input dim
    hidden_size2
        hidden dim
    """
    def __init__(self, hidden_size:int, hidden_size2:int = 512):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
    def forward(self, input_embd):
        x = F.leaky_relu(self.hidden(input_embd), 0.2)
        x = F.leaky_relu(self.hidden2(x), 0.2)
        return self.output(x)

class transformation(torch.nn.Module):
    r"""
    Transformation in LGCN
    
    Parameters
    ----------
    hidden_size
        input dim
    """
    def __init__(self, hidden_size:int = 512):
        super(transformation, self).__init__()
        self.trans = torch.nn.Parameter(torch.eye(hidden_size))
    def forward(self, input_embd):
        return input_embd.mm(self.trans)


class notrans(torch.nn.Module):
    r"""
    LGCN without transformation
    """
    def __init__(self):
        super(notrans, self).__init__()
    def forward(self, input_embd:torch.Tensor):
        return input_embd


class ReconDNN(torch.nn.Module):
    r"""
    Data reconstruction network
    
    Parameters
    ----------
    hidden_size
        input dim
    feature_size
        output size (feature input size)
    hidden_size2
        hidden size
    """
    def __init__(self, hidden_size:int, feature_size:int, hidden_size2: int = 512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, feature_size)
    def forward(self, input_embd:torch.Tensor):
        return self.output(F.relu(self.hidden(input_embd)))


class AttentionWeighted(torch.nn.Module):
    def __init__(self, NFeature, alpha = 0.2):
        super(AttentionWeighted, self).__init__()
        self.NFeature  = NFeature
        self.alpha     = alpha
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.a1        = nn.Parameter(torch.empty(size=(2*NFeature, 1)))
        self.a2        = nn.Parameter(torch.empty(size=(2*NFeature, 1)))

        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def inference( self, h1 = None, h2 = None):
          
        e         = self.prepare_attentional_input(h1, h2)
        lamda = torch.nn.functional.softmax(e, dim=1)
        # lamda     = F.dropout(lamda, self.dropout, training=self.training)

        h_prime1 = lamda[:, 0:1] * h1  
        h_prime2 = lamda[:, 1:2] * h2 

        h_robust = h_prime1 + h_prime2
        h_combine = torch.cat([h_prime1, h_prime2], dim=1)
        return lamda, h_combine


    def forward(self, h1 = None, h2 = None):

        lamda, h_robust = self.inference(h1, h2)

        return lamda, h_robust

    def prepare_attentional_input(self, h1=None, h2=None):
         h_cat = torch.cat((h1.clone(), h2.clone()), dim = 1)

         Wh1 = torch.matmul(h_cat, self.a1.clone())
         Wh2 = torch.matmul(h_cat, self.a2.clone())

         e = torch.cat([Wh1, Wh2], dim = 1)

         return self.leakyrelu(e)



class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau):
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

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        # print("Sim Matrix:", sim_matrix)
        return sim_matrix

    def forward(self, embd0, embd1, pos):
        embd0_proj    = self.proj(embd0)
        embd1_proj    = self.proj(embd1)

        matrix = self.sim(embd0_proj, embd1_proj).clone()
        matrix_norm = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8)
        loss = -torch.log((matrix_norm * pos).sum(dim=-1) + 1e-8).mean()
        return embd0_proj, embd1_proj, loss

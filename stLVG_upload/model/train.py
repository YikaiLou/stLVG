r"""
Training functions with different strategy
"""
from math import ceil
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from .loss import feature_reconstruct_loss

# Adversarial distance training
def train_GAN(wdiscriminator:torch.nn.Module,
                optimizer_d:torch.optim.Optimizer,
                embds:List[torch.Tensor],
                batch_d_per_iter:Optional[int]=5,  # Number of discriminator training steps per iteration
                anchor_scale:Optional[float]=0.8  # Ratio of anchor cells to select
    )->torch.Tensor:
    r"""
    GAN training strategy using Wasserstein distance
    
    Parameters
    ----------
    wdiscriminator
        Wasserstein GAN discriminator
    optimizer_d
        Optimizer for WGAN parameters
    embds
        List of graph embeddings from LGCN [embd0, embd1]
    batch_d_per_iter
        Number of discriminator updates per training iteration
    anchor_scale
        Proportion of cells to use as anchors
    """
    embd0, embd1 = embds
    
    wdiscriminator.train()
    anchor_size = ceil(embd1.size(0)*anchor_scale)

    for j in range(batch_d_per_iter):
        w0 = wdiscriminator(embd0)
        w1 = wdiscriminator(embd1)
        # Select top and bottom anchors based on discriminator scores
        anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
        anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
        embd0_anchor = embd0[anchor0, :].clone().detach()
        embd1_anchor = embd1[anchor1, :].clone().detach()
        # Update discriminator with gradient penalty
        optimizer_d.zero_grad()
        critic_loss = -torch.mean(wdiscriminator(embd0_anchor)) + torch.mean(wdiscriminator(embd1_anchor))
        critic_loss.backward()
        optimizer_d.step()
        # Apply weight clipping for Lipschitz constraint
        for p in wdiscriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
    # Calculate final generator loss
    w0 = wdiscriminator(embd0)
    w1 = wdiscriminator(embd1)
    anchor1 = w1.view(-1).argsort(descending=True)[: anchor_size]
    anchor0 = w0.view(-1).argsort(descending=False)[: anchor_size]
    embd0_anchor = embd0[anchor0, :]
    embd1_anchor = embd1[anchor1, :]
    loss = -torch.mean(wdiscriminator(embd1_anchor))
    return loss

# Train reconstruction models to recover features from embeddings
def train_reconstruct(recon_models:torch.nn.Module,
                        optimizer_recons,
                        embds:List[torch.Tensor],
                        features:List[torch.Tensor],
                        batch_r_per_iter:Optional[int]=10
    )->torch.Tensor:
    r"""
    Feature reconstruction training strategy
    
    Parameters
    ----------
    recon_models
        Pair of reconstruction models [model0, model1]
    optimizer_recons
        Optimizers for reconstruction models [optim0, optim1]
    embds
        List of graph embeddings [embd0, embd1]
    features
        Original node features for reconstruction [features0, features1]
    batch_r_per_iter
        Number of reconstruction steps per training iteration
    """
    recon_model0, recon_model1 = recon_models
    optimizer_recon0, optimizer_recon1 = optimizer_recons
    embd0, embd1 = embds
    
    recon_model0.train()
    recon_model1.train()
    # Detach embeddings for stable reconstruction training
    embd0_copy = embd0.clone().detach()
    embd1_copy = embd1.clone().detach()    
    # Update first reconstruction model
    for t in range(batch_r_per_iter):
        optimizer_recon0.zero_grad()
        loss = feature_reconstruct_loss(embd0_copy, features[0], recon_model0)
        loss.backward()
        optimizer_recon0.step()
    # Update second reconstruction model
    for t in range(batch_r_per_iter):
        optimizer_recon1.zero_grad()
        loss = feature_reconstruct_loss(embd1_copy, features[1], recon_model1)
        loss.backward()
        optimizer_recon1.step()
    # Calculate combined reconstruction loss
    loss = 0.5 * feature_reconstruct_loss(embd0, features[0], recon_model0) + 0.5 * feature_reconstruct_loss(embd1, features[1], recon_model1)

    return loss

# Evaluate embedding alignment accuracy using ground truth
def check_align(embds:List[torch.Tensor],
                ground_truth:torch.Tensor,
                k:Optional[int]=[5,10],
                mode:Optional[str]='cosine'
    )->List[float]:
    r"""
    Evaluate embedding alignment accuracy under ground truth mapping
    
    Parameters
    -----------
    embds
        Pair of graph embeddings [embd0, embd1]
    ground_truth
        Ground truth mapping matrix (2, num_nodes)
    k
        List of top-k values to evaluate [k1, k2] where k2 > k1
    mode
        Distance metric for similarity calculation
    """
    embd0, embd1 = embds
    assert k[1] > k[0]
    # Create ground truth mapping dictionary
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    
    # Calculate cosine similarity matrix
    cossim = torch.zeros(embd1.size(0), embd0.size(0))
    for i in range(embd1.size(0)):
        cossim[i] = F.cosine_similarity(embd0, embd1[i:i+1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
    
    # Get top-k predictions
    ind = cossim.argsort(dim=1, descending=True)[:, :k[1]]
    a1 = 0
    ak0 = 0
    ak1 = 0
    # Calculate hit rates at different k levels
    for i, node in enumerate(g_list):
        if ind[node, 0].item() == g_map[node]:
            a1 += 1
            ak0 += 1
            ak1 += 1
        else:
            # Check in k0 range
            for j in range(1, k[0]):
                if ind[node, j].item() == g_map[node]:
                    ak0 += 1
                    ak1 += 1
                    break
                else:
                    # Check in k1 range
                    for l in range(k[0], k[1]):
                        if ind[node, l].item() == g_map[node]:
                            ak1 += 1
                        break

    # Normalize scores
    a1 /= len(g_list)
    ak0 /= len(g_list)
    ak1 /= len(g_list)
    print(f'H@1:{a1*100}; H@{k[0]}:{ak0*100}; H@{k[1]}:{ak1*100}')
    return a1, ak0, ak1
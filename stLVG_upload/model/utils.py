r"""
Useful functions
"""
import time
import math
from typing import List, Mapping, Optional, Union
from pathlib import Path
from joblib import Parallel, delayed

import faiss
import scanpy as sc
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData
from typing import Iterable
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse

from ..utils import get_free_gpu
from .train import train_GAN, train_reconstruct
from .graphmodel import LGCN, LGCN_AGF_norm, LGCN_mlp, LGCN_mlp_AGF, WDiscriminator, ReconDNN, Contrast
from .loaddata import load_anndatas
from .preprocess import Cal_Spatial_Net

def run_LGCN(features:List,
            edges:List,
            LGCN_layer:Optional[int]=2
    ):
    """
    Run LGCN model
    
    Parameters
    ----------
    features
        list of graph node features
    edges
        list of graph edges
    LGCN_layer
        LGCN layer number, we suggest set 2 for barcode based and 4 for fluorescence based
    """
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)
    
    LGCN_model =LGCN(input_size=features[0].size(1), K=LGCN_layer).to(device=device)
    
    time1 = time.time()
    embd0 = LGCN_model(features[0], edges[0])
    embd1 = LGCN_model(features[1], edges[1])
    
    run_time = time.time() - time1
    print(f'LGCN time: {run_time}')
    return embd0, embd1, run_time   

def run_SLAT(features:List,
            edges:List,
            epochs:Optional[int]=6,
            LGCN_layer:Optional[int]=1,
            mlp_hidden:Optional[int]=256,
            hidden_size:Optional[int]=2048,
            alpha:Optional[float]=0.01,
            anchor_scale:Optional[float]=0.8,
            lr_mlp:Optional[float]=0.0001,
            lr_wd:Optional[float]=0.0001,
            lr_recon:Optional[float]=0.01,
            batch_d_per_iter:Optional[int]=5,
            batch_r_per_iter:Optional[int]=10
    ) -> List:
    r"""
    Run SLAT model
    
    Parameters
    ----------
    features
        list of graph node features
    edges
        list of graph edges
    epochs
        epoch number of SLAT (not exceed 10)
    LGCN_layer
        LGCN layer number, we suggest set 1 for barcode based and 4 for fluorescence based
    mlp_hidden
        MLP hidden layer size
    hidden_size
        size of LGCN output
    transform
        if use transform
    alpha
        scale of loss
    anchor_scale
        ratio of cells selected as pairs
    lr_mlp
        learning rate of MLP
    lr_wd
        learning rate of WGAN discriminator
    lr_recon
        learning rate of reconstruction
    batch_d_per_iter
        batch number for WGAN train per iter
    batch_r_per_iter
        batch number for reconstruct train per iter
    
    Return
    ----------
    embd0
        cell embedding of dataset1
    embd1
        cell embedding of dataset2
    time
        run time of SLAT model
    """
    
    feature_size = features[0].size(1)
    feature_output_size = hidden_size
    
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    feature_size = features[0].size(1)
    feature_output_size = hidden_size

    LGCN_model = LGCN_mlp(feature_size, hidden_size, K=LGCN_layer, hidden_size=mlp_hidden).to(device)
    optimizer_LGCN = torch.optim.Adam(LGCN_model.parameters(), lr=lr_mlp, weight_decay=5e-4)

    wdiscriminator = WDiscriminator(feature_output_size).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=lr_wd, weight_decay=5e-4)

    recon_model0 = ReconDNN(feature_output_size, feature_size).to(device)
    recon_model1 = ReconDNN(feature_output_size, feature_size).to(device)
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)


    print('Running')
    time1 = time.time()
    for i in range(1, epochs + 1):
        print(f'---------- epochs: {i} ----------')
        
        LGCN_model.train()
        optimizer_LGCN.zero_grad()
        # norm
        embd0, norm0 = LGCN_model(features[0], edges[0])
        embd1, norm1 = LGCN_model(features[1], edges[1])

        loss_gan = train_GAN(wdiscriminator, optimizer_wd, [embd0,embd1], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
        loss_feature = train_reconstruct([recon_model0, recon_model1], [optimizer_recon0, optimizer_recon1], [embd0,embd1], features,batch_r_per_iter=batch_r_per_iter)
        loss = (1-alpha) * loss_gan + alpha * loss_feature

        loss.backward()
        optimizer_LGCN.step()
        
    LGCN_model.eval()
    embd0, norm0 = LGCN_model(features[0], edges[0])
    embd1, norm1 = LGCN_model(features[1], edges[1])

    time2 = time.time()
    print('Training model time: %.2f' % (time2-time1))
    # torch.cuda.empty_cache()
    return embd0, embd1, norm0, norm1, time2-time1


def run_SLAT_mlp_AGF(features: List,
                     edges: List,
                     coordinates: List,
                     epochs: Optional[int] = 6,
                     LGCN_layer: Optional[int] = 1,
                     mlp_hidden: Optional[int] = 256,
                     output_size: Optional[int] = 256,
                     alpha: Optional[float] = 0.01,
                     anchor_scale: Optional[float] = 0.8,
                     lr_mlp: Optional[float] = 0.0001,
                     lr_wd: Optional[float] = 0.0001,
                     lr_recon: Optional[float] = 0.01,
                     batch_d_per_iter: Optional[int] = 5,
                     batch_r_per_iter: Optional[int] = 10
                     ) -> List:
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    # torch.autograd.set_detect_anomaly(True)

    feature_size = features[0].size(1)
    feature_output_size = output_size

    # Initialize models for m=0 and m=1
    LGCN_model_0 = LGCN_mlp_AGF(feature_size, feature_output_size, coordinates, 0, K=LGCN_layer, hidden_size=mlp_hidden).to(device)
    LGCN_model_1 = LGCN_mlp_AGF(feature_size, feature_output_size, coordinates, 1, K=LGCN_layer, hidden_size=mlp_hidden).to(device)

    optimizer_LGCN_0 = torch.optim.Adam(LGCN_model_0.parameters(), lr=lr_mlp, weight_decay=5e-4)
    optimizer_LGCN_1 = torch.optim.Adam(LGCN_model_1.parameters(), lr=lr_mlp, weight_decay=5e-4)

    wdiscriminator0 = WDiscriminator(feature_output_size).to(device)
    wdiscriminator1 = WDiscriminator(feature_output_size).to(device)
    
    optimizer_wd0 = torch.optim.Adam(wdiscriminator0.parameters(), lr=lr_wd, weight_decay=5e-4)
    optimizer_wd1 = torch.optim.Adam(wdiscriminator1.parameters(), lr=lr_wd, weight_decay=5e-4)
    
    recon_model0 = ReconDNN(feature_output_size, feature_size).to(device)
    recon_model1 = ReconDNN(feature_output_size, feature_size).to(device)
    
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)
    
    attention_layer = AttentionWeighted(NFeature=feature_output_size).to(device)
    optimizer_attention = torch.optim.Adam(attention_layer.parameters(), lr=0.005)


    print('Running')
    time1 = time.time()

    def train_stage_1(model, optimizer, wdiscriminator, optimizer_wd, recon_model, optimizer_recon, m):
        
        for epoch in range(1, epochs + 1):
            print(f'---------- epochs: {epoch} ----------')
            model.train()
            optimizer.zero_grad()
            embd0 = model(features[0], edges[0], coordinates[0], m)
            embd1 = model(features[1], edges[1], coordinates[1], m)

            loss_gan = train_GAN(wdiscriminator, optimizer_wd, [embd0, embd1], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
            loss_recon = train_reconstruct([recon_model, recon_model], [optimizer_recon, optimizer_recon], [embd0, embd1], features, batch_r_per_iter=batch_r_per_iter)
            loss = (1 - alpha) * loss_gan + alpha * loss_recon      
            loss.backward(retain_graph=True)
            optimizer.step()
        
        model.eval()
        embd0 = model(features[0], edges[0], coordinates[0], m)
        embd1 = model(features[1], edges[1], coordinates[1], m)

        return embd0, embd1

    # Train for m=0
    embd0_0, embd1_0 = train_stage_1(LGCN_model_0, optimizer_LGCN_0, wdiscriminator0, optimizer_wd0, recon_model0, optimizer_recon0, m=0)
    
    # Train for m=1
    embd0_1, embd1_1 = train_stage_1(LGCN_model_1, optimizer_LGCN_1, wdiscriminator1, optimizer_wd1, recon_model1, optimizer_recon1, m=1)

    # # Concatenate embeddings from m=0 and m=1
    embd0 = torch.cat([embd0_0, embd0_1], dim=1)
    embd1 = torch.cat([embd1_0, embd1_1], dim=1)

    wdiscriminator_combined = WDiscriminator(feature_output_size ).to(device)
    optimizer_wd_combined = torch.optim.Adam(wdiscriminator_combined.parameters(), lr=lr_wd, weight_decay=5e-4)

    recon_model_combined_0 = ReconDNN(feature_output_size , feature_size).to(device)
    recon_model_combined_1 = ReconDNN(feature_output_size , feature_size).to(device)
    optimizer_recon_combined_0 = torch.optim.Adam(recon_model_combined_0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon_combined_1 = torch.optim.Adam(recon_model_combined_1.parameters(), lr=lr_recon, weight_decay=5e-4)

    torch.autograd.set_detect_anomaly(True)

    contrast_loss_fn = Contrast(hidden_dim=feature_output_size, tau=0.8, lam=0.5).to(device)
    optimizer_contrast = torch.optim.Adam(contrast_loss_fn.parameters(), lr=0.1, weight_decay=5e-4)

    pos_1 = torch.eye(embd0_0.size(0)).to(device)
    pos_2 = torch.eye(embd1_0.size(0)).to(device)

    # Initialize dimensionality reduction model
    input_size = embd0.size(1)
    output_size_2 = output_size
    dimension_model = DimensionalityReductionModel(input_size, output_size_2).to(device)
    optimizer_dim_reduction = torch.optim.Adam(dimension_model.parameters(), lr=lr_recon, weight_decay=5e-4)

    for epoch in range(epochs + 1):
        print(f'---------- Combined epochs: {epoch} ----------')
        dimension_model.train()
        optimizer_dim_reduction.zero_grad()
        optimizer_wd_combined.zero_grad()
        optimizer_recon_combined_0.zero_grad()
        optimizer_recon_combined_1.zero_grad()

        embd0_proj_0, embd0_proj_1, contrastive_loss_0 = contrast_loss_fn(embd0_0, embd0_1, pos_1)
        embd1_proj_0, embd1_proj_1, contrastive_loss_1 = contrast_loss_fn(embd1_0, embd1_1, pos_2)

        contrastive_loss = contrastive_loss_0 + contrastive_loss_1

        weighted_embd0 = torch.concat([embd0_proj_0, embd0_proj_1], dim=1)
        weighted_embd1 = torch.concat([embd1_proj_0, embd1_proj_1], dim=1)

        embd0_reduced = dimension_model(weighted_embd0)
        embd1_reduced = dimension_model(weighted_embd1)

        loss_gan_combined = train_GAN(wdiscriminator_combined, optimizer_wd_combined, [embd0_reduced, embd1_reduced], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
        loss_recon_combined = train_reconstruct([recon_model_combined_0, recon_model_combined_1], [optimizer_recon_combined_0, optimizer_recon_combined_1], [embd0_reduced, embd1_reduced], features, batch_r_per_iter=batch_r_per_iter)
        loss_combined = (1 - alpha) * loss_gan_combined + alpha * loss_recon_combined
        loss_combined.backward(retain_graph=True)
        optimizer_wd_combined.step()
        optimizer_recon_combined_0.step()
        optimizer_recon_combined_1.step()
        optimizer_dim_reduction.step()
        print(f'GAN Loss Combined: {(1 - alpha) * loss_gan_combined.item()}')
        print(f'Reconstruction Loss Combined: {alpha * loss_recon_combined.item()}')
        print(f'Total Combined Loss: {loss_combined.item()}')
    dimension_model.eval()
    embd0_reduced = dimension_model(weighted_embd0)
    embd1_reduced = dimension_model(weighted_embd1)

    time2 = time.time()
    return embd0_0, embd0_1, embd1_0, embd1_1, embd0, embd1, embd0_reduced, embd1_reduced, time2 - time1

def run_SLAT_AGF_contrast(features: List,
                     edges: List,
                     coordinates: List,
                     epochs: Optional[int] = 6,
                     LGCN_layer: Optional[int] = 1,
                     mlp_hidden: Optional[int] = 256,
                     output_size: Optional[int] = 256,
                     alpha: Optional[float] = 0.01,
                     anchor_scale: Optional[float] = 0.8,
                     lr_mlp: Optional[float] = 0.0001,
                     lr_wd: Optional[float] = 0.0001,
                     lr_recon: Optional[float] = 0.01,
                     batch_d_per_iter: Optional[int] = 5,
                     batch_r_per_iter: Optional[int] = 10,
                     limit_loss: Optional[float] = 0.0001
                     ) -> List:
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    # torch.autograd.set_detect_anomaly(True)
    feature_size = features[0].size(1)
    feature_output_size = output_size

    # Initialize models for m=0 and m=1
    LGCN_model_0 = LGCN_mlp_AGF(feature_size, feature_output_size, coordinates, 0, K=LGCN_layer, hidden_size=mlp_hidden).to(device)
    LGCN_model_1 = LGCN_mlp_AGF(feature_size, feature_output_size, coordinates, 1, K=LGCN_layer, hidden_size=mlp_hidden).to(device)

    optimizer_LGCN_0 = torch.optim.Adam(LGCN_model_0.parameters(), lr=lr_mlp, weight_decay=5e-4)
    optimizer_LGCN_1 = torch.optim.Adam(LGCN_model_1.parameters(), lr=lr_mlp, weight_decay=5e-4)

    wdiscriminator0 = WDiscriminator(feature_output_size).to(device)
    wdiscriminator1 = WDiscriminator(feature_output_size).to(device)
    
    optimizer_wd0 = torch.optim.Adam(wdiscriminator0.parameters(), lr=lr_wd, weight_decay=5e-4)
    optimizer_wd1 = torch.optim.Adam(wdiscriminator1.parameters(), lr=lr_wd, weight_decay=5e-4)
    
    recon_model0 = ReconDNN(feature_output_size, feature_size).to(device)
    recon_model1 = ReconDNN(feature_output_size, feature_size).to(device)
    
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)

    print('Running')
    time1 = time.time()

    def train_stage_1(model, optimizer, wdiscriminator, optimizer_wd, recon_model, optimizer_recon, m):
        
        for epoch in range(1, epochs + 1):
            print(f'---------- epochs: {epoch} ----------')
            model.train()
            optimizer.zero_grad()
            embd0 = model(features[0], edges[0], coordinates[0], m)
            embd1 = model(features[1], edges[1], coordinates[1], m)

            loss_gan = train_GAN(wdiscriminator, optimizer_wd, [embd0, embd1], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
            loss_recon = train_reconstruct([recon_model, recon_model], [optimizer_recon, optimizer_recon], [embd0, embd1], features, batch_r_per_iter=batch_r_per_iter)
            loss = (1 - alpha) * loss_gan + alpha * loss_recon

            loss.backward(retain_graph=True)
            optimizer.step()
        
        model.eval()
        embd0 = model(features[0], edges[0], coordinates[0], m)
        embd1 = model(features[1], edges[1], coordinates[1], m)

        return embd0, embd1

    # Train for m=0
    embd0_0, embd1_0 = train_stage_1(LGCN_model_0, optimizer_LGCN_0, wdiscriminator0, optimizer_wd0, recon_model0, optimizer_recon0, m=0)    
    # Train for m=1
    embd0_1, embd1_1 = train_stage_1(LGCN_model_1, optimizer_LGCN_1, wdiscriminator1, optimizer_wd1, recon_model1, optimizer_recon1, m=1)

    # Initialize Contrastive Loss
    contrast_loss_fn = Contrast(hidden_dim=feature_output_size, tau=0.5).to(device)
    optimizer_contrast = torch.optim.Adam(contrast_loss_fn.parameters(), lr=0.05)

    num_nodes_1 = torch.max(edges[0]) + 1
    num_nodes_2 = torch.max(edges[1]) + 1

    pos_1 = torch.zeros(num_nodes_1, num_nodes_1)
    pos_2 = torch.zeros(num_nodes_2, num_nodes_2)
    for i in range(edges[0].shape[1]):
        pos_1[edges[0][0, i], edges[0][1, i]] = 1
        pos_1[edges[0][1, i], edges[0][0, i]] = 1 

    for i in range(edges[1].shape[1]):
        pos_2[edges[1][0, i], edges[1][1, i]] = 1
        pos_2[edges[1][1, i], edges[1][0, i]] = 1 
    epoch = 0
    previous_contrastive_loss = None
    while True:  # Infinite loop until break condition is met
        print(f'---------- Combined epochs: {epoch} ----------')
        contrast_loss_fn.train()
        optimizer_contrast.zero_grad()

        embd0_proj_0, embd0_proj_1, contrastive_loss_0 = contrast_loss_fn(embd0_0, embd0_1, pos_1)
        embd1_proj_0, embd1_proj_1, contrastive_loss_1 = contrast_loss_fn(embd1_0, embd1_1, pos_2)

        contrastive_loss = contrastive_loss_0 + contrastive_loss_1
        weighted_embd0 = torch.concat([embd0_proj_0, embd0_proj_1], dim=1)
        weighted_embd1 = torch.concat([embd1_proj_0, embd1_proj_1], dim=1)

        contrastive_loss.backward(retain_graph=True)
        optimizer_contrast.step()
        # Check for stopping condition using the difference between the two losses
        if epoch > 0 and abs(previous_contrastive_loss - contrastive_loss.item()) < limit_loss:
            print(f'Training stopped at epoch: {epoch}')
            break

        # Save the current total contrastive loss for the next iteration
        previous_contrastive_loss = contrastive_loss.item()
        epoch += 1
    contrast_loss_fn.eval()
    weighted_embd0 = torch.concat([embd0_proj_0, embd0_proj_1], dim=1)
    weighted_embd1 = torch.concat([embd1_proj_0, embd1_proj_1], dim=1)
    time2 = time.time()
    return embd0_0, embd0_1, embd1_0, embd1_1, weighted_embd0, weighted_embd1, time2 - time1

def stLVG_contrast_norm(features: List,
                     edges: List,
                     coordinates: List,
                     epochs: Optional[int] = 6,
                     LGCN_layer: Optional[int] = 1,
                     mlp_hidden: Optional[int] = 256,
                     output_size: Optional[int] = 256,
                     alpha: Optional[float] = 0.01,
                     anchor_scale: Optional[float] = 0.8,
                     lr_mlp: Optional[float] = 0.0001,
                     lr_wd: Optional[float] = 0.0001,
                     lr_recon: Optional[float] = 0.01,
                     batch_d_per_iter: Optional[int] = 5,
                     batch_r_per_iter: Optional[int] = 10,
                     ) -> List:
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    # torch.autograd.set_detect_anomaly(True)
    feature_size = features[0].size(1)
    feature_output_size = output_size

    # Initialize models for m=0 and m=1
    LGCN_model_0 = LGCN_AGF_norm(feature_size, feature_output_size, coordinates, 0, K=LGCN_layer, hidden_size=mlp_hidden).to(device)
    LGCN_model_1 = LGCN_AGF_norm(feature_size, feature_output_size, coordinates, 1, K=LGCN_layer, hidden_size=mlp_hidden).to(device)

    optimizer_LGCN_0 = torch.optim.Adam(LGCN_model_0.parameters(), lr=lr_mlp, weight_decay=5e-4)
    optimizer_LGCN_1 = torch.optim.Adam(LGCN_model_1.parameters(), lr=lr_mlp, weight_decay=5e-4)

    wdiscriminator0 = WDiscriminator(feature_output_size).to(device)
    wdiscriminator1 = WDiscriminator(feature_output_size).to(device)
    
    optimizer_wd0 = torch.optim.Adam(wdiscriminator0.parameters(), lr=lr_wd, weight_decay=5e-4)
    optimizer_wd1 = torch.optim.Adam(wdiscriminator1.parameters(), lr=lr_wd, weight_decay=5e-4)
    
    recon_model0 = ReconDNN(feature_output_size, feature_size).to(device)
    recon_model1 = ReconDNN(feature_output_size, feature_size).to(device)
    
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=lr_recon, weight_decay=5e-4)

    print('Running')
    time1 = time.time()

    def train_stage_1(model, optimizer, wdiscriminator, optimizer_wd, recon_model, optimizer_recon, m):
        
        for epoch in range(1, epochs + 1):
            print(f'---------- epochs: {epoch} ----------')
            model.train()
            optimizer.zero_grad()
            embd0, norm0 = model(features[0], edges[0], coordinates[0], m)
            embd1, norm1 = model(features[1], edges[1], coordinates[1], m)

            loss_gan = train_GAN(wdiscriminator, optimizer_wd, [embd0, embd1], batch_d_per_iter=batch_d_per_iter, anchor_scale=anchor_scale)
            loss_recon = train_reconstruct([recon_model, recon_model], [optimizer_recon, optimizer_recon], [embd0, embd1], features, batch_r_per_iter=batch_r_per_iter)
            loss = (1 - alpha) * loss_gan + alpha * loss_recon

            loss.backward(retain_graph=True)
            optimizer.step()
        
        model.eval()
        embd0, norm0 = model(features[0], edges[0], coordinates[0], m)
        embd1, norm1 = model(features[1], edges[1], coordinates[1], m)

        return embd0, embd1, norm0, norm1

    # Train for m=0
    embd0_0, embd1_0, norm0, norm1 = train_stage_1(LGCN_model_0, optimizer_LGCN_0, wdiscriminator0, optimizer_wd0, recon_model0, optimizer_recon0, m=0)    
    # Train for m=1
    embd0_1, embd1_1, norm2, norm3 = train_stage_1(LGCN_model_1, optimizer_LGCN_1, wdiscriminator1, optimizer_wd1, recon_model1, optimizer_recon1, m=1)

    time2 = time.time()
    return embd0_0, embd0_1, norm0, norm1, embd1_0, embd1_1, norm2, norm3, time2 - time1

def spatial_match(embds:List[torch.Tensor],
                  reorder:Optional[bool]=True,
                  smooth:Optional[bool]=True,
                  smooth_range:Optional[int]=20,
                  scale_coord:Optional[bool]=True,
                  adatas:Optional[List[AnnData]]=None,
                  return_euclid:Optional[bool]=False,
                  verbose:Optional[bool]=False,
                  get_null_distri:Optional[bool]=False
    )-> List[Union[np.ndarray,torch.Tensor]]:
    r"""
    Use embedding to match cells from different datasets based on cosine similarity
    
    Parameters
    ----------
    embds
        list of embeddings
    reorder
        if reorder embedding by cell numbers
    smooth
        if smooth the mapping by Euclid distance
    smooth_range
        use how many candidates to do smooth
    scale_coord
        if scale the coordinate to [0,1]
    adatas
        list of adata object
    verbose
        if print log
    get_null_distri
        if get null distribution of cosine similarity
    
    Note
    ----------
    Automatically use larger dataset as source
    
    Return
    ----------
    Best matching, Top n matching and cosine similarity matrix of top n  
    
    Note
    ----------
    Use faiss to accelerate, refer https://github.com/facebookresearch/faiss/issues/95
    """
    if reorder and embds[0].shape[0] < embds[1].shape[0]:
        embd0 = embds[1]
        embd1 = embds[0]
        adatas = adatas[::-1] if adatas is not None else None
    else:
        embd0 = embds[0]
        embd1 = embds[1]
        
    if get_null_distri:
        embd0 = torch.tensor(embd0)
        embd1 = torch.tensor(embd1)
        sample1_index = torch.randint(0, embd0.shape[0], (1000,))
        sample2_index = torch.randint(0, embd1.shape[0], (1000,))
        cos = torch.nn.CosineSimilarity(dim=1)
        null_distri = cos(embd0[sample1_index], embd1[sample2_index])

    index = faiss.index_factory(embd1.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    embd0_np = embd0.detach().cpu().numpy() if torch.is_tensor(embd0) else embd0
    embd1_np = embd1.detach().cpu().numpy() if torch.is_tensor(embd1) else embd1
    embd0_np = embd0_np.copy().astype('float32')
    embd1_np = embd1_np.copy().astype('float32')
    faiss.normalize_L2(embd0_np)
    faiss.normalize_L2(embd1_np)
    index.add(embd0_np)
    similarity, order = index.search(embd1_np, smooth_range)
    best = []
    if smooth and adatas != None:
        if verbose:
            print('Smoothing mapping, make sure object is in same direction')
        if scale_coord:
            # scale spatial coordinate of every adata to [0,1]
            adata1_coord = adatas[0].obsm['spatial'].copy()
            adata2_coord = adatas[1].obsm['spatial'].copy()
            for i in range(2):
                    adata1_coord[:,i] = (adata1_coord[:,i]-np.min(adata1_coord[:,i]))/(np.max(adata1_coord[:,i])-np.min(adata1_coord[:,i]))
                    adata2_coord[:,i] = (adata2_coord[:,i]-np.min(adata2_coord[:,i]))/(np.max(adata2_coord[:,i])-np.min(adata2_coord[:,i]))
        dis_list = []
        for query in range(embd1_np.shape[0]):
            ref_list = order[query, :smooth_range]
            dis = euclidean_distances(adata2_coord[query,:].reshape(1, -1),
                                      adata1_coord[ref_list,:])
            dis_list.append(dis)
            best.append(ref_list[np.argmin(dis)])
    else:
        best = order[:,0]

    if return_euclid and smooth and adatas != None:
        dis_array = np.squeeze(np.array(dis_list))
        if get_null_distri:
            return np.array(best), order, similarity, dis_array, null_distri
        else:
            return np.array(best), order, similarity, dis_array
    else:
        return np.array(best), order, similarity
    


def probabilistic_match(cos_cutoff:float=0.6, euc_cutoff:int=5, **kargs)-> List[List[int]]:
    
    best, index, similarity, eucli_array, null_distri = \
        spatial_match(**kargs, return_euclid=True, get_null_distri=True)
    # filter the cosine similarity via p_value
    # mask1 = similarity > cos_cutoff
    null_distri = np.sort(null_distri)
    p_val = 1 - np.searchsorted(null_distri, similarity) / null_distri.shape[0]
    mask1 = p_val < 0.05
    
    # filter the euclidean distance
    sorted_indices = np.argpartition(eucli_array, euc_cutoff, axis=1)[:, :euc_cutoff]
    mask2 = np.full(eucli_array.shape, False, dtype=bool)
    mask2[np.arange(eucli_array.shape[0])[:, np.newaxis], sorted_indices] = True
    
    mask_mat = np.logical_and(mask1, mask2)
    filter_list = [row[mask].tolist() for row, mask in zip(index, mask_mat)]
    matching = [ [i,j] for i,j in zip(np.arange(index.shape[0]), filter_list) ]

    return matching

def calc_k_neighbor(features:List[torch.Tensor],
                    k_list:List[int]
    ) -> Mapping:
    r"""
    cal k nearest neighbor
    
    Parameters:
    ----------
    features
        feature list to find KNN
    k
        list of k to find (must have 2 elements)
    """
    assert len(k_list) == 2
    k_list = sorted(k_list)
    nbr_dict = {}
    for k in k_list:
        nbr_dict[k] = [None, None]

    for i, feature in enumerate(features): # feature loop first
        for k in k_list:     # then k list loop
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(feature)
            distances, indices = nbrs.kneighbors(feature) # indices include it self
            nbr_dict[k][i] = nbrs

    return nbr_dict


def add_noise(adata,
              noise:Optional[str]='nb',
              inverse_noise:Optional[float]=5
    ) -> AnnData:
    r"""
    Add poisson or negative binomial noise on raw counts
    also run scanpy pipeline to PCA step
    
    Parameters
    ----------
    adata
        anndata object
    noise
        type of noise, one of 'poisson' or 'nb'
    inverse_noise
        if noise is 'nb', control the noise level 
        (smaller means larger variance) 
    """
    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()
    mu = torch.tensor(adata.X.todense())
    if noise.lower() == 'poisson':
        adata.X = torch.distributions.poisson.Poisson(mu).sample().numpy()
    elif noise.lower() == 'nb':
        adata.X = torch.distributions.negative_binomial.NegativeBinomial(inverse_noise,logits=(mu.log()-math.log(inverse_noise))).sample().numpy()
    else:
        raise NotImplementedError('Can not add this type noise')
    return adata.copy()

def compute_lisi(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata."""
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    knn = NearestNeighbors(n_neighbors=int(perplexity * 3), algorithm='kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:, i] = 1 / simpson
    return lisi_df

def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5
):
    """
    Parameters:
    -----------
    distances : np.ndarray
        KNN distances matrix (neighbors x cells)
    indices : np.ndarray
        KNN indices matrix (neighbors x cells)
    labels : pd.Categorical
        Category labels for each cell
    n_categories : int
        Number of unique categories
    perplexity : float
        Target entropy of distribution
    tol : float, optional
        Convergence tolerance
        
    Note
    ----------
    refer https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
    """
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        if H == 0:
            simpson[i] = -1
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def compute_lisi_for_adata(adata, obsm_key, obs_key_list, perplexity=30):
    
    X = np.array(adata.obsm[obsm_key])
    metadata = adata.obs[obs_key_list]
    lisi_scores = compute_lisi(X, metadata, label_colnames=obs_key_list, perplexity=perplexity)
    cLISI = np.mean(lisi_scores[:, 0])
    iLISI = np.mean(lisi_scores[:, 1])
    
    return {'cLISI': cLISI, 'iLISI': iLISI}


def zscore(matrix: Union[np.ndarray, csr_matrix],
           axis: int = 0,
           ) -> np.ndarray:
    """
    Z-score data matrix along desired dimension (e.g., rows or columns).
    """

    # Compute mean along the given axis (for rows if axis=1, for columns if axis=0)
    E_x = matrix.mean(axis=axis, keepdims=True)

    # Handle sparse matrix case
    if issparse(matrix):
        squared = matrix.copy()
        squared.data **= 2
        E_x2 = squared.mean(axis=axis, keepdims=True)
    else:
        E_x2 = np.square(matrix).mean(axis=axis, keepdims=True)

    # Calculate variance
    variance = E_x2 - np.square(E_x)

    # Perform Z-score standardization, ensuring correct broadcasting
    zscored_matrix = (matrix - E_x) / np.sqrt(variance)

    if isinstance(zscored_matrix, np.matrix):
        zscored_matrix = np.array(zscored_matrix)

    # Handle NaN values (e.g., for rows/columns with zero variance)
    zscored_matrix = np.nan_to_num(zscored_matrix)

    return zscored_matrix



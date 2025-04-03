"""
Spatial Transcriptomics Data Processing Module for Graph Neural Networks

This module handles conversion of AnnData objects with spatial information 
into PyTorch Geometric compatible graph datasets.
"""

from typing import Optional, List, Union
import scanpy as sc
import numpy as np
import scipy.sparse
from anndata import AnnData
import torch
from torch_geometric.data import Data
from .batch import dual_pca
from .preprocess import scanpy_workflow

def Transfer_pyg_Data(adata: AnnData, feature: Optional[str] = 'PCA') -> Data:
    """Convert AnnData object with spatial information to PyG Data object.
    
    Args:
        adata: Input spatial transcriptomics data
        feature: Feature space to use (options: 'PCA', 'HVG', 'raw')
    
    Returns:
        PyG Data object containing graph structure and node features
    
    Note:
        Requires precomputed spatial network in adata.uns['Spatial_Net']
    """
    adata = adata.copy()
    spatial_net = adata.uns['Spatial_Net'].copy()
    
    # Map cell names to indices
    cells = np.array(adata.obs_names)
    cell_id_map = {name: idx for idx, name in enumerate(cells)}
    spatial_net['Cell1'] = spatial_net['Cell1'].map(cell_id_map)
    spatial_net['Cell2'] = spatial_net['Cell2'].map(cell_id_map)

    # Build adjacency matrix with self-loops
    adj_matrix = scipy.sparse.coo_matrix(
        (np.ones(len(spatial_net)), 
         (spatial_net['Cell1'], spatial_net['Cell2'])),
        shape=(adata.n_obs, adata.n_obs)
    )
    adj_matrix += scipy.sparse.eye(adj_matrix.shape[0])
    edge_index = np.nonzero(adj_matrix)

    # Process features based on selected method
    feature = feature.lower()
    assert feature in ['hvg', 'pca', 'raw']
    
    if feature == 'raw':
        features = adata.X.todense() if scipy.sparse.issparse(adata.X) else adata.X
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        if feature == 'hvg':
            sc.pp.highly_variable_genes(
                adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata = adata[:, adata.var.highly_variable]
            features = adata.X.todense()
        else:
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack')
            features = adata.obsm['X_pca'].copy()

    return Data(
        edge_index=torch.LongTensor(np.array(edge_index)),
        x=torch.FloatTensor(features)
    )

def load_anndata(
    adata: AnnData,
    feature: str = 'PCA',
    noise_level: float = 0,
    noise_type: str = 'uniform',
    edge_homo_ratio: float = 0.9,
    return_PCs: bool = False
) -> List:
    """Generate augmented graph pairs for testing purposes.
    
    Args:
        adata: Input spatial dataset
        feature: Feature space to use
        noise_level: Magnitude of additive noise
        noise_type: Type of noise ('uniform'/'normal')
        edge_homo_ratio: Edge preservation ratio
        return_PCs: Whether to return PCA components
    
    Returns:
        List containing original and augmented graph components
    
    Warning: Experimental function for data augmentation testing
    """
    dataset, PCs = Transfer_pyg_Data(adata, feature)
    edge_index, features = dataset.edge_index, dataset.x

    # Create augmented graph
    perm = torch.randperm(features.size(0))
    edge_index_aug = edge_index[:, torch.randperm(edge_index.size(1))]
    edge_index_aug = edge_index_aug[:, :int(edge_index_aug.size(1)*edge_homo_ratio)]
    edge_index_aug = perm[edge_index_aug.view(-1)].view(2, -1)
    edge_index_aug = edge_index_aug[:, torch.argsort(edge_index_aug[0])]

    # Apply permutation and noise
    features_aug = torch.zeros_like(features)
    features_aug[perm] = features.clone()
    
    if noise_type == 'uniform':
        features_aug += 2*(torch.rand_like(features_aug)-0.5)*noise_level
    elif noise_type == 'normal':
        features_aug += torch.randn_like(features_aug)*noise_level

    return [edge_index, features, edge_index_aug, features_aug, 
            torch.stack([torch.arange(len(perm)), perm])] + \
           ([PCs] if return_PCs else [])

def load_anndatas(adatas:List[AnnData],
                feature:Optional[str]='DPCA',
                dim:Optional[int]=50,
                self_loop:Optional[bool]=False,
                join:Optional[str]='inner',
                backend:Optional[str]='sklearn',
                singular:Optional[bool]=True,
                check_order:Optional[bool]=True,
                n_top_genes:Optional[int]=2500,
    ) -> List[Data]:
    r"""
    Transfer adatas with spatial info into PyG datasets
    
    Parameters:
    ----------
    adatas
        List of Anndata objects
    feature
        use which data to build graph
        - `PCA` (default)
        - `DPCA` (For batch effect correction)
        - `Harmony` (For batch effect correction)
        - `GLUE` (**NOTE**: only suitable for multi-omics integration)
    
    dim
        dimension of embedding, works for ['PCA', 'DPCA', 'Harmony', 'GLUE']
    self_loop
        whether to add self loop on graph
    join
        how to concatenate two adata
    backend
        backend to calculate DPCA
    singular
        whether to multiple singular value in DPCA
    check_order
        whether to check the order of adata1 and adata2
    n_top_genes
        number of highly variable genes
        
    Note:
    ----------
    Only support 'Spatial_Net' which store in `adata.uns` yet
    """
    assert len(adatas) == 2
    assert feature.lower() in ['raw','hvg','pca','dpca','harmony','glue','scglue']
    if check_order and adatas[0].shape[0] < adatas[1].shape[0]:
        raise ValueError('Please change the order of adata1 and adata2 or set `check_order=False`')
    gpu_flag = True if torch.cuda.is_available() else False

    adatas = [adata.copy() for adata in adatas ] # May consume more memory
    
    # Edge
    edgeLists = []
    for adata in adatas:
        G_df = adata.uns['Spatial_Net'].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

        # build adjacent matrix
        G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), 
                                    shape=(adata.n_obs, adata.n_obs))
        if self_loop:
            G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    datas = []
    print(f'Use {feature} feature to format graph')
    if feature.lower() == 'raw':
        for i, adata in enumerate(adatas):
            if type(adata.X) == np.ndarray:
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])),
                            x=torch.FloatTensor(adata.X))  # .todense()
            else:
                data = Data(edge_index=torch.LongTensor(np.array(
                    [edgeLists[i][0], edgeLists[i][1]])), x=torch.FloatTensor(adata.X.todense()))
            datas.append(data)
    
    elif feature.lower() in ['glue','scglue']:
        for i, adata in enumerate(adatas):
            assert 'X_glue' in adata.obsm.keys()
            data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])),
                        x=torch.FloatTensor(adata.obsm['X_glue'][:,:dim]))
            datas.append(data)
    
    elif feature.lower() in ['hvg','pca','harmony']:
        adata_all = adatas[0].concatenate(adatas[1], join=join)
        adata_all = scanpy_workflow(adata_all, n_top_genes=n_top_genes, n_comps=-1)
        if feature.lower() == 'hvg':
            if not adata_all.var.highly_variable is None:
                adata_all = adata_all[:, adata_all.var.highly_variable]
            for i in len(adatas):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.X.todense()))
                datas.append(data)
        sc.tl.pca(adata_all, svd_solver='auto')
        if feature.lower() == 'pca':
            for i in range(len(adatas)):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.obsm['X_pca'][:,:dim]))
                datas.append(data)
        elif feature.lower() == 'harmony':
            from harmony import harmonize
            if gpu_flag:
                print('Harmony is using GPU!')
            Z = harmonize(adata_all.obsm['X_pca'], adata_all.obs, random_state=0, 
                        max_iter_harmony=30, batch_key='batch', use_gpu=gpu_flag)
            adata_all.obsm['X_harmony'] = Z[:,:dim]
            for i in range(len(adatas)):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.obsm['X_harmony'][:,:dim]))
                datas.append(data)

    elif feature.lower() == 'dpca':
        adata_all = adatas[0].concatenate(adatas[1], join=join)
        sc.pp.highly_variable_genes(adata_all, n_top_genes=12000, flavor="seurat_v3")
        adata_all = adata_all[:, adata_all.var.highly_variable]
        sc.pp.normalize_total(adata_all)
        sc.pp.log1p(adata_all)
        adata_1 = adata_all[adata_all.obs['batch'] == '0']
        adata_2 = adata_all[adata_all.obs['batch'] == '1']
        sc.pp.scale(adata_1)
        sc.pp.scale(adata_2)
        if gpu_flag:
            print('Warning! Dual PCA is using GPU, which may lead to OUT OF GPU MEMORY in big dataset!')
        Z_x, Z_y = dual_pca(adata_1.X, adata_2.X, dim=dim, singular=singular, backend=backend, use_gpu=gpu_flag)
        data_x = Data(edge_index=torch.LongTensor(np.array([edgeLists[0][0], edgeLists[0][1]])),
                    x=Z_x)
        data_y = Data(edge_index=torch.LongTensor(np.array([edgeLists[1][0], edgeLists[1][1]])),
                    x=Z_y)
        datas = [data_x, data_y]
    
    edges = [dataset.edge_index for dataset in datas]
    features = [dataset.x for dataset in datas]
    return edges, features
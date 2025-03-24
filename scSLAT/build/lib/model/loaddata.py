r"""
Extract cell features from anndata
"""
from random import random
from typing import Optional, List, Union
from joblib import Parallel, delayed

import scanpy as sc
import numpy as np
import scipy
import scipy.sparse
from anndata import AnnData
import torch
from torch_geometric.data import Data

from .batch import dual_pca
from .preprocess import scanpy_workflow


def Transfer_pyg_Data(adata:AnnData,
                    feature:Optional[str]='PCA'
    ) -> Data:
    r"""
    Transfer an adata with spatial info into PyG dataset (only for test)
    
    Parameters:
    ----------
    adata
        Anndata object
    feature
        use which data to build graph
        - PCA (default)
        
    Note:
    ----------
    Only support 'Spatial_Net' which store in `adata.uns` yet
    """
    adata = adata.copy()
    G_df = adata.uns['Spatial_Net'].copy()#复制adata，并从中得到空间网络信息
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)#获得细胞名称，并对应到索引中

    # build Adjacent Matrix
    G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + scipy.sparse.eye(G.shape[0])

    edgeList = np.nonzero(G)
    
    #构建邻接矩阵，并且得到边的索引

    # select feature
    assert feature.lower() in ['hvg','pca','raw']#确保参数feature的值在指定的列表['hvg','pca','raw']中
    if feature.lower() == 'raw':
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()将稀疏矩阵转化为稠密矩阵
        return data
    elif feature.lower() in ['pca','hvg']:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)#归一化，对数化处理
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        if feature.lower() == 'hvg':
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))#对于hvg基因进行额外处理
        sc.pp.scale(adata, max_value=10)
        print('Use PCA to format graph')
        sc.tl.pca(adata, svd_solver='arpack')
        data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['X_pca'].copy()))
        return data, adata.varm['PCs']#对数据进行pca降维

#仅测试
def load_anndata(
    adata: AnnData,
    feature: Optional[str]='PCA',
    noise_level: Optional[float]=0,
    noise_type: Optional[str]='uniform',#噪声类型和水平
    edge_homo_ratio: Optional[float]=0.9,#边同质比例
    return_PCs: Optional[bool]=False#是否返回主成分
    ) -> List:
    r"""
    Create 2 graphs from single anndata (only for test)
    
    Parameters
    ----------
    adata
        Anndata object
    feature
        feature to use to build graph and align, now support
        - `PCA`
        - `Harmony` (default)
    noise_level
        node noise 
    noise_type
        type of noise, support 'uniform' and 'normal'
    edge_homo_ratio
        ratio of edge in graph2
    return_PCs
        if return adata.varm['PCs'] if use feature 'PCA' (just for benchmark)
        
    Warning
    ----------
    This function is only for test. It generates `two` graphs 
    from single anndata by data augmentation
    """
    if feature=='PCA':
        dataset, PCs = Transfer_pyg_Data(adata, feature=feature)
    else:
        dataset = Transfer_pyg_Data(adata, feature=feature)
    edge1 = dataset.edge_index
    feature1 = dataset.x
    edge2 = edge1.clone()
    ledge = edge2.size(1) # get edge numbers
    edge2 = edge2[:, torch.randperm(ledge)[:int(ledge*edge_homo_ratio)]]
    perm = torch.randperm(feature1.size(0))
    perm_back = torch.tensor(list(range(feature1.size(0))))
    perm_mapping = torch.stack([perm_back, perm])
    edge2 = perm[edge2.view(-1)].view(2, -1) # reset edge order 
    edge2 = edge2[:, torch.argsort(edge2[0])]
    feature2 = torch.zeros(feature1.size())
    feature2[perm] = feature1.clone()#对第一个图的边和特征进行复制后重排列，生成第二个图
    if noise_type == 'uniform':
        feature2 = feature2 + 2 * (torch.rand(feature2.size())-0.5) * noise_level
    elif noise_type == 'normal':
        feature2 = feature2 + torch.randn(feature2.size()) * noise_level
    if feature=='PCA' and return_PCs:
        return edge1, feature1, edge2, feature2, perm_mapping, PCs
    return edge1, feature1, edge2, feature2, perm_mapping

#将多个adata转化为pyg
def load_anndatas(adatas:List[AnnData],
                feature:Optional[str]='DPCA',#双重主成分分析
                dim:Optional[int]=50,
                self_loop:Optional[bool]=False,#是否添加环
                join:Optional[str]='inner',#连接方式，inner，outer（保留共有/所有变量）
                backend:Optional[str]='sklearn',#计算选择的后端
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

    # Feature
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
        adata_all = adatas[0].concatenate(adatas[1], join=join) # join can not be 'outer'!
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
        # adata_1, adata_2 = Parallel(n_jobs=2)(delayed(sc.pp.scale)(adata) 
        #                                         for adata in [adata_1, adata_2])
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
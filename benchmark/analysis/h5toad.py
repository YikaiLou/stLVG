import scanpy as sc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scSLAT
from scSLAT.viz import Sankey

from importlib import reload
reload(scSLAT.viz)


adata1 = sc.read_h5ad("../results/visium_human_DLPFC/cells:0/seed:0/adata1.h5ad")
adata2 = sc.read_h5ad("../results/visium_human_DLPFC/cells:0/seed:0/adata2.h5ad")


matching = np.loadtxt("../results/visium_human_DLPFC/cells:0/seed:0/SLAT_dpca/matching.csv").astype(int)


# adata1.obs['target_celltype'] = adata2.obs.iloc[matching[0,:],:]['cell_type'].to_list()
adata2.obs['target_celltype'] = adata1.obs.iloc[matching[1,:],:]['cell_type'].to_list()
adata2.obs['target_region'] = adata1.obs.iloc[matching[1,:],:]['layer_guess'].to_list()


# adata1.obs["vis"] = 'celltype_false_region_false'
adata2.obs["vis"] = 'celltype_false_region_false'
adata2.obs["vis"] = adata2.obs["vis"].astype('str')



cell_type_match = adata2.obs['cell_type'] == adata2.obs['target_celltype']
region_match = adata2.obs['layer_guess'] == adata2.obs['target_region']
cell_type_match = cell_type_match.to_numpy()
region_match = region_match.to_numpy()

adata2.obs.loc[np.logical_and(cell_type_match, region_match), 'vis'] = 'celltype_true_region_true'
adata2.obs.loc[np.logical_and(~cell_type_match, region_match), 'vis'] = 'celltype_false_region_true'
adata2.obs.loc[np.logical_and(cell_type_match, ~region_match), 'vis'] = 'celltype_true_region_false'


sc.pl.spatial(adata2, color="vis", spot_size=5, title="matching", palette=['red', 'purple', 'yellow','green'])

matching


# reverse to adata1
adata1.obs['target_celltype'] = adata2.obs.iloc[matching[1,:],:]['cell_type'].to_list()
adata1.obs['target_region'] = adata2.obs.iloc[matching[1,:],:]['layer_guess'].to_list()

adata2.obs['cell_type'] = 'celltype_' + adata2.obs['cell_type'].astype('str')
matching_table = adata2.obs.groupby(['cell_type','target_celltype']).size().unstack(fill_value=0)
matching_table.index = adata2.obs['cell_type'].unique()
matching_table.columns = adata2.obs['cell_type'].unique()


matching_table

Sankey(matching_table, prefix=['Slide1', 'Slide2'], save_name='region', format='svg',
       width=1000, height=1000)

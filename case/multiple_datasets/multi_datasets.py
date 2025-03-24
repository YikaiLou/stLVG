import os
from pathlib import Path
from operator import itemgetter

import scanpy as sc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scSLAT.model import run_SLAT_multi
from scSLAT.viz import build_3D

sc.set_figure_params(dpi=150, dpi_save=150)
file_path = '../../data/stereo_seq/counts/E15.5/'
file_list = [file for file in Path(file_path).iterdir() if 'filter' not in str(file)]
print(file_list)


adata1,adata2,adata3,adata4 = Parallel(n_jobs=len(file_list)+1)\
    (delayed(sc.read_h5ad)(file) for file in itemgetter(*[3,1,0,2])(file_list))

#current_directory = os.getcwd()


# # 定义文件保存的路径
save_path = '/home/zuocm/Share_data/louyikai/SLAT-main/case/multiple_datasets/adata1_spatial.png'

# # 使用 scanpy 绘制并保存图像
sc.pl.spatial(adata1, spot_size=1, color='annotation', save=save_path)
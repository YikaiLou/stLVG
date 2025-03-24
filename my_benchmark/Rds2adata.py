import pyreadr
import anndata
import pandas as pd

# 读取 Rds 文件
file_path = "/home/zuocm/Share_data/louyikai/data/mRNA.Rds"
result = pyreadr.read_r(file_path)

# 检查结果的键
print("Available keys in Rds file:", result.keys())

# 获取数据
data = result[None]
print(f"Data class: {type(data)}")
print(data)

# 确保所有列的类型都是适当的
data = data.astype({col: str for col in data.columns})

# 转换为 AnnData 对象
adata = anndata.AnnData(data)

# 保存为 H5AD 文件
output_path = "/home/zuocm/Share_data/louyikai/data/mRNA.h5ad"
adata.write(output_path)
print("Data has been successfully converted to H5AD format.")



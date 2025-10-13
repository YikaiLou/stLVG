# stLVG: Spatio-Temporal Lightweight Vector-guided Graph Network
[figure1_919.tif](https://github.com/user-attachments/files/22888393/figure1_919.tif)
This repository contains the basic code and examples for stLVG, a novel vector-driven graph neural network for spatial multi-slice multi-omics integration.
## Installation
We recommend setting up a conda environment and then cloning this repository.
### Step 1
Create a environment named stLVG with python
```
conda create -n stLVG python=3.8.19 pip conda-forge jupyter
conda activate stLVG
```
### Step 2
```
git clone https://github.com/YikaiLou/stLVG.git
cd stLVG
```
### Step 3
```
conda env create -f environment.yml
conda activate stLVG
```
### Step 4
```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.2.1+cpu.html # Change according to your computer model
```

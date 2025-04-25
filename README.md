# stLVG: Spatio-Temporal Lightweight Vector Graph Network
![figure1](https://github.com/user-attachments/assets/1d0bdfe2-299c-4c9c-b0c3-a8b1e812025c)
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
```
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.2.1+cpu.html # Change according to your computer model
```

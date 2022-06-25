# Accelerating Stochastic Simulation with Interactive Neural Processes

This repository is a PyTorch implementation of Accelerating Stochastic Simulation with Interactive Neural Processes.

## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* torch
* tables
* future
* sklearn
* matplotlib
* gpytorch
* math

To install requirements:
```
pip install -r requirements.txt
```
## Model Training and Evaluation for SEIR Simulator
```
cd 2d_seir_code/*
python main.py
```

## [Dataset for LEAM-US Simulator](https://drive.google.com/drive/folders/1l5gqueulNXIrNc6yElx3WU8w-joxFiYj?usp=sharing)
Download test_data to dataset_dir: data/data/test_data

## Model Training and Evaluation for LEAM-US Simulator
```
cd leam_us_code/STNP/*
python dcrnn_train_pytorch.py
cd leam_us_code/offline/*
python dcrnn_train_pytorch.py
```

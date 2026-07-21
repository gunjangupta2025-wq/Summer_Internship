import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import drive

drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive'
DATA_DIR = os.path.join(BASE_DIR, 'split_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATASETS = ["BRCA", "BLCA", "OV"]
SPLITS = ["10", "20", "30", "40", "50"]

METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')
CONF_DIR = os.path.join(RESULTS_DIR, 'confidence')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

for folder in [METRICS_DIR, PRED_DIR, CONF_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device initialized: {DEVICE}")
print(f"Data root directory configured at: {DATA_DIR}")
print(f"Results will be exported to: {RESULTS_DIR}")

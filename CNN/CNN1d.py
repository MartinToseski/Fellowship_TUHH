'''
PREPROCESSING STEPS:
1. WFDB Format Handling                                                     ✓
2. Sampling Rate Handling (100 or 500 Hz)                                   ✓
    + Optional Bandpass Filter                                              -
3. Convert SCP Codes into Binary Vector                                     ✓
4. Signal Per-Record or Per-Lead Normalization                              ✓
5. Class Imbalance Handling - Weighted Loss Function Tested in Grid         -                                                                                      
6. Training/Validation/Split According to Folds                             ✓
    (1-8 training, 9 for validation, and 10 for testing)                    -
7. Reformat signal dimensions depending on model input requirements         ✓
8. Data Augmentation                                                        ✓
'''


import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms

from preprocessing import split_data
from utils import print_all_sizes, remove_empty_diagnosis, print_superclass_distribution_statistics



# ---------- DATA + CONFIG ----------
X_train, y_train, X_val, y_val, X_test, y_test = split_data()
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ---------- INITIAL STATS ----------
print_all_sizes("Initial", X_train, y_train, X_val, y_val, X_test, y_test)

# Remove empty lists with no diagnosis
X_train, y_train = remove_empty_diagnosis(X_train, y_train)
X_val, y_val = remove_empty_diagnosis(X_val, y_val)
X_test, y_test = remove_empty_diagnosis(X_test, y_test)
print_all_sizes("After removing", X_train, y_train, X_val, y_val, X_test, y_test)

# Print superclass distribution statistics
print_superclass_distribution_statistics(X_train, y_train, X_val, y_val, X_test, y_test)


# ---------- DATA PREPARATION ----------
# Convert aggregated superclasses to binary vector 
mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
y_train = mlb.fit_transform(y_train)
y_val = mlb.transform(y_val)
y_test = mlb.transform(y_test)

# Global per-lead normalization 
#print("X_train shape:", X_train.shape)
mean = X_train.mean(axis=(0,1), keepdims=True)
std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Correct format for CNN (lead, time)
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))


# ---------- LIGHTNING MODULE ----------

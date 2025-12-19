# import required libraries 

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class churnANN(nn.Module):
  def __init__(self, input_dim):
    super(churnANN, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_dim, 64),   # fully connnected layer 
      nn.ReLU(),                  # Rectified Linear Unit
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
  def forward(self, x):
    return self.model(x)
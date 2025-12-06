import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric

from data_utils.preprocess import normalize


class MLP(nn.Module):
    def __init__(self, num_electrodes=62, datapoints=128, num_classes=3, hidden_dims=[512, 256, 128], dropout=0.5):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.datapoints = datapoints
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Calculate input size based on number of electrodes and datapoints
        input_size = num_electrodes * datapoints
        
        # Create hidden layers
        layers = []
        current_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)

    def get_param(self):
        return

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape -> (batch_size, channels, datapoints)
        batch_size = x.shape[0]
        # Flatten the input: (batch_size, channels, datapoints) -> (batch_size, channels * datapoints)
        x = x.reshape(batch_size, -1)
        # Pass through MLP
        x = self.mlp(x)
        return x
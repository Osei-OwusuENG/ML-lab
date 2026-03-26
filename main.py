import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# creating an seq2seq model
class simplelstmModel(nn.modules):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM()

    def forward(self, X):
        
        output, (hn, cn) = self.lstm(X)

        return (hn, cn)

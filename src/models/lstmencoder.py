import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np



# creating an seq2seq model
class lstmencoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):

        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)

        return hidden, cell
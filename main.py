import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# creating an seq2seq model
class lstmencoderModel(nn.modules):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, X):
        
        output, (hidden, cell) = self.lstm(X)

        return hidden, cell
    

class decoderModel(nn.modules):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell
    

class seq2seqModel(nn.modules):
    def __init__(self, encoder, decoder):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len, output_dim = trg.shape

        outputs = torch.zeros(batch_size, trg_len, output_dim)

        hidden, cell = self.encoder(src)

        input = trg[:, 0:1, :] # first token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t:t+1, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            input = trg[:, t:t+1, :] if teacher_force else output

        return outputs



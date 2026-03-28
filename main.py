import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# creating an seq2seq model
class lstmencoderModel(nn.module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):

        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)

        return hidden, cell
    

class decoderModel(nn.module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell
    

class seq2seqModel(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)

        batch_size, trg_len = trg.shape

        outputs = torch.zeros(batch_size, trg_len, self.vocab_size)

        hidden, cell = self.encoder(src)

        # first token (<sos>)
        input = trg[:, 0]   # (batch,)

        for t in range(1, trg_len):
            input = input.unsqueeze(1)  # (batch, 1)

            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[:, t:t+1, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            top1 = output.argmax(2).squeeze(1)

            input = trg[:, t] if teacher_force else top1

        return outputs



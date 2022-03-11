# -*- coding: utf-8 -*-
# @Time : 2021/12/17 11:10
# @Author : Cao yu
# @File : transformer.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model=300, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        #  x: Tensor, shape [seq_len, batch_size, embedding_dim]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, label_size, mode='gru', bidirectional=True, cuda=True, is_training=True,
                 intent_size=26):
        super(Transformer, self).__init__()
        self.is_training = is_training
        embedding_dim = 300
        hidden_size = 300

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          bidirectional=False,
                          batch_first=True)

        self.fc_slot = nn.Linear(300, label_size)
        self.fc_intent = nn.Linear(300, 26)
        self.position = PositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, X):
        embed = self.embedding(X)

        embed = self.position(embed)

        embed = self.transformer_encoder(embed)  # 100

        _, intent_outs = self.rnn(embed)

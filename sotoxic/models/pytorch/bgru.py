import importlib
from sotoxic.models.pytorch import gru
from sotoxic.models.pytorch import attention
importlib.reload(gru)
importlib.reload(attention)

import torch
from collections import OrderedDict
from torch import nn

import importlib

from sotoxic.models import torch_base
from sotoxic.models.pytorch import gru
from sotoxic.models.pytorch import attention
importlib.reload(gru)
importlib.reload(attention)


class GRUClassifier(torch_base.BaseModel):

    def __init__(self, input_size, hidden_size, embedding):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, num_layers=2, dropout=0.35, bidirectional=True)
        self.attn = attention.DotAttention(hidden_size=2*hidden_size)

        self.classifier = nn.Sequential(
            OrderedDict([
                ('gru_dropout', nn.Dropout(0.5)),
                ('h1', nn.Linear(self.hidden_size * 6, 108)),
                ('relu1', nn.ReLU()),
                ('out', nn.Linear(108, 6)),
            ]))

    def set_dropout(self, ratio1):
        pass

    def forward(self, _input, hidden=None, lengths=None):
        _input, lengths = _input
        embedded = self.embedding(_input)

        out, _ = self.gru(embedded)
        last = out[:, -1, :]
        attn, _ = self.attn.forward(out)
        max, _ = torch.max(out, dim=1)
        concatenated = torch.cat([last, max, attn], dim=1)
        result = self.classifier(concatenated)
        return result


class BayesianGRUClassifier(torch_base.BaseModel):

    def __init__(self, input_size, hidden_size, embedding):
        super(BayesianGRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        #self.embedding_dropout = nn.Dropout(0.15)

        self.gru_1 = gru.BiBayesianGRU(input_size=input_size, hidden_size=hidden_size, dropout=0.3)
        self.gru_2 = gru.BiBayesianGRU(input_size=2 * hidden_size, hidden_size=hidden_size, dropout=0.3)
        #self.attn = attention.Attention(attention_size=2 * hidden_size)

        self.classifier = nn.Sequential(
            OrderedDict([
                ('gru_dropout', nn.Dropout(0.5)),
                ('h1', nn.Linear(self.hidden_size * 4, 72)),
                ('relu1', nn.ReLU()),
                ('out', nn.Linear(72, 6)),
            ]))

    def set_dropout(self, ratio1):
        self.gru_1.set_dropout(ratio1)
        self.gru_2.set_dropout(ratio1)

    def forward(self, _input, hidden=None, lengths=None):
        _input, lengths = _input
        embedded = self.embedding(_input)

        out1, _ = self.gru_1.forward(embedded, lengths=lengths)
        out2, _ = self.gru_2.forward(out1, lengths=lengths)

        last = out2[:, -1, :]
        #attn, _ = self.attn(out2, lengths)
        max, _ = torch.max(out2, dim=1)

        concatenated = torch.cat([last, max], dim=1)

        result = self.classifier(concatenated)
        return result

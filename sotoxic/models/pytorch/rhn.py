import torch
from torch.autograd import Variable
from torch import nn
from collections import OrderedDict
from sotoxic.models.pytorch import dropout as dr
from sotoxic.models.pytorch import attention
import importlib

from sotoxic.models import torch_base
importlib.reload(attention)


class RecurrentHighwayClassifier(torch_base.BaseModel):

    def __init__(self, input_size, hidden_size, recurrence_length, embedding, recurrent_dropout=0.3):
        super(RecurrentHighwayClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = recurrence_length
        self.recurrent_dropout = recurrent_dropout
        self.highways = nn.ModuleList()
        self.highways.append(RHNCell(self.input_size, self.hidden_size, is_first_layer=True, recurrent_dropout=recurrent_dropout))

        for _ in range(self.L - 1):
            self.highways.append(RHNCell(self.input_size, self.hidden_size, is_first_layer=False, recurrent_dropout=recurrent_dropout))

        self.embedding = embedding

        self.classifier = nn.Sequential(
            OrderedDict([
                ('h1_dropout', nn.Dropout(0.5)),
                ('h1', nn.Linear(self.hidden_size * 4, 74)),
                ('relu1', nn.ReLU()),
                ('out', nn.Linear(74, 6)),
            ]))

    def init_state(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        return hidden

    def train_mode(self, prob):
        self.set_dropout(prob)
        self.train()

    def set_dropout(self, p):
        for rhn_cell in self.highways:
            rhn_cell.set_dropout(p)

    def eval_mode(self):
        self.eval()

    def forward(self, _input, hidden=None, lengths=None):
        _input, lengths = _input
        batch_size = _input.size(0)
        max_time = _input.size(1)

        if hidden is None:
            hidden = self.init_state(batch_size)
        embed_batch = self.embedding(_input)

        lefts = []
        rights = []

        for time in range(max_time):
            for tick in range(self.L):
                next_hidden = self.highways[tick](embed_batch[:, time, :], hidden)
                mask = (time < lengths).float().unsqueeze(1).expand_as(next_hidden)
                hidden = next_hidden * mask + hidden * (1 - mask)  # for mask the padding dynamically
            lefts.append(hidden.unsqueeze(1))
        lefts = torch.cat(lefts, 1)

        for rhn_cell in self.highways:
            rhn_cell.end_of_sequence()

        for time in range(max_time):
            for tick in range(self.L):
                next_hidden = self.highways[tick](embed_batch[:, time, :], hidden)
                mask = (time < lengths).float().unsqueeze(1).expand_as(next_hidden)
                hidden = next_hidden * mask + hidden * (1 - mask)  # for mask the padding dynamically
            rights.append(hidden.unsqueeze(1))
        rights = torch.cat(rights, 1)

        for rhn_cell in self.highways:
            rhn_cell.end_of_sequence()

        outputs = torch.cat((lefts, rights), dim=2)

        last = outputs[:, -1, :]
        max, _ = torch.max(outputs, dim=1)

        concatenated = torch.cat([last, max], dim=1)
        result = self.classifier(concatenated)
        return result


class RHNCell(nn.Module):

    def __init__(self, input_size, hidden_size, is_first_layer, recurrent_dropout):
        super(RHNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_first_layer = is_first_layer

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.set_dropout(recurrent_dropout)

        # input weight matrices
        if self.is_first_layer:
            self.W_H = nn.Linear(input_size, hidden_size)
            self.W_C = nn.Linear(input_size, hidden_size)

        # hidden weight matrices
        self.R_H = nn.Linear(hidden_size, hidden_size, bias=True)
        self.R_C = nn.Linear(hidden_size, hidden_size, bias=True)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = dr.SequentialDropout(p=dropout)
        self.drop_ii = dr.SequentialDropout(p=dropout)
        self.drop_hr = dr.SequentialDropout(p=dropout)
        self.drop_hi = dr.SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()

    def forward(self, _input, prev_hidden):
        c_i = self.drop_hr(prev_hidden)
        h_i = self.drop_hi(prev_hidden)

        if self.is_first_layer:
            x_i = self.drop_ii(_input)
            x_r = self.drop_ir(_input)
            hl = self.tanh(self.W_H(x_i) + self.R_H(h_i))
            tl = self.sigmoid(self.W_C(x_r) + self.R_C(c_i))
        else:
            hl = self.tanh(self.R_H(h_i))
            tl = self.sigmoid(self.R_C(c_i))

        h = (hl * tl) + (prev_hidden * (1 - tl))
        return h

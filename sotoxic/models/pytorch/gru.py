import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sotoxic.models.pytorch.dropout import SequentialDropout


class AbstractGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # Modules
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError


class GRUCell(AbstractGRUCell):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(GRUCell, self).__init__(input_size, hidden_size,
                                      bias_ih, bias_hh)

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        r = F.sigmoid(self.weight_ir(x) + self.weight_hr(hx))
        i = F.sigmoid(self.weight_ii(x) + self.weight_hi(hx))
        n = F.tanh(self.weight_in(x) + r * self.weight_hn(hx))
        hx = (1 - i) * n + i * hx
        return hx


class BayesianGRUCell(AbstractGRUCell):
    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False,
                       dropout=0.25):
        super(BayesianGRUCell, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)
        self.set_dropout(dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = SequentialDropout(p=dropout)
        self.drop_ii = SequentialDropout(p=dropout)
        self.drop_in = SequentialDropout(p=dropout)
        self.drop_hr = SequentialDropout(p=dropout)
        self.drop_hi = SequentialDropout(p=dropout)
        self.drop_hn = SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_in.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()
        self.drop_hn.end_of_sequence()

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        x_ir = self.drop_ir(x)
        x_ii = self.drop_ii(x)
        x_in = self.drop_in(x)
        x_hr = self.drop_hr(hx)
        x_hi = self.drop_hi(hx)
        x_hn = self.drop_hn(hx)
        r = F.sigmoid(self.weight_ir(x_ir) + self.weight_hr(x_hr))
        i = F.sigmoid(self.weight_ii(x_ii) + self.weight_hi(x_hi))
        n = F.tanh(self.weight_in(x_in) + r * self.weight_hn(x_hn))
        hx = (1 - i) * n + i * hx
        return hx


class AbstractGRU(nn.Module):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(AbstractGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        raise NotImplementedError

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:,i,:], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx


class GRU(AbstractGRU):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(GRU, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = GRUCell(self.input_size, self.hidden_size,
                                self.bias_ih, self.bias_hh)


class BiBayesianGRU(AbstractGRU):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False,
                       dropout=0.25):
        self.dropout = dropout
        self.hidden_size = hidden_size
        super(BiBayesianGRU, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = BayesianGRUCell(self.input_size, self.hidden_size,
                                        self.bias_ih, self.bias_hh,
                                        dropout=self.dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gru_cell.set_dropout(dropout)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()

    def forward(self, x, hx=None, max_length=None, lengths=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        lefts = []
        rights = []

        '''
        max len = 8
        A C E P P P P P l=3
        B D F G P P P P l=4
        '''

        # left part
        lhx = self.init_hidden(batch_size)
        for time in range(max_length):
            new_hx = self.gru_cell(x[:, time, :], hx=lhx)
            #print("X", x[:, time, :])
            #print("H", new_hx)
            mask = (time < lengths).float().unsqueeze(1).expand_as(new_hx)
            #print("M", mask)
            lhx = new_hx * mask + lhx * (1 - mask) # for mask the padding dynamically
            lefts.append(lhx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
        lefts = torch.cat(lefts, 1)


        # right part
        rhx = self.init_hidden(batch_size)
        for time in range(max_length - 1, -1, -1):
            new_hx = self.gru_cell(x[:, time, :], hx=rhx)
            mask = (time < lengths).float().unsqueeze(1).expand_as(new_hx)
            rhx = new_hx * mask + rhx * (1 - mask)
            rights.append(rhx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
        rights = torch.cat(rights, 1)


        output = torch.cat((lefts, rights), dim=2)
        return output, lhx


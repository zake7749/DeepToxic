import torch
import numpy as np
import importlib

from torch.autograd import Variable
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from sotoxic.config import model_config
from sotoxic.models.pytorch import dropout
importlib.reload(dropout)


def mini_batches_generator(inputs, targets, batch_size, row_shuffle=False):
    inputs_data_size = len(inputs)
    targets_data_size = len(targets)
    assert inputs_data_size == targets_data_size, "The length of inputs({}) and targets({}) must be consistent".format(
        inputs_data_size, targets_data_size)

    if row_shuffle:
        for input_seqs in inputs:
            np.random.shuffle(input_seqs)

    shuffled_input, shuffled_target = shuffle(inputs, targets)
    mini_batches = [
        (shuffled_input[k: k + batch_size], shuffled_target[k: k + batch_size])
        for k in range(0, inputs_data_size, batch_size)
    ]
    dp = dropout.EmbeddingDropout(p=0.2)

    for batch_xs, batch_ys in mini_batches:
        lengths = [len(s) for s in batch_xs]
        max_length = min(model_config.MAX_SENTENCE_LENGTH, max(lengths))
        batch_tensors = pad_sequences(batch_xs, maxlen=max_length, padding='post', truncating='pre')

        lengths_var = Variable(torch.Tensor(lengths), requires_grad=False)
        inputs_tensor = torch.from_numpy(batch_tensors).long()
        inputs_dropped_tensor = dp.forward(inputs_tensor)
        inputs_var = Variable(inputs_dropped_tensor)
        targets_var = Variable(torch.from_numpy(batch_ys).float())

        if model_config.use_cuda:
            inputs_var = inputs_var.cuda()
            targets_var = targets_var.cuda()
            lengths_var = lengths_var.cuda()
        yield (inputs_var, lengths_var), targets_var


def test_batches_generator(inputs, batch_size):
    inputs_data_size = len(inputs)
    mini_batches = [inputs[k: k + batch_size] for k in range(0, inputs_data_size, batch_size)]

    for batch_xs in mini_batches:
        lengths = [len(s) for s in batch_xs]
        max_length = min(model_config.MAX_SENTENCE_LENGTH, max(lengths))
        batch_tensors = pad_sequences(batch_xs, maxlen=max_length, padding='post', truncating='pre')

        lengths_var = Variable(torch.Tensor(lengths))
        inputs_var = Variable(torch.from_numpy(batch_tensors).long())

        if model_config.use_cuda:
            inputs_var = inputs_var.cuda()
            lengths_var = lengths_var.cuda()

        yield (inputs_var, lengths_var)

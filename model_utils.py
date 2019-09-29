import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data as tud
import torch.nn.utils.rnn as rnn_utils

import numpy as np
import math
import random


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)

        if module.bias is not None:
            nn.init.normal_(module.bias.data)

        print('initialized Linear')

    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        print('initialized Conv')

    elif isinstance(module, nn.RNNBase) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.normal_(param.data)

        print('initialized LSTM')

    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        print('initialized BatchNorm')


def init_params_uniformly(module):
    init = lambda x: nn.init.uniform_(x, -1, 1)

    if isinstance(module, nn.Linear):
        init(module.weight.data)

        if module.bias is not None:
            init(module.bias.data)

        print('initialized Linear')

    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        init(module.weight)
        print('initialized Conv')

    elif isinstance(module, nn.RNNBase) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                init(param.data)
            elif 'bias' in name:
                init(param.data)

        print('initialized LSTM')

    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        init(module.weight.data)
        print('initialized BatchNorm')


def build_len_masks_batch(
    # [batch_size], []
    len_batch, max_len=None
):
    if max_len is None:
        max_len = len_batch.max().item()
    # try:
    batch_size, = len_batch.shape
    # [batch_size, max_len]
    idxes_batch = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(len_batch.device)
    # [batch_size, max_len] = [batch_size, max_len] < [batch_size, 1]
    return idxes_batch < len_batch.view(-1, 1)

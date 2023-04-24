from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.0001)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        in_features = n_inputs
        for (i, out_features) in enumerate(n_hidden):
            self.layers.add_module('linear' + str(i), nn.Linear(in_features, out_features))
            self.layers.add_module('relu' + str(i), nn.ReLU())
            in_features = out_features
        self.layers.add_module('linear' + str(len(n_hidden)), nn.Linear(in_features, n_classes))
        self.layers.add_module('softmax', nn.Softmax(dim=1))
        self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out

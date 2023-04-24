from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import argparse
import numpy as np
import torch.nn as nn
from pytorch_mlp import MLP
from torch.optim import SGD
from sklearn import datasets, model_selection

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    predicted_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    return torch.mean((predicted_labels == true_labels).float())


def one_hot(input, num_classes=2):
    return np.eye(num_classes)[input]


def train():
    dnn_hidden_units = FLAGS.dnn_hidden_units
    n_hidden = dnn_hidden_units.split(",")
    n_hidden = [int(x) for x in n_hidden]
    learning_rate = FLAGS.learning_rate
    max_epochs = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq

    inputs, labels = datasets.make_moons(n_samples=1000, shuffle=True)
    mlp = MLP(2, n_hidden, 2)

    optimizer = SGD(mlp.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, shuffle=True, test_size=0.2)
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    n_samples = x_train.shape[0]
    best_train_loss, best_train_acc = 1e5, 0
    for epoch in range(max_epochs):
        total_loss = 0
        total_acc = 0
        shuffled_indices = np.random.permutation(n_samples)
        for i in range(n_samples):
            idx = shuffled_indices[i]
            x = x_train[idx]
            y = y_train[idx]
            x, y = x[np.newaxis, :], y[np.newaxis, :]
            x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            pd = mlp(x)
            loss = criterion(pd, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(pd, y)
            total_acc += acc
        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        if best_train_loss > avg_loss:
            best_train_loss = avg_loss
        if best_train_acc < avg_acc:
            best_train_acc = avg_acc
        if epoch % eval_freq == 0 or epoch == (max_epochs - 1):
            print(f'SGD Epoch {epoch}:')
            print(f'Average Train Loss: {avg_loss} | '
                  f'Average Train Accuracy: {avg_acc}')


def main():
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()

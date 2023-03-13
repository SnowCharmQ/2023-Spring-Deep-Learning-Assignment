from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from sklearn import datasets

from optimizer import *
from mlp_numpy import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'SGD'

FLAGS = None


def train():
    dnn_hidden_units = FLAGS.dnn_hidden_units
    n_hidden = dnn_hidden_units.split(",")
    n_hidden = [int(x) for x in n_hidden]
    learning_rate = FLAGS.learning_rate
    max_steps = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq
    optimizer = FLAGS.optimizer

    inputs, labels = datasets.make_moons(n_samples=1000, shuffle=True)

    mlp = MLP(2, n_hidden, 2)
    if optimizer.lower() == 'bgd':
        optimizer = BGD(mlp)
    elif optimizer.lower() == 'sgd':
        optimizer = SGD(mlp)
    else:
        raise TypeError
    optimizer.optimize(inputs, labels, learning_rate, max_steps, eval_freq)


def main():
    """
    Main function
    """
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
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER_DEFAULT,
                        help='Optimizer to optimize the model')
    FLAGS, unparsed = parser.parse_known_args()
    main()

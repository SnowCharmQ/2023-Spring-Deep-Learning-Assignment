from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tqdm
import argparse
import numpy as np
from sklearn import datasets
from sklearn import model_selection

from optimizer import BGD
from mlp_numpy import MLP
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

ONE_HOT_DIM = 2

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    # Find the index of the maximum value in each row of the predictions array
    predicted_labels = np.argmax(predictions, axis=1)
    # Find the index of the maximum value in each row of the targets array
    true_labels = np.argmax(targets, axis=1)
    # Compute the average number of correct predictions
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


def one_hot(input, num_classes=ONE_HOT_DIM):
    return np.eye(num_classes)[input]


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should test the model on the whole test set each eval_freq iterations.
    """
    dnn_hidden_units = FLAGS.dnn_hidden_units
    n_hidden = dnn_hidden_units.split(",")
    n_hidden = [int(x) for x in n_hidden]
    learning_rate = FLAGS.learning_rate
    max_steps = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq

    inputs, labels = datasets.make_moons(n_samples=1000, shuffle=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, shuffle=True, test_size=0.2)
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    train_data = list(zip(x_train, y_train))

    mlp = MLP(2, n_hidden, ONE_HOT_DIM)
    optimizer = BGD(mlp, learning_rate)
    criterion = CrossEntropy()

    train_acc = []
    test_acc = []
    train_loss = []

    for epoch in tqdm.tqdm(range(max_steps)):
        total_loss = 0
        train_pred = np.zeros_like(y_train)
        for i, (x, y) in enumerate(train_data):
            pd = mlp.forward(x)
            loss = criterion.forward(pd, y)
            total_loss += loss
            d_loss = criterion.backward()
            mlp.backward(d_loss)
            optimizer.step()
            train_pred[i] = pd
        acc = accuracy(train_pred, y_train)
        train_acc.append(acc)
        train_loss.append(total_loss)

        if epoch % eval_freq == 0:
            test_pred = np.zeros_like(y_test)
            for i, x in enumerate(x_test):
                test_pred[i] = mlp.forward(x)
            acc = accuracy(test_pred, y_test)
            test_acc.append((epoch, acc))


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
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

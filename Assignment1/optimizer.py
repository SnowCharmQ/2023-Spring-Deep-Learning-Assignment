import numpy as np
from sklearn import model_selection

from modules import CrossEntropy


def one_hot(input, num_classes=2):
    return np.eye(num_classes)[input]


def accuracy(predictions, targets):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return np.mean(predicted_labels == true_labels)


class BGD(object):
    def __init__(self, mlp):
        self.mlp = mlp

    def optimize(self, inputs, labels, lr, max_epochs, batch_size=4):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, shuffle=True, test_size=0.2)
        y_train, y_test = one_hot(y_train), one_hot(y_test)
        num_batches = int(np.ceil(len(x_train) / batch_size))
        criterion = CrossEntropy()
        for epoch in range(max_epochs):
            total_loss = 0
            total_acc = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_train))
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                pd_batch = self.mlp.forward(x_batch)
                loss = criterion.forward(pd_batch, y_batch)
                total_loss += loss
                d_loss = criterion.backward()
                self.mlp.backward(d_loss)
                self.mlp.update(lr)
                acc = accuracy(pd_batch, y_batch)
                total_acc += acc
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            print(f'Epoch {epoch}: Average Loss: {avg_loss} , Average Accuracy: {avg_acc}')
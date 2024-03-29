import numpy as np
from sklearn import model_selection

from modules import CrossEntropy
from matplotlib import pyplot as plt


def one_hot(input, num_classes=2):
    return np.eye(num_classes)[input]


def accuracy(predictions, targets):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return np.mean(predicted_labels == true_labels)


def draw(train_loss, test_loss, train_acc, test_acc, epochs):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, color='red', alpha=0.8, linewidth=1, label='Train Loss')
    plt.plot(epochs, test_loss, color='blue', alpha=0.8, linewidth=1, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Chart')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, color='red', alpha=0.8, linewidth=1, label='Train Accuracy')
    plt.plot(epochs, test_acc, color='blue', alpha=0.8, linewidth=1, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Chart')
    plt.show()


class BGD(object):
    def __init__(self, mlp):
        self.mlp = mlp

    def optimize(self, inputs, labels, lr, max_epochs, eval_freq):
        batch_size = len(inputs)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, shuffle=True, test_size=0.2)
        y_train, y_test = one_hot(y_train), one_hot(y_test)
        num_batches = int(np.ceil(len(x_train) / batch_size))
        criterion = CrossEntropy()
        train_loss = []
        best_train_loss = 1e5
        train_acc = []
        best_train_acc = 0
        test_loss = []
        best_test_loss = 1e5
        test_acc = []
        best_test_acc = 0
        epochs = []
        for epoch in range(max_epochs):
            epochs.append(epoch)
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
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            if best_train_loss > avg_loss:
                best_train_loss = avg_loss
            if best_train_acc < avg_acc:
                best_train_acc = avg_acc
            pd_test = self.mlp.forward(x_test)
            total_loss = criterion.forward(pd_test, y_test)
            total_acc = accuracy(pd_test, y_test)
            test_loss.append(total_loss)
            test_acc.append(total_acc)
            if best_test_loss > total_loss:
                best_test_loss = total_loss
            if best_test_acc < total_acc:
                best_test_acc = total_acc
            if epoch % eval_freq == 0 or epoch == (max_epochs - 1):
                print(f'BGD Epoch {epoch}:')
                print(f'Average Train Loss: {avg_loss} | '
                      f'Average Train Accuracy: {avg_acc}')
                print(f'Average Test Loss: {total_loss} | '
                      f'Average Test Accuracy: {total_acc}')
                print()
        draw(train_loss, test_loss, train_acc, test_acc, epochs)
        print(f'Best Train Loss: {best_train_loss} | '
              f'Best Train Accuracy: {best_train_acc}\n'
              f'Best Test Loss: {best_test_loss} | '
              f'Best Test Accuracy: {best_test_acc}')


class SGD(object):
    def __init__(self, mlp):
        self.mlp = mlp

    def optimize(self, inputs, labels, lr, max_epochs, eval_freq):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, shuffle=True, test_size=0.2)
        y_train, y_test = one_hot(y_train), one_hot(y_test)
        criterion = CrossEntropy()
        train_loss = []
        best_train_loss = 1e5
        train_acc = []
        best_train_acc = 0
        test_loss = []
        best_test_loss = 1e5
        test_acc = []
        best_test_acc = 0
        epochs = []
        n_samples = x_train.shape[0]
        for epoch in range(max_epochs):
            epochs.append(epoch)
            total_loss = 0
            total_acc = 0
            shuffled_indices = np.random.permutation(n_samples)
            for i in range(n_samples):
                idx = shuffled_indices[i]
                x = x_train[idx]
                y = y_train[idx]
                x, y = x[np.newaxis, :], y[np.newaxis, :]
                pd = self.mlp.forward(x)
                loss = criterion.forward(pd, y)
                total_loss += loss
                d_loss = criterion.backward()
                self.mlp.backward(d_loss)
                self.mlp.update(lr)
                acc = accuracy(pd, y)
                total_acc += acc
            avg_loss = total_loss / n_samples
            avg_acc = total_acc / n_samples
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            if best_train_loss > avg_loss:
                best_train_loss = avg_loss
            if best_train_acc < avg_acc:
                best_train_acc = avg_acc
            pd_test = self.mlp.forward(x_test)
            total_loss = criterion.forward(pd_test, y_test)
            total_acc = accuracy(pd_test, y_test)
            test_loss.append(total_loss)
            test_acc.append(total_acc)
            if best_test_loss > total_loss:
                best_test_loss = total_loss
            if best_test_acc < total_acc:
                best_test_acc = total_acc
            if epoch % eval_freq == 0 or epoch == (max_epochs - 1):
                print(f'SGD Epoch {epoch}:')
                print(f'Average Train Loss: {avg_loss} | '
                      f'Average Train Accuracy: {avg_acc}')
                print(f'Average Test Loss: {total_loss} | '
                      f'Average Test Accuracy: {total_acc}')
                print()
        draw(train_loss, test_loss, train_acc, test_acc, epochs)
        print(f'Best Train Loss: {best_train_loss} | '
              f'Best Train Accuracy: {best_train_acc}\n'
              f'Best Test Loss: {best_test_loss} | '
              f'Best Test Accuracy: {best_test_acc}')

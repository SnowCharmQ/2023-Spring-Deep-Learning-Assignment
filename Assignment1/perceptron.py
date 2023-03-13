import numpy as np
import matplotlib.pyplot as plt


def decision_boundary(model, x):
    return - (model.weights[0] + model.weights[1] * x) / model.weights[2]


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        self.n_inputs = n_inputs
        self.weights = np.random.normal(0, 1, n_inputs + 1)
        self.max_epochs = max_epochs
        self.lr = learning_rate

    def forward(self, input):
        label = np.dot(self.weights[1:], input) + self.weights[0]
        return label

    def train(self, training_inputs, labels):
        for epoch in range(int(self.max_epochs)):
            for x, y in zip(training_inputs, labels):
                pred = self.forward(x)
                mask = y * pred <= 0
                self.weights[0] += self.lr * np.sum(y[mask])
                self.weights[1:] += self.lr * np.dot(y[mask], x[mask])

    def test(self, testing_inputs, labels):
        cnt = 0
        for x, y in zip(testing_inputs, labels):
            pred = self.forward(x)
            pred = np.where(pred >= 0, 1, -1)
            if int(pred) == int(y):
                cnt += 1
        return cnt


mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
mean2 = [3, 3]
cov2 = [[1, 0], [0, 1]]

for i in range(10):
    print(f'TEST {i + 1}')

    data1 = np.random.multivariate_normal(mean1, cov1, 100)
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    train_data = np.concatenate((data1[:80], data2[:80]))
    train_labels = np.concatenate((np.ones(80), -np.ones(80)))
    test_data = np.concatenate((data1[80:], data2[80:]))
    test_labels = np.concatenate((np.ones(20), -np.ones(20)))

    shuffle_index = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_index]
    train_labels = train_labels[shuffle_index]

    shuffle_index = np.random.permutation(len(test_data))
    test_data_shuffle = test_data[shuffle_index]
    test_labels_shuffle = test_labels[shuffle_index]

    model = Perceptron(2)
    model.train(train_data, train_labels)
    cnt = model.test(test_data_shuffle, test_labels_shuffle)
    print("The number of data points is {}, the number of correct predictions is {}, "
          "the classification accuracy is {}".format(40, cnt, cnt / 40))

    if i == 0:
        x_vals = np.linspace(-5, 5, 100)
        y_vals = decision_boundary(model, x_vals)
        plt.plot(x_vals, y_vals, '-', c='black')
        plt.scatter(test_data[:20, 0], test_data[:20, 1], c='blue', label='1')
        plt.scatter(test_data[20:, 0], test_data[20:, 1], c='red', label='-1')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Decision Boundary')
        plt.legend(loc='lower left')
        plt.show()

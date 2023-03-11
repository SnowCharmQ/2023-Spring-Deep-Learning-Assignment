import numpy as np

from modules import Linear


class BGD(object):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.params['weight'] -= self.lr * layer.grads['weight']
                layer.params['bias'] -= self.lr * layer.grads['bias']
                layer.grads['weight'] = np.zeros_like(layer.params['weight'])
                layer.grads['bias'] = np.zeros_like(layer.params['bias'])

import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        self.params = {'weight': np.random.normal(loc=0, scale=0.0001, size=(in_features, out_features)),
                       'bias': np.zeros(out_features)}
        self.grads = {'weight': np.zeros_like(self.params['weight']),
                      'bias': np.zeros_like(self.params['bias'])}
        self.x = None

    def forward(self, x):
        out = np.dot(x, self.params['weight']) + self.params['bias']
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params['weight'].T)
        self.grads['weight'] = np.outer(self.x, dout)
        self.grads['bias'] = dout
        return dx


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.maximum(0, x)
        return out

    def backward(self, dout):
        dx = dout * (self.x > 0)
        return dx


class SoftMax(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        exp_vals = np.exp(x - np.max(x))
        self.out = exp_vals / np.sum(exp_vals)
        return self.out

    def backward(self, dout):
        out = self.out.reshape(1, self.out.size)
        jacobian = - np.dot(out.reshape(-1, 1), out.reshape(1, -1))
        diag_indices = np.diag_indices_from(jacobian)
        jacobian[diag_indices] = out * (1 - out)
        dx = np.dot(dout, jacobian)
        return dx


class CrossEntropy(object):
    def __init__(self):
        self.pd = None
        self.gt = None

    def forward(self, pd, gt):
        self.pd = pd
        self.gt = gt
        eps = 1e-9
        loss = - np.sum(gt * np.log(pd + eps))
        return loss

    def backward(self):
        eps = 1e-9
        d_loss = - self.gt / (self.pd + eps)
        return d_loss

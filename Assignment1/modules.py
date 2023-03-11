import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = {}
        self.params['weight'] = np.random.normal(loc=0, scale=0.0001, size=(in_features, out_features))
        self.params['bias'] = np.zeros(out_features)
        self.grads = {}
        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])
        self.x = None

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation.
        This is true for *all* forward methods of *all* modules in this class
        """
        out = np.dot(x, self.params['weight']) + self.params['bias']
        self.x = x
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        dx = np.dot(dout, self.params['weight'].T)
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0)
        return dx


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        out = np.maximum(0, self.x)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = dout * (self.x > 0)
        return dx


class SoftMax(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = self.out * dout - self.out * np.sum(dout * self.out, axis=-1, keepdims=True)
        return dx

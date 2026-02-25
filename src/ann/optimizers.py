"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
# define diff class for each optimizers.


import numpy as np  

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            # apply weigh decay if specified
            layer.grad_W += self.weight_decay * layer.W

            # update rule SGD
            # what about mini batch ? the batch size is already decided in the outer neural_network class, also the gradient (averaged over the batch) is already computed in the backward pass of the neural layer, so we can just use that gradient to update the weights and bias here, we dont need to worry about batch size here
            layer.W -= self.learning_rate * layer.grad_W
            layer.bias -= self.learning_rate * layer.grad_b

class Momentum:
    
    # very similar, but we need to maintain and update velocity
    def __init__(self, learning_rate=0.01, weight_decay=0.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = momentum

    def update(self, layers):
        # LAZY INITIALIZATION: If this is epoch 1, batch 1, the memory is empty.
        # We must create zero-matrices the exact same shape as the layer's W and b.
        velocity_w = {}
        velocity_b = {}

        for i,layer in enumerate(layers):
            # for first time, we need to initialize as zeros, we need to check if it present already
            if i not in velocity_w:
                velocity_w[i] = np.zeros_like(layer.W)
                velocity_b[i] = np.zeros_like(layer.bias)
            # apply weight decay if specified (to make large gradients and large updates)
            layer.grad_W += self.weight_decay * layer.W
            # update velocity   
            # from old velocity matrix to new 
            velocity_w[i] = self.learning_rate * self.grad_W + self.gamma * velocity_w[i]
            velocity_b[i] = self.learning_rate * self.grad_b + self.gamma * velocity_b[i]
            # update weights and bias using velocity
            layer.W = layer.W - velocity_w[i]
            layer.bias = layer.bias - velocity_b[i]

class Nag:
    # to be implemented
    pass
class Adam:
    # to be implemented
    pass
class Nadam:
    # to be implemented
    pass

class RMSprop:
    # to be implemented
    pass
        

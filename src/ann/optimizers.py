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
        self.velocity_w = {}
        self.velocity_b = {}

    def update(self, layers):
        # LAZY INITIALIZATION: If this is epoch 1, batch 1, the memory is empty.
        # We must create zero-matrices the exact same shape as the layer's W and b.

        for i,layer in enumerate(layers):
            # for first time, we need to initialize as zeros, we need to check if it present already
            if i not in self.velocity_w:
                self.velocity_w[i] = np.zeros_like(layer.W)
                self.velocity_b[i] = np.zeros_like(layer.bias)
            # apply weight decay if specified (to make large gradients and large updates)
            layer.grad_W += self.weight_decay * layer.W
            # update velocity   
            # from old velocity matrix to new 
            self.velocity_w[i] = self.learning_rate * layer.grad_W + self.gamma * self.velocity_w[i]
            self.velocity_b[i] = self.learning_rate * layer.grad_b + self.gamma * self.velocity_b[i]
            # update weights and bias using velocity
            layer.W = layer.W - self.velocity_w[i]
            layer.bias = layer.bias - self.velocity_b[i]

class NAG:
    def __init__(self, learning_rate=0.01, weight_decay=0.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = momentum
        self.velocity_w = {}
        self.velocity_b = {}
    def update(self, layers):
        # LAZY INITIALIZATION: If this is epoch 1, batch 1, the memory is empty.
        # We must create zero-matrices the exact same shape as the layer's W and b.

        for i,layer in enumerate(layers):
            # for first time, we need to initialize as zeros, we need to check if it present already
            if i not in self.velocity_w:
                self.velocity_w[i] = np.zeros_like(layer.W)
                self.velocity_b[i] = np.zeros_like(layer.bias)
            # apply weight decay if specified (to make large gradients and large updates)
            layer.grad_W += self.weight_decay * layer.W
            # update velocity   
            # from old velocity matrix to new 
            self.velocity_w[i] = self.learning_rate * layer.grad_W + self.gamma * self.velocity_w[i]
            self.velocity_b[i] = self.learning_rate * layer.grad_b + self.gamma * self.velocity_b[i]
            # update weights and bias using NAG update rule
            # excellent mathematical trick to compute the look ahead gradient without actually computing the look ahead weights, we can directly use the current velocity to compute the look ahead gradient, and then use that look ahead gradient to update the weights and bias, this is because the velocity is already a combination of the current gradient and the past velocity, so it already contains the information about the look ahead position, so we can directly use it to compute the look ahead gradient.
            layer.W = layer.W - (self.gamma * self.velocity_w[i] + self.learning_rate * layer.grad_W)
            layer.bias = layer.bias - (self.gamma * self.velocity_b[i] + self.learning_rate * layer.grad_b)
            # here the weight matrix is not the normal weights, but they are weights - gamma*velocity, which is the look ahead position, and we are using the gradient at the current position to update the weights, which is the NAG update rule. This is a very clever trick to compute the look ahead gradient without actually computing the look ahead weights, which saves a lot of computation and also gives better convergence properties.

# ADAGRAD
# RMSPROP
# ADAM
# NADAM
class Adam:
    # to be implemented
    def __init__(self, learning_rate=0.01, weight_decay=0.0, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0 # time step counter for bias correction

    def update(self, layers):
        
        self.t += 1
        for i,layer in enumerate(layers):
            if i not in self.m_w:
                self.m_w[i] = np.zeros_like(layer.W)
                self.v_w[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.bias)
                self.v_b[i] = np.zeros_like(layer.bias)

            # apply weight decay if specified
            layer.grad_W += self.weight_decay * layer.W

            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b

            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update weights and bias using Adam update rule
            layer.W -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Nadam:
    # to be implemented
    pass

class RMSprop:
    # to be implemented
    def __init__(self, learning_rate=0.01, weight_decay=0.0, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = decay_rate
        self.velocity_w = {}
        self.velocity_b = {}

    def update(self, layers):
        for i,layer in enumerate(layers):
            if i not in self.velocity_w:
                self.velocity_w[i] = np.zeros_like(layer.W)
                self.velocity_b[i] = np.zeros_like(layer.bias)
                
            # apply weight decay if specified (to make large gradients and large updates)
            layer.grad_W += self.weight_decay * layer.W
            # update velocity   
            self.velocity_w[i] = self.beta * self.velocity_w[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.velocity_b[i] = self.beta * self.velocity_b[i] + (1 - self.beta) * (layer.grad_b ** 2)
            # update weights and bias using RMSprop update rule
            # epsilon = 1e-8 is added to the denominator to prevent division by zero, and it also helps to stabilize the updates when the velocity is very small, which can happen when the gradients are very small, and it can also help to prevent the updates from becoming too large when the gradients are very large, so it is a very important hyperparameter in RMSprop and Adam optimizers.
            layer.W = layer.W - (self.learning_rate / (np.sqrt(self.velocity_w[i]) + 1e-8)) * layer.grad_W
            layer.bias = layer.bias - (self.learning_rate / (np.sqrt(self.velocity_b[i]) + 1e-8)) * layer.grad_b

    pass
        

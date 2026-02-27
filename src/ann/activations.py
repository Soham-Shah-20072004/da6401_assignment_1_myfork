"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np
# ReLU Activation Function and Derivative
def relu(x):
    return np.maximum(0,x) # returns relu value for eache element in X matrix. X hopefully is a matrix of size (neurons,batch)
def relu_derivative(x):
    return (x > 0).astype(float) # returns a matrix of same size as x, with 1 where x>0 and 0 elsewhere

# Sigmoid Activation Function and Derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    h = sigmoid(x)
    return h*(1-h)
# Tanh Activation Function and Derivative
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    # der(tanh) = sech^2(x) = 1 - tanh^2(x)
    h = tanh(x)
    return (1 - h**2)
# Softmax Activation Function and Derivative
def softmax(x):
    
    shifted_x = x - np.max(x, axis = 0, keepdims=True)
    # this shifting wont cause any change in the derivative because its just a constant shift, and the derivative of a constant is zero, so it wont affect the gradients during backpropagation,
    # in cases of overflow, due to karge values of z (here x), therefore we subtract the max value from each of them
    exp_k = np.exp(shifted_x)
    s = exp_k/np.sum(exp_k, axis=0, keepdims=True)
    return s # returns a matrix of same size as x, where each column is the softmax of the corresponding sample of x in batch
def softmax_derivative(x):
    s = softmax(x)
    # if we dont use the direct short cut formula known to us, then there is need for making 3d matrix at this point and extra work. so lets directly use the result that we know
    # with categorical cross entropy loss, the derivative of softmax is just s - y, where y is the true label in one hot encoding
    return np.ones_like(x) # this is not the actual derivative, but we will use it in combination with the loss function derivative to get the correct gradient during backpropagation

# so what we will do is in back prop step, we will for output layer, 
# we will directly write error of this layer as s - y, 
# where s is the output of softmax and y is the true label in one hot encoding, 
# and for hidden layers we will use the actual derivative of the activation function as usual. 
# his way we can avoid the need for computing the jacobian matrix of softmax which is computationally expensive.

# lets map the string name of activation function to the tuple of function,derivative
# so we can easily get the fn and derivative by name when we are building neural netowrk from terminal args
ACTIVATION_MAPPING = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
}

# to grab that tuple by the string name
def get_activation(name):
    """Get activation function and derivative by name"""
    if name not in ACTIVATION_MAPPING:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATION_MAPPING[name]

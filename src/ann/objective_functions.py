"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
def categorical_cross_entropy(y_true, y_pred):
    # to avoid log(0) which is undefined, we add a small epsilon value to y_pred
    epsilon = 1e-15
    y_inter = y_true * np.log(y_pred + epsilon) # element wise product makes sense !
    # this will give us a matrix of same size as y_true and y_pred, where each element is the product of corresponding elements of y_true and log of corresponding elements of y_pred
    loss = -np.sum(y_inter, axis=0) # this will give us a vector of size (batch_size,) where each element is the sum of corresponding column of y_inter
    return loss # this is a vector of loss values per sample in the batch as column

def categorical_cross_entropy_derivative(y_true, y_pred):
    # the derivative of categorical cross entropy loss with respect to the output of softmax activation function is just s - y, where s is the output of softmax and y is the true label in one hot encoding
    return y_pred - y_true # this will give us a matrix of same size as y_true and y_pred, where each element is the difference between corresponding elements of y_pred and y_true
def mse(y_true, y_pred):
    loss = np.mean((y_true - y_pred)**2, axis=0) # this will give us a vector of size (batch_size,) where each element is the mean of squared differences between corresponding elements of y_true and y_pred for each sample in the batch
    return loss # this is a vector of loss values for each sample in the batch
def mse_derivative(y_true, y_pred):
    return 2*(y_pred - y_true)/y_true.shape[0] # this will give us a matrix of same size as y_true and y_pred, where each element is the derivative of mse loss with respect to corresponding elements of y_pred, which is 2 times the difference between corresponding elements of y_pred and y_true divided by the number of samples in the batch

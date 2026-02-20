"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

# layer should know the input size and its own size, also initialization of weights technique
import numpy as np  
import activations
class Layer:
    def __init__(self, input_size, layer_size, output_layer_size,weight_init = 'zeroes'):
        self.input_size = input_size
        self.layer_size = layer_size
        
        # initialize bias with zeros
        self.bias = np.zeros((layer_size, 1))
        # initialize weights based on the specified technique
        if weight_init == 'zeroes':
            self.W = np.zeroes((layer_size, input_size))
        elif weight_init == 'random':
            self.W = np.random.rand(layer_size, input_size)  # intializes random weights within 0 and 1
        elif weight_init == 'xavier':
            std_dev = np.sqrt(2 / (input_size + output_layer_size))
            self.W = np.random.randn(layer_size, input_size) * std_dev  # Xavier  initialization (because standard generation by randn has variance 1)

        else:
            raise ValueError(f"Unknown weight initialization technique {weight_init}. Use 'zeroes', 'random', or 'xavier'.")
        
        # initialize the gradients of weights and bias of this layer, will get updated during backward()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.bias)

    # definin forward pass method, which takes the input vector and computes the activations/output of this layer
    def forward(self, input_vector):
        self.input_vector = input_vector  
        self.weighted_sum = np.dot(self.W, input_vector) + self.bias  # store the input and weighted sum z for each neuron -- for use in backward pass
        self.output = activations.activation_function(self.weighted_sum)  # applying activation function to the weighted sum to get the output of this layer
        return self.output
    
    # this is not implenenting the whole backprop, but just doing it locally for one layer object
    def backward(self, local_error,z_prev_layer):
        # given the del(local error) for this layer, we compute the gradients of weights and bias for this layer, and also compute the del(local error) to be passed to the previous layer
        # del - layer_size * 1
        # input_vector - input_size * 1
        # grad_W - layer_size * input_size
        # grad_b - layer_size * 1

        # get del for this layer
        # get grad_W and grad_b for this layer
        self.grad_W = np.dot(local_error,self.input_vector.T) # del*X^T
        self.grad_b = local_error # as del = dL/dz, and dz/db=1, dL/dz*dz/db = dL/db = del
        del_prev_layer = np.dot((self.W).T,local_error) * activations.activation_derivative_z(z_prev_layer) # del_prev_layer = W^T * del_this_layer * activation_derivative(z_prev_layer)

        # if successful, we have gradients of weights and bias for this layer stored as attributes of this layer object, and we also have the del(local error) to be passed to the previous layer
        return del_prev_layer
"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

# layer should know the input size and its own size, also initialization of weights technique
import numpy as np  
import activations
class Layer:
    def __init__(self, input_size, layer_size,weight_init = 'zeros',layer_type = 'hidden'):
        self.input_size = input_size
        self.layer_size = layer_size
        self.layer_type = layer_type
        
        # initialize bias with zeros
        self.bias = np.zeros((layer_size, 1))
        # initialize weights based on the specified technique
        if weight_init == 'zeros':
            self.W = np.zeros((layer_size, input_size))
        elif weight_init == 'random':
            self.W = np.random.rand(layer_size, input_size)  # intializes random weights within 0 and 1
        elif weight_init == 'xavier':
            # i learned now that Nout in Xaviers refers to current layer not the last output layer
            std_dev = np.sqrt(2 / (self.input_size + self.layer_size))
            self.W = np.random.randn(layer_size, input_size) * std_dev  # Xavier  initialization (because standard generation by randn has variance 1)

        else:
            raise ValueError(f"Unknown weight initialization technique {weight_init}. Use 'zeroes', 'random', or 'xavier'.")
        
        # initialize the gradients of weights and bias of this layer, will get updated during backward()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.bias)
    
        if self.layer_type == 'hidden':
            self.activation_function = activations.hidden_activation_function
            self.activation_derivative = activations.hidden_activation_derivative
        elif self.layer_type == 'output':
            self.activation_function = activations.output_activation_function
            self.activation_derivative = activations.output_activation_derivative
        
    # defining forward pass method, which takes the input vector and computes the activations/output of this layer
    def forward(self, input_vector):
        self.input_vector = input_vector  
        self.weighted_sum = np.dot(self.W, input_vector) + self.bias  # store the input and weighted sum z for each neuron -- for use in backward pass
        self.output = self.activation_function(self.weighted_sum)  # applying activation function to the weighted sum to get the output of this layer
        return self.output
    
    # this is not implenenting the whole backprop, but just doing it locally for one layer object
    def backward(self, error_w_next):
        # given the del(local error) for this layer, we compute the gradients of weights and bias for this layer, and also compute the del(local error) to be passed to the previous layer
        # del - layer_size * 1
        # input_vector - input_size * 1
        # grad_W - layer_size * input_size
        # grad_b - layer_size * 1

        # get del for this layer
        # it outputs del column vector for this layer
        local_error = error_w_next * self.activation_derivative(self.weighted_sum)
        # get grad_W and grad_b for this layer
        self.grad_W = np.dot(local_error,self.input_vector.T) # del*X^T
        self.grad_b = np.sum(local_error, axis=1, keepdims=True) # as del = dL/dz, and dz/db=1, dL/dz*dz/db = dL/db = del
        raw_error_w_prev = np.dot((self.W).T,local_error) 
        
        # del_prev_layer = W^T . del_this_layer * activation_derivative(z_prev_layer) -- 
        # this will be calclated in the previous layer's backward() method, 
        # as we need the z of the previous layer to calculate the activation derivative for the previous layer, so we just pass the raw error to the previous layer and let it calculate its own del and pass it to its previous layer

        # if successful, we have gradients of weights and bias for this layer stored as attributes of this layer object, and we also have the del(local error) to be passed to the previous layer
        return raw_error_w_prev
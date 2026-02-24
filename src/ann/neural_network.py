"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from ann.neural_layer import Layer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        self.num_of_hidden_layers = cli_args.num_layers
        self.hidden_layers_sizes = cli_args.hidden_size
        self.weight_init_method = cli_args.weight_init
        self.hidden_activation_function = cli_args.activation

        self.input_size = 784  # for MNIST dataset
        self.output_size = 10  # for MNIST dataset

        for i in range(self.num_of_hidden_layers):
            if i==0:
                current_input_size = self.input_size
            else:
                current_input_size = self.hidden_layers_sizes[i-1] # for hidden layer i (starting from 0 index), the input size is the layer size of the i-1 th hidden layer
            hidden_layer_i = Layer(current_input_size, self.hidden_layers_sizes[i], weight_init=self.weight_init_method,layer_type="hidden")
            self.layers.append(hidden_layer_i)

            # so we are done with a list of hidden layers

        # now we add the output layer, which takes input from the last hidden layer and outputs the final output
        output_layer = Layer(input_size=self.hidden_layers_sizes[-1], layer_size=self.output_size, weight_init=self.weight_init_method,layer_type="output")
        self.layers.append(output_layer)

        pass
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        current_input_signal = X
        for layer in self.layers:
            current_output = layer.forward(current_input_signal)
            current_input_signal = current_output
        return current_output
        # this is the final output of the output layer (if everythying works fine, this is predicted proabalities for each class for each sample in the batch, so it should be of size (output_size, batch_size)) = y_hat
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        error_signal = y_pred - y_true # to start with the last error signal
        # we will directly use the short cut formula for the output layer, which is del = s - y, where s is the output of softmax and y is the true label in one hot encoding, so we dont need to compute the actual derivative of softmax activation function, which is computationally expensive, and we can directly use this error signal for backpropagation through the output layer and then through the hidden layers as usual using their activation function derivatives.
        # so basically the backward function needs del.w from next layer and then it calculates its del. this del is used for calcuting gradients. but for output layer, the del i.e. dL/dz is already known as error = y_pred - y-true
        # so ASSUMING WE HAVE ACTIVATION DERIVATIVE FOR OUTPUT LAYER SET AS 1, we can directly input this error signal right from the output layer, and everything will work fine.

        for layer in self.layers.reverse():
            intermed_error = layer.backward(error_signal)  # this will compute and store the graadients and return del.w value which is the input for the next layer in the reverse order
            error_signal = intermed_error
        pass
    
        # with this we have gradients stored in each layer object where each value in the matrix rep the sum of gradients of Loss wrt to that weight across the batch
        # so we might have to divide by batch size to get the mean gradient for each weight, and then we can use these gradients to update the weights using the optimizer, which will be implemented in the update_weights() method of this class.
        
    def update_weights(self):
        """
        Update weights using the optimizer.
        """

        
        pass
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        pass
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        pass

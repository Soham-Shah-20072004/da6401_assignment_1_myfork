"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from wandb.util import np

from ann import optimizers
from ann.neural_layer import Layer
import objective_functions


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
        self.loss_type = cli_args.loss
        # 1. Initialize the Optimizer Switchboard
        if cli_args.optimizer == 'sgd':
            self.optimizer = optimizers.SGD(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = optimizers.Momentum(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nag':
            self.optimizer = optimizers.NAG(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")

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
        self.optimizer.update(self.layers)
        
        pass
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        epoch_loss = []
        for epoch in range(epochs):
            number_of_batches = X_train.shape[1] // batch_size
            # but if the number of samples is not perfectly divisible by batch size, then we will have some remaining samples in the last batch, so we need to handle that case as well, we can simply ignore those remaining samples for simplicity, or we can create a smaller batch for those remaining samples, but for now we will ignore those remaining samples for simplicity.
            batch_loss = []
            batch_accuracy = []
            total_epoch_loss = 0
            for batch_number in range(0,number_of_batches):
                # make the batch X from X_train and y_train
                batch_X = X_train[:,batch_number*batch_size:(batch_number+1)*batch_size] # last one is not included in slicing
                batch_y = y_train[:,batch_number*batch_size:(batch_number+1)*batch_size] # last one is not included in slicing
                # forward pass
                batch_output= self.forward(batch_X)
                # backward pass
                self.backward(batch_y, batch_output) # this does not return anything, but it computes and stores the gradients in each layer object
                # update weights
                self.update_weights() # this will use the gradients stored in each layer object to update the weights and bias of each layer using the specified optimizer
            # track the loss per epoch
                if self.loss_type == 'cross_entropy':
                    # objective_function.categorical_cross_entropy(batch_y, batch_output) # this will give us a vector of loss values for each sample in the batch, we can take the mean of this vector to get the average loss for the batch,
                    batch_loss = np.mean(objective_functions.categorical_cross_entropy(batch_y, batch_output)) # this will give us a vector of loss values for each sample in the batch,
                
                print(f"Epoch {epoch+1}, Batch {batch_number+1}, Loss: {batch_loss}")
                batch_loss.append(batch_loss)
            
                predicted_class_index = np.argmax(batch_output, axis=0)
                true_class_index = np.argmax(batch_y, axis=0)
                correct_predictions = np.sum(predicted_class_index == true_class_index) # this sums over number of times value in first list is equal to corresponding value in second list, giving us the number of correct predictions in the batch
                batch_accuracy.append(correct_predictions / batch_y.shape[1]) # this gives us the accuracy for the batch, which is the number of correct predictions divided by the total number of samples in the batch
                # np.argmax gives the index of the maximum value on the specified axis, so np.argmax(batch_output, axis=0) gives us the predicted class for each sample in the batch, and np.argmax(batch_y, axis=0) gives us the true class for each sample in the batch, so we can compare these two to get the number of correct predictions in the batch, and then we can calculate the accuracy for the batch as correct_predictions / batch_size, and we can track this accuracy for each batch and also for each epoch to see how our model is performing during training.
                total_epoch_loss += batch_loss*batch_size
                
            epoch_loss_i = total_epoch_loss/X_train.shape[1] # average loss for the epoch
            print(f"Epoch {epoch+1}, Loss: {epoch_loss_i}, Accuracy: {np.mean(batch_accuracy)}")
            epoch_loss.append(epoch_loss_i)
        pass
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        self.forward(X)
        # Compute accuracy
        predicted_class_index = np.argmax(self.output, axis=0)
        true_class_index = np.argmax(y, axis=0)
        correct_predictions = np.sum(predicted_class_index == true_class_index)
        accuracy = correct_predictions / y.shape[1]
        print(f"Evaluation Accuracy: {accuracy}")

        # compute loss
        if self.loss_type == 'cross_entropy':
            loss = np.mean(objective_functions.categorical_cross_entropy(y, self.output))
            print(f"Evaluation Loss: {loss}")
        
        return accuracy, loss

"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
# downloader
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset_name):
    #mnist_784 means flattened 
    if dataset_name == 'mnist':
        mnist= fetch_openml('mnist_784', version=1,parser= "liac-arff", as_frame=False) # this will download the MNIST dataset and return the data and labels as numpy arrays, where data is of shape (70000, 784) and labels is of shape (70000,)
    # dont want frames but want numpy arrays
        X,y = mnist.data, mnist.target
        X = X.astype(np.float64)
        y = y.astype(np.uint8)
    
    elif dataset_name == 'fashion_mnist':
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1,parser= "liac-arff", as_frame=False)
        X,y = fashion_mnist.data, fashion_mnist.target
        X = X.astype(np.float64)
        y = y.astype(np.uint8)

    # Split the data into training and testing sets
    # Setting a random_state ensures reproducibility of the split
    # 'stratify=y' ensures that the class distribution ratio is maintained at the overall datset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test



def pre_processing_data(X,y):
    # normalizer
    X = X / 255.0 # X could be matrix of training or for testing or for a single sample for inference.
    # but it is expected here that X has proper shape and correct datatypes which are deffined in load_data fucntion
    # so it is expected that X(input of this method) comes from load_data
    # one_hot encoding
    # superb technique to one hot encode labels, create identity matrix which serves as a loopup table for each integer accessed by the index of its row.
    ten_ten_identity = np.eye(10)
    # y_train serves as a list of integers which are the indices of tthe rows that needs to be grabbed from this ten cross ten matrix
    y_train_one_hot = ten_ten_identity[y]
    return X, y_train_one_hot


# batch generator
def batch_generator(X,y,batch_size):
    # shuffling the dataset
    no_of_samples = X.shape[0]
    indices = np.arange(no_of_samples) # this gives a list of integers from 0 to number=X.shape[0] = nu of samples
    np.random.shuffle(indices) # inplace shuffled indices list

    for start in range(0,no_of_samples,batch_size):
        # to handle not perfectly divisible datset size by batch size, to handle the last batch
        end_index = min(start+batch_size,no_of_samples)
        # slices the indices of indices
        samples_indices =indices[start:end_index]

        X_batch,y_batch = X[samples_indices], y[samples_indices]
        
        # yield vs return understood
        yield X_batch.T, y_batch.T


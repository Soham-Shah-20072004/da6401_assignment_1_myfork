"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
# downloader
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

import os

def load_data(dataset_name):
    if os.path.exists(f"src/{dataset_name}_X_train.npy"):
        print(f"Loading {dataset_name} from local memmapped cache...")
        X_train = np.load(f"src/{dataset_name}_X_train.npy", mmap_mode='r')
        y_train = np.load(f"src/{dataset_name}_y_train.npy", mmap_mode='r')
        X_test = np.load(f"src/{dataset_name}_X_test.npy", mmap_mode='r')
        y_test = np.load(f"src/{dataset_name}_y_test.npy", mmap_mode='r')
        return X_train, y_train, X_test, y_test
        
    print(f"Downloading {dataset_name} from internet (this will only happen once)...")
    
    # 2. Otherwise download and parse with OpenML
    if dataset_name == 'mnist':
        mnist= fetch_openml('mnist_784', version=1,parser= "liac-arff", as_frame=False) 
        X,y = mnist.data, mnist.target
        X = X.astype(np.float32)
        y = y.astype(np.uint8)
    
    elif dataset_name == 'fashion_mnist':
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1,parser= "liac-arff", as_frame=False)
        X,y = fashion_mnist.data, fashion_mnist.target
        X = X.astype(np.float32)
        y = y.astype(np.uint8)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # 3. Save to local disk cache for future runs (saves hundreds of MBs of RAM and huge amounts of time)
    np.save(f"src/{dataset_name}_X_train.npy", X_train)
    np.save(f"src/{dataset_name}_y_train.npy", y_train)
    np.save(f"src/{dataset_name}_X_test.npy", X_test)
    np.save(f"src/{dataset_name}_y_test.npy", y_test)

    return X_train, y_train, X_test, y_test



def pre_processing_data(X,y):
    # Instead of allocating an empty float32 array, process entirely in minimal memory chunk by chunk and append
    chunk_size = 1000
    X_normalized_chunks = []
    
    for i in range(0, X.shape[0], chunk_size):
        end_idx = min(i + chunk_size, X.shape[0])
        # Direct chunk operation and float32 conversion
        chunk = X[i:end_idx].astype(np.float32) / 255.0
        X_normalized_chunks.append(chunk)

    # Combine back together
    X_normalized = np.vstack(X_normalized_chunks)
    
    # one_hot encoding
    # superb technique to one hot encode labels, create identity matrix which serves as a lookup table
    ten_ten_identity = np.eye(10)
    # y_train serves as a list of integers which are the indices of tthe rows that needs to be grabbed from this ten cross ten matrix
    y_one_hot = ten_ten_identity[y]
    return X_normalized.T, y_one_hot.T


# batch generator
def batch_generator(X,y,batch_size):
    # shuffling the dataset
    no_of_samples = X.shape[1]
    indices = np.arange(no_of_samples) # this gives a list of integers from 0 to number=X.shape[0] = nu of samples
    np.random.shuffle(indices) # inplace shuffled indices list

    for start in range(0,no_of_samples,batch_size):
        # to handle not perfectly divisible datset size by batch size, to handle the last batch
        end_index = min(start+batch_size,no_of_samples)
        # slices the indices of indices
        samples_indices =indices[start:end_index]

        X_batch,y_batch = X[:,samples_indices], y[:,samples_indices]
        
        # yield vs return understood
        yield X_batch, y_batch


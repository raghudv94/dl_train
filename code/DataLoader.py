import os
import pickle
import numpy as np
import torch
from torchvision import transforms as tf


"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    files = os.listdir(data_dir)

    x_train = np.zeros(shape=[0,32,32,3], dtype=float)
    y_train = np.asarray([], dtype=int)

    x_test = np.zeros(shape=[0,32,32,3], dtype=float)
    y_test = np.asarray([], dtype=int)

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_xy_batch(cur_file_path):
        current_set = unpickle(cur_file_path)
        x_temp = np.array(current_set[b'data'], dtype=float) / 255.0
        x_temp = x_temp.reshape([-1, 3, 32, 32])
        x_temp = x_temp.transpose([0, 2, 3, 1])
        y_temp = np.array(current_set[b'labels'])

        return x_temp, y_temp

    for file in files:
        cur_file_path = os.path.join(data_dir, file)
        if 'data_batch' in file:
            x_temp, y_temp = get_xy_batch(cur_file_path)
            x_train = np.concatenate((x_train, x_temp), axis=0)
            y_train = np.concatenate((y_train, y_temp), axis=0)

        elif 'test_batch' in file:
            x_temp, y_temp = get_xy_batch(cur_file_path)
            x_test = np.concatenate((x_test, x_temp))
            y_test = np.concatenate((y_test, y_temp))

    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE

    x_test = np.load(data_dir)
    print(x_test.shape)
    x_test = torch.tensor(x_test.transpose(0, 3, 1, 2) / 255.0).float()
    # data = data.permute(0,3,1,2)
    print(x_test.shape)
    tf_norm = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    x_test = tf_norm(x_test)
    return x_test

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.
        split_index: integer value

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    #print(x_train_new.shape, y_train_new.shape, x_valid.shape, y_valid.shape)


    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid


### YOUR CODE HERE
# import tensorflow as tf

"""
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset 
		x_test = load_testing_images(args.data_dir)
		predictions = model.predict_prob(x_test)
		np.save(args.result_dir, predictions)
"""

from ImageUtils import parse_record
from DataLoader import load_data, train_valid_split
from Model import Cifar

import os
import argparse

import torch

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=18,
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=512, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10,
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v2', help='model directory')
    parser.add_argument("--modelname", type=str, default='densenet_withpreimg_v2', help='model name')

    parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data","train-cifar10")
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)

    x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)
    #print("Resnet batch size: 512, lr = 0.256, epochs=500, momentum=0875, weight_decay=0.00125")	
    #if torch.cuda.is_available():
    #lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    #batch_size_list = [256, 128, 64, 32]
    lr_list = [0.01]
    batch_size_list = [64]    

    for lr in lr_list:
        for batch_size in batch_size_list:
            print("Parameteres: ")
            print(lr, batch_size, config.modelname)
            model = Cifar(config, lr, batch_size).cuda()
    #else:
    #    model = Cifar(config)
    #print(model)

    ### YOUR CODE HERE
    # First step: use the train_new set and the valid set to choose hyperparameters.
            model.train(x_train_new, y_train_new, 500, x_valid, y_valid)
    #model.test_or_validate(x_valid, y_valid, [10, 20, 40, 70, 100, 140, 160, 170, 180, 190, 200, 250, 300, 350, 400, 500])

    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    #model.train(x_train, y_train, 10)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    #model.test_or_validate(x_test, y_test, [10])
    ### END CODE HERE


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)

### END CODE HERE


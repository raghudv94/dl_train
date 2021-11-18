### YOUR CODE HERE
# import tensorflow as tf
# import torch
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import sys

"""This script defines the network.
"""
"""
class MyNetwork(object):

    def __init__(self, configs):
        self.configs = configs

    def __call__(self, inputs, training):
    	'''
    	Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
    	'''
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        return inputs
'''
"""

class ResNet(nn.Module):
    def __init__(self,
                 resnet_version,
                 resnet_size,
                 num_classes,
                 first_num_filters,
                 ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.

        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.

        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.

        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        self.start_layer = nn.Conv2d(3, first_num_filters, (7, 7), padding=(3,3))

        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters,
                eps=1e-5,
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        previous_layer_filters = self.first_num_filters

        for i in range(3):
            filters = self.first_num_filters * (2 ** i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, previous_layer_filters))
            if self.resnet_version > 1:
                previous_layer_filters = filters * 4
            else:
                previous_layer_filters = filters

        if self.resnet_version > 1:
            self.output_layer = output_layer(filters * 4, self.resnet_version, self.num_classes)
        else:
            self.output_layer = output_layer(filters, self.resnet_version, self.num_classes)

    def forward(self, inputs):
        #print("Input Shape")
        #print(inputs.shape)
        outputs = self.start_layer(inputs)
        # print("First Layer Output shape")
        # print(outputs.shape)
        if self.resnet_version == 1:
            # print("BN Relu")
            outputs = self.batch_norm_relu_start(outputs)
            # print(outputs.shape)
        for i in range(3):
            # print("Running Stack " + str(i))
            outputs = self.stack_layers[i](outputs)
            # print(outputs.shape)
        # print("Running Output layer")
        outputs = self.output_layer(outputs)
        # print(outputs.shape)
        return outputs


#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.bn1 = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.relu = nn.ReLU()

        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        outputs = self.bn1(inputs)
        outputs = self.relu(outputs)

        return outputs
        ### YOUR CODE HERE


class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.projection_shortcut = projection_shortcut
        '''
        if self.projection_shortcut is not None:
            self.conv1 = nn.Conv2d(int(filters / 2), filters, (3, 3), stride=2, padding=1)
            self.projection = nn.Conv2d(int(filters / 2), filters, (1, 1), stride=2, padding=0)
        else:
            self.conv1 = nn.Conv2d(int(filters), filters, (3, 3), stride=1, padding=1)
        '''
        self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size=(3,3), stride=strides, padding=(1,1), padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.997)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3,3), stride=1, padding=(1,1))
        self.bnrl = batch_norm_relu_layer(filters)
        self.relu = nn.ReLU()


        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE

        input_2 = inputs
        if self.projection_shortcut is not None:
            input_2 = self.projection_shortcut(input_2)
            input_2 = self.bn1(input_2)

        outputs = self.conv1(inputs)
        outputs = self.bnrl(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn1(outputs)

        outputs = outputs + input_2

        outputs = self.relu(outputs)

        return outputs
        ### YOUR CODE HERE


class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.projection_shortcut = projection_shortcut

        self.bnrl1 = batch_norm_relu_layer(first_num_filters)
        '''
        if self.projection_shortcut is not None:
            self.conv1 = nn.Conv2d(int(filters / 2), int(filters / 4), (1, 1), stride=2, padding=0)
            self.projection = nn.Conv2d(int(filters / 2), filters, (1, 1), stride=2, padding=0)
            self.bnrl1 = batch_norm_relu_layer(int(filters / 2))
        else:
            self.conv1 = nn.Conv2d(filters, int(filters / 4), (1, 1), stride=1, padding=0)
            self.bnrl1 = batch_norm_relu_layer(int(filters))
        '''
        self.conv1 = nn.Conv2d(first_num_filters, filters // 4, kernel_size=(1,1), stride=strides, bias=False)

        self.conv2 = nn.Conv2d(filters // 4, filters // 4, (3, 3), stride=1, padding=(1,1), padding_mode='zeros', bias=False)
        self.bnrl2 = batch_norm_relu_layer(filters // 4)

        self.conv3 = nn.Conv2d(filters // 4, filters, kernel_size= (1, 1), stride=1, bias=False)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.

        input_2 = inputs
        if self.projection_shortcut is not None:
            input_2 = self.projection_shortcut(input_2)

        outputs = self.bnrl1(inputs)
        outputs = self.conv1(outputs)
        outputs = self.bnrl2(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bnrl2(outputs)
        outputs = self.conv3(outputs)

        outputs = outputs + input_2

        return outputs
        ### YOUR CODE HERE

class projection(nn.Module):
    def __init__(self, filters, strides, first_num_filters):
        super(projection, self).__init__()
        self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size= (1,1), stride=strides)

    def forward(self, inputs):
        return self.conv1(inputs)

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        self.resnet_size = resnet_size

        projection_shortcut = projection(filters_out, strides, first_num_filters)
        self.block_stack = nn.ModuleList()
        for i in range(self.resnet_size):
            if i ==0:
                self.block_stack.append(block_fn(filters_out, projection_shortcut, strides, first_num_filters))
            else:
                self.block_stack.append(block_fn(filters_out, None, 1, filters_out))
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides

        ### END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        for i in range(self.resnet_size):
            inputs = self.block_stack[i](inputs)

        return inputs
        ### END CODE HERE


class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """

    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        self.resnet_version = resnet_version
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)

        ### END CODE HERE
        self.maxpool = nn.MaxPool2d((8,8))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.resnet_version == 2:
            self.fc = nn.Linear(256, 1000)
        else:
            self.fc = nn.Linear(64, 1000)
        self.fc_final = nn.Linear(1000, 10)
        self.fc1 = nn.Linear(filters, num_classes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(filters, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 10)
        ### END CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        #outputs = F.adaptive_avg_pool2d(inputs, output_size=1)

        if self.resnet_version == 2:
            inputs = self.bn_relu(inputs)
        
        inputs = self.avgpool(inputs)
        #inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
        outputs = torch.flatten(inputs, 1)
        outputs = self.relu(self.fc(outputs))
        #outputs = self.relu(self.fc3(outputs))
        outputs = self.fc_final(outputs)
        # print(outputs.shape)
        return outputs
        ### END CODE HERE
### END CODE HERE

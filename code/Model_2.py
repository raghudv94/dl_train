### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, time
import numpy as np
#from Network import MyNetwork
from ImageUtils import parse_record

import torch.nn as nn
import numpy as np
from tqdm import tqdm

from Network import ResNet
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""
''''
class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)

    def model_setup(self):
        pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        pass

    def evaluate(self, x, y):
        pass

    def predict_prob(self, x):
        pass
'''
from densenet import DenseNet
def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    return model

def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)

from typing import Type, Any, Callable, Union, List, Optional

from resnet import ResNet, BasicBlock, Bottleneck

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    return model

def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

from pyramidnet import PyramidNet

def pyramidnet164(bottleneck=True, **kwargs):
    """PyramidNet164 for CIFAR and SVHN"""
    return PyramidNet(164, 270, 10, bottleneck=True)


def pyramidnet272(bottleneck=True, **kwargs):
    """PyramidNet272 for CIFAR and SVHN"""
    return PyramidNet(272, 200, 10, bottleneck=True)


class Cifar(nn.Module):
    def __init__(self, config, lr, batch_size):
        super(Cifar, self).__init__()
        self.config = config
        '''
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        '''
        self.batch_size = batch_size
        self.network = densenet201(pretrained= False, progress= True, memory_efficient = False)
        #self.network = wide_resnet101_2()
        #self.network = pyramidnet272(bottleneck=True)
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.875, weight_decay=0.00125)
        self.lr_schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200,300, 390], gamma=0.1)
        ### YOUR CODE HERE

    def train(self, x_train, y_train, max_epoch, x_valid, y_valid):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.batch_size

        print('### Training... ###')
        #print(x_train.shape)
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
            ### YOUR CODE HERE
          
            training_correct_labels = 0

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                starting_index = i * self.batch_size
                ending_index = min(curr_x_train.shape[0], i * self.batch_size + self.batch_size)
                curr_x_train_batch = [parse_record(curr_x_train[j], True) for j in range(starting_index, ending_index)]
                #curr_x_train_batch = [curr_x_train[j] for j in range(starting_index, ending_index)]
                curr_y_train_batch = curr_y_train[starting_index: ending_index]

                #if torch.cuda.is_available():
                tensor_cur_x_batch = torch.from_numpy(np.array(curr_x_train_batch)).float().cuda()
                tensor_cur_y_batch = torch.from_numpy(np.array(curr_y_train_batch)).long().cuda()
                #else:
                #    tensor_cur_x_batch = torch.from_numpy(np.array(curr_x_train_batch)).float()
                #    tensor_cur_y_batch = torch.from_numpy(np.array(curr_y_train_batch)).long()

                cur_batch_output = self.network(tensor_cur_x_batch)
                # print(type(cur_batch_output))
                # print(type(tensor_cur_y_batch))

                training_correct_labels += float((cur_batch_output.max(dim=1)[1] == tensor_cur_y_batch).sum())

                loss = self.criterion(cur_batch_output, tensor_cur_y_batch)

                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss))
            
            training_accuracy = training_correct_labels/num_samples
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Training Accuracy {:.6f} Duration {:.3f} seconds.'.format(epoch, loss,training_accuracy , duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)
                self.test_or_validate(x_valid, y_valid, [epoch])
            self.lr_schedular.step()        

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            #checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (checkpoint_num))
            checkpointfile = os.path.join(os.path.dirname(os.getcwd()), 'saved_models', self.config.modelname, 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE

                test_x = parse_record(x[i], False)
                #test_x = x[i]
                #if torch.cuda.is_available():
                x_value = torch.from_numpy(np.array(test_x)).float().cuda()[None,...]
                #else:
                #    x_value = torch.from_numpy(np.array(test_x)).float()[None, ...]

                output = self.network(x_value)
                calculated_output = torch.argmax(output, dim=1)
                preds.append(calculated_output)

                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

    def save(self, epoch):
        checkpoint_path = os.path.join(os.path.dirname(os.getcwd()), 'saved_models', self.config.modelname)
        checkpoint_file = os.path.join(checkpoint_path, 'model-%d.ckpt' % (epoch))
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_file)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE

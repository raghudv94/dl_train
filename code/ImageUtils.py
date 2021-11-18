import numpy as np
#from torchvision import transforms as tf
import random

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    '''
    depth_major = record.reshape((3, 32, 32))

    image = np.transpose(depth_major, [1, 2, 0])

    ### END CODE HERE

    #image = preprocess_image(image, training) # If any.
    image = np.transpose(image, [2, 0, 1])
    '''
    if training:
        transform = tf.Compose([  # tf.ToPILImage(),
            tf.RandomHorizontalFlip(),
            tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # internet says these are good values for cifar 10
            tf.RandomCrop((32, 32), 4, padding_mode='edge'),
            # tf.RandomRotation(15),
            tf.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.05, 20), value=0, inplace=False),
        ])
    else:
        transform = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    image = transform(record)


    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE

    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.

        #print("Before padding: " + str(image.shape))
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=255)
        #print("After padding: " + str(image.shape))
        ### END CODE HERE
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # image = tf.random_crop(image, [32, 32, 3])
        # HINT: randomly generate the upper left point of the image
        x_offset = random.randrange(1, 9)
        y_offset = random.randrange(1, 9)
        image = image[x_offset:x_offset + 32, y_offset:y_offset + 32, :]
        #print("After crop: " + str(image.shape))
        ### END CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        # image = tf.image.random_flip_left_right(image)
        if random.random() < 0.5:
            image = np.flip(image, 1)
        #print("After  flip: " + str(image.shape))
        ### END CODE HERE


    image = (image - np.mean(image)) / np.std(image);
    ### END CODE HERE

    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE

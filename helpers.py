import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model


def load_and_process_image(path):
    """
    Loads an image from path and processes it as to use later in the VGG19
    model
    """
    img = load_img(path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img


# def deprocess(img):
#     """Deprocess a generated image for displaying it"""
#     # Reverse of tensorflow.python.keras.applications.vgg19.preprocess_input
#     # applied to every channel
#     img[:, :, 0] += 103.939
#     img[:, :, 1] += 116.779
#     img[:, :, 2] += 123.68
#     # Invert the order of the channels
#     img = img[:, :, ::-1]
#     img = np.clip(img, 0, 255).astype('uint8')

#     return img


# def display_image(img, name=None):
#     """Displays an array-like image"""
#     # Remove (if presented) an extra dimension which corresponds to the number
#     # of training examples (as we always have only one)
#     if len(img.shape) == 4:
#         img = np.squeeze(img, axis=0)

#     img = deprocess(img)

#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(name)
#     plt.imshow(img)

def deprocess(img):
    """Deprocess a generated image for displaying it"""
    # Reverse of tensorflow.python.keras.applications.vgg19.preprocess_input
    # applied to every channel
    img = img.numpy()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Invert the order of the channels
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')

    return img


def display_image(img, name=None):
    """Displays an array-like image"""
    # Remove (if presented) an extra dimension which corresponds to the number
    # of training examples (as we always have only one)
    if len(img.shape) == 4:
        img = tf.squeeze(img, axis=0)

    img = deprocess(img)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    plt.imshow(img)

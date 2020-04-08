"""
@author: Cole Hill
University of South Florida
Spring 2020
Computer Vision
"""
import numpy as np
import tensorflow as tf

from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model

def initialize_model(h,w,c,needs_scaling=True,dropout=0.1,summary=False):
    """
    Initializes and return CNN implementation of UNET

    @params
        h - height of image in pixels
        w - width of image in pixels
        c - number of channels in image (e.g. RGB image has 3 channels)
        needs_scaling = divides color values by 255 if set to True (default)
        dropout - metaparameter to avoid overfitting
        summary - Prints model summary if set to True

    References
        [1] - https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
    """
    img_shape = (h,w,c)
    input_layer = Input(img_shape)
    if needs_scaling:
        scaled_input = Lambda(lambda x: x / 255)(input_layer)
    else:
        scaled_input = input_layer

    down_convs = []
    up_convs = []

    pool_layer = scaled_input
    conv_window = 16
    for i in range(5):
        c = Conv2D(conv_window,(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_layer)
        c = Dropout(dropout)(c)
        c = Conv2D(conv_window,(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        pool_layer = MaxPooling2D((2,2))(c)
        down_convs.append(c)
        conv_window*=2
    
    conv_window/=2
    prev = down_convs[-1]
    for i in range(4):
        conv_window/=2
        conv_window = int(conv_window)
        u = Conv2DTranspose(conv_window, (2,2), strides=(2,2), padding='same')(prev)
        u = concatenate([u,down_convs[-(i+2)]])
        c = Conv2D(conv_window, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
        c = Dropout(dropout)(c)
        c = Conv2D(conv_window, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c)
        prev = c

    output_layer = Conv2D(1, (1, 1), activation='sigmoid') (prev)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    if summary:
        model.summary()

    return model


if __name__ == "__main__":

    modlel = initialize_model(256,256,3)


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

    #Set up input layer dimensions
    img_shape = (h,w,c)
    input_layer = Input(img_shape)
    #Adds divide by 255 if user specifies
    if needs_scaling:
        scaled_input = Lambda(lambda x: x / 255)(input_layer)
    else:
        scaled_input = input_layer

    #Array to keep track of conv layers for up-sampling section
    down_convs = []

    #Initialize conv window size and input
    pool_layer = scaled_input
    conv_window = 16
    #Iteratively generate down-sampling layers
    for i in range(5):
        #First conv stage with previous output (input on first iteration)
        c = Conv2D(conv_window,(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_layer)
        #Dropout for overfitting proteciton
        c = Dropout(dropout)(c)
        #Second conv stage
        c = Conv2D(conv_window,(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        #Pool
        pool_layer = MaxPooling2D((2,2))(c)
        #Store conv layer reference for up-sampling
        down_convs.append(c)
        #Double convolution window size
        conv_window*=2
    
    #Initialize the conv window size and input
    conv_window/=2
    prev = down_convs[-1]
    #Iteratively generate up-sampling layers
    for i in range(4):
        #Half the conv window size
        conv_window/=2
        conv_window = int(conv_window)
        #Up sampling layer
        u = Conv2DTranspose(conv_window, (2,2), strides=(2,2), padding='same')(prev)
        #Combine prev conv layer and up-sample layer
        u = concatenate([u,down_convs[-(i+2)]])
        #First conv stage
        c = Conv2D(conv_window, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
        #Dropout for overfitting proteciton
        c = Dropout(dropout)(c)
        #Second conv stage
        c = Conv2D(conv_window, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c)
        #Keep track of previous output
        prev = c

    output_layer = Conv2D(1, (1, 1), activation='sigmoid') (prev)
    
    #Compile model
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    #Print summary if requested
    if summary:
        model.summary()

    return model


if __name__ == "__main__":

    modlel = initialize_model(256,256,3)


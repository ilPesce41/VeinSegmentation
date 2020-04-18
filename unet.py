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
from scipy.ndimage import gaussian_filter
from cv2 import imread,imwrite,resize, connectedComponentsWithStats
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

seed = 1
random_state = np.random.RandomState(seed) #Alwasy have the same test indices

#Configurations
FIRST_LOAD=False
LOAD_OLD = True
OLD_MODEL = "model.h5"
SAVE_NAME = "model.h5"
TRAIN = False
X_dir = "toh_output"
X_fil = "out_t"
Y_dir = "toh_output"
Y_fil = "out_v"

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
    input_layer = Input(shape=img_shape)
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

def load_images(dir,x,y,to_gray=False,fil=''):
    images = os.listdir(dir)
    def file_select(x):
        if (x.endswith('.png') or x.endswith('.jpg')) and '_' in x and (fil in x):
            return True
        return False
    images = list(filter(file_select,images))
    im_out = []
    for im in tqdm(images):
        tmp = imread(os.path.join(dir,im))
        if to_gray:
            tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
            tmp = tmp/255
        try:
            im_out.append(resize(tmp,(x,y)))
        except:
            pass
    return np.array(im_out)

if __name__ == "__main__":

    # if FIRST_LOAD:
    #     print("Loading Input Images")
    #     X = load_images(X_dir,256,256,fil=X_fil)
    #     print("Loading Output Images")
    #     Y = load_images(Y_dir,256,256,fil=Y_fil,to_gray=True)
    #     Y = np.expand_dims(Y,axis=-1)
    #     np.save("X",X)
    #     np.save("Y",Y)
    # else:
    #     X = np.load("X.npy")
    #     Y = np.load("Y.npy")

    # print("Loading Input Images")
    # X1 = load_images('selected_images',256,256,fil='')
    # print("Loading Output Images")
    # Y1 = load_images('vessel_images',256,256,fil='',to_gray=True)
    # Y1 = np.expand_dims(Y1,axis=-1)
    # X = np.concatenate([X,X1])
    # Y = np.concatenate([Y,Y1])

    print("Images loaded")
    # print("Input Shape: {}".format(str(X.shape)))
    # print("Output Shape: {}".format(str(Y.shape)))

    # indices = random_state.permutation(X.shape[0])
    # train = indices[:int(len(indices)*.90)]
    # test = indices[int(len(indices)*.90):]

    # x_train,y_train = X[train],Y[train]
    # x_test,y_test = X[test],Y[test]
    
    strategy = tf.distribute.MirroredStrategy()
    if LOAD_OLD:
        model = tf.keras.models.load_model(OLD_MODEL)
    if TRAIN:
        with strategy.scope():
            if not LOAD_OLD:
                model = initialize_model(256,256,3,summary=True)

            callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
            )]
            model.fit(x_train,y_train,epochs=50,validation_split=0.1,batch_size=8,callbacks=callbacks)

    model.save(SAVE_NAME)

    # preds = model.predict(x_test)
    # for i in range(preds.shape[0]):
    #     tmp = x_test[i].copy()
    #     for x in range(tmp.shape[0]):
    #         for y in range(tmp.shape[1]):
    #             # print(preds[i,x,y])
    #             if preds[i,x,y]>.3:
    #                 # print(i)
    #                 tmp[x,y] = [0,255,0]
    #     imwrite("pred{}.jpg".format(i),tmp)


    # model = tf.keras.models.load_model("model.h5")

    print("Preds")

    im = imread(r"C:\Users\chill\Desktop\normed.jpg")
    # im = imread(r"C:\Users\chill\Pictures\Camera Roll\WIN_20200416_14_53_50_Pro (2).jpg")
    im = imread(r"C:\Users\chill\Pictures\Camera Roll\WIN_20200416_16_40_00_Pro (2).jpg")
    # for i in range(im.shape[0]):
    #     for j in range(im.shape[1]):
    #         if im[i,j,0]>200:
    #             im[i,j] = [0,0,0]
    im = cv2.resize(im,(256,256))
    im2 = gaussian_filter(im,5,mode='wrap') 
    img = np.array([im2])
    # img = cv2.normalize(img, img, 0, 255,norm_type=cv2.NORM_MINMAX)
    # imwrite("clipped.png",img)

    pred = model.predict(np.array([im2]))
    
    pred = pred[0]
    cpy = im.copy()
    imwrite("temp.png",(255*pred).astype(int))
    pred = imread("temp.png")
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    pred = cv2.threshold(pred,120,255,cv2.THRESH_BINARY)[1]
    # os.remove("temp.png")
    comps, out, stats, _ = connectedComponentsWithStats(pred,connectivity=4)
    sizes = stats[1:,-1]
    comps-=1

    for i in range(comps):
        if sizes[i]<400:
            pred[out==i+1] = [0]

    for i in range(cpy.shape[0]):
        for j in range(cpy.shape[1]):
            if pred[i,j]>0:
                cpy[i,j,1] = 255
    # pred = pred*255
    # pred2 = pred2*255
    imwrite("pred{}.jpg".format(0),cpy)
    imwrite("mask.jpg",pred.astype(int))
    # imwrite("mask2.jpg",pred2.astype(int))
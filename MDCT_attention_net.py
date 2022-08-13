from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import losses
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy.random as random
import numpy as np
from numpy import r_
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras import layers
from MDCT_attention import *


def branch_model(batch_size, width, height, num_ch, num_classes, learning_rate, d, thresh):
    imgv = Input(batch_shape=(batch_size, width, height, num_ch))
    imgn = Input(batch_shape=(batch_size, width, height, num_ch))

    ## Block 1
    conv1_1v = Conv2D(16, (3, 3), activation="relu", padding = 'same')(imgv)
    conv1_2v = Conv2D(16, (3, 3), activation="relu", padding = 'same')(conv1_1v) 

    pool1v = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2v)

    conv1_1n = Conv2D(16, (3, 3), activation="relu", padding = 'same')(imgn)
    conv1_2n = Conv2D(16, (3, 3), activation="relu", padding = 'same')(conv1_1n) 

    pool1n = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2n)
    
    ## Block 2
    conv2_1v = Conv2D(32, (3, 3), activation="relu", padding = 'same')(pool1v)
    conv2_2v = Conv2D(32, (3, 3), activation="relu", padding = 'same')(conv2_1v)
    
    pool2v = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2v)
    
    conv2_1n = Conv2D(32, (3, 3), activation="relu", padding = 'same')(pool1n)
    conv2_2n = Conv2D(32, (3, 3), activation="relu", padding = 'same')(conv2_1n)
    
    pool2n = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2n)
    
    ## Block 3
    conv3_1v = Conv2D(64, (3, 3), activation="relu", padding = 'same')(pool2v)
    conv3_2v = Conv2D(64, (3, 3), activation="relu", padding = 'same')(conv3_1v)

    pool3v = MaxPooling2D((2, 2), strides=(2, 2))(conv3_2v)
    attended_pool3v = att_dct(pool3v, thresh)
    att_out_pool3v = BatchNormalization()(attended_pool3v)
    
    conv3_1n = Conv2D(64, (3, 3), activation="relu", padding = 'same')(pool2n)
    conv3_2n = Conv2D(64, (3, 3), activation="relu", padding = 'same')(conv3_1n)

    pool3n = MaxPooling2D((2, 2), strides=(2, 2))(conv3_2n)
    attended_pool3n = att_dct(pool3n, thresh)
    att_out_pool3n = BatchNormalization()(attended_pool3n)
    
    ## Block 4
    conv4_1v = Conv2D(128, (3, 3), activation="relu", padding = 'same')(att_out_pool3v)
    conv4_2v = Conv2D(128, (3, 3), activation="relu", padding = 'same')(conv4_1v)

    pool4v = MaxPooling2D((2, 2), strides=(2, 2))(conv4_2v)
    attended_pool4v = att_dct(pool4v, thresh)
    att_out_pool4v = BatchNormalization()(attended_pool4v)
    
    conv4_1n = Conv2D(128, (3, 3), activation="relu", padding = 'same')(att_out_pool3n)
    conv4_2n = Conv2D(128, (3, 3), activation="relu", padding = 'same')(conv4_1n)

    pool4n = MaxPooling2D((2, 2), strides=(2, 2))(conv4_2n)
    attended_pool4n = att_dct(pool4n, thresh)
    att_out_pool4n = BatchNormalization()(attended_pool4n)
    
    flatten_features_5v = Flatten()(att_out_pool4v)
    drop1_features_5v = Dropout(d)(flatten_features_5v)
    dense1_features_5v = Dense(512, activation='relu')(drop1_features_5v)
    drop2_features_5v = Dropout(d)(dense1_features_5v)
    dense2_features_5v = Dense(512, activation='relu')(drop2_features_5v)

    flatten_features_5n = Flatten()(att_out_pool4n)
    drop1_features_5n = Dropout(d)(flatten_features_5n)
    dense1_features_5n= Dense(512, activation='relu')(drop1_features_5n)
    drop2_features_5n = Dropout(d)(dense1_features_5n)
    dense2_features_5n = Dense(512, activation='relu')(drop2_features_5n)
    
    L2_layer = Lambda(lambda tensors: K.square(tensors[0] - tensors[1]))
    distance = L2_layer([dense2_features_5v, dense2_features_5n])
    prediction = Dense(2, activation='softmax')(distance)

    siamese_net = Model(inputs=[imgv, imgn], outputs=prediction)
    siamese_net.compile(loss="categorical_crossentropy", metrics = ['accuracy'], optimizer=Adam(learning_rate = 0.0001))
    siamese_net.summary()
    
    return siamese_net


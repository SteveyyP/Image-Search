"""
Model Utils

Created by: @SteveyyP

"""

# Import Libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from tqdm import tqdm_notebook
from scipy.spatial.distance import hamming, cosine

# Model Inputs
def model_inputs(image_size):
    """ 
    Defines the input variables to a model 
    
    Input
    size - tuple of size of image (w, h)
    
    Output
    inputs - tf.placeholder for input shape
    targets - tf.placeholder for labels
    dropout_rate - tf.placehodler for dropout_rate
    """

    
    # [Batch_size, height, width, channels]
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], 3], name='images')
    
    targets = tf.placeholder(dtype=tf.int32, shape=[None], name='targets')
    
    dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
    
    return inputs, targets, dropout_rate

# Convolutional Block
def conv_block(inputs, number_filt, kernel_size, strides = (1,1), padding = 'SAME', activation = tf.nn.relu, max_pool = True, batch_norm = True):
    """ 
    Defines a conv layer 
    
    Inputs
    input - input size
    number_filt - number of filters being used
    kernel_size - tuple, size of filter being used
    strides - tuple, size of stride
    padding - string, type of padding
    activation - which activation function is used
    max_pool - bool, whether or not to use max_pooling
    batch_norm - bool, whether or not to use batch_normalization
    
    Output
    layer - final output of convolutional block after convolutions (potentially max_pool and batch_norm)
    conv_features - convolutional layer
    """
    
    conv_features = layer = tf.layers.conv2d(inputs = inputs, 
                                             filters = number_filt, 
                                             kernel_size = kernel_size, 
                                             strides = strides, 
                                             padding = padding, 
                                             activation = activation)
    
    if max_pool:
        layer = tf.layers.max_pooling2d(layer, 
                                        pool_size = (2,2), 
                                        strides = (2,2), 
                                        padding = 'SAME')
        
    if batch_norm:
        layer = tf.layers.batch_normalization(layer)
        
    return layer, conv_features

# Dense Block
def dense_block(inputs, units, activation = tf.nn.relu, dropout_rate = None, batch_norm = True):
    """ 
    Defines dense block layer 
    
    Input
    inputs - input size
    units - number of neurons
    activation - which activation function to use
    dropout_rate - the dropout rate
    batch_norm - bool, whether or not dropout_rate is used
    
    Output
    layer - final output of convolutional block after convolutions (potentially max_pool and batch_norm)
    dense_features - dense layer
    """
    dense_features = layer = tf.layers.dense(inputs = inputs,
                                              units = units,
                                              activation = activation)
    if dropout_rate is not None:
        layer = tf.layers.dropout(layer, 
                                  rate = dropout_rate)
    if batch_norm:
        layer = tf.layers.batch_normalization(layer)
        
    return layer, dense_features

# Optimizer and Loss Function
def opt_loss(logits, targets, learning_rate):
    """ 
    Defines optimizer and loss function 
    
    Input
    logits - values from model
    targets - true labels
    learning_rate - float value that determines step size of descent (learning_rate)
    
    Output
    loss - float, loss value of model
    optimizer - object, optimizer object that minimizes loss
    """
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    return loss, optimizer
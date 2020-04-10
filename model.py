"""
Model Script

Created by: @SteveyyP

"""

# Import Libraries
import tensorflow as tf
from utils.model import *


class ImageSearchModel(object):
    
    def __init__(self, learning_rate, image_size, number_of_classes = 10):
        tf.reset_default_graph()
        self.inputs, self.targets, self.dropout_rate = model_inputs(image_size)
        
        normalized_image = tf.layers.batch_normalization(self.inputs)
        
        # Conv1
        conv_block_1, self.conv_1_features = conv_block(inputs = normalized_image, 
                                                        number_filt = 64, 
                                                        kernel_size = (3,3), 
                                                        strides = (1,1), 
                                                        padding = 'SAME',
                                                        activation = tf.nn.relu,
                                                        max_pool = True,
                                                        batch_norm = True)
        
        # Conv2
        conv_block_2, self.conv_2_features = conv_block(inputs = conv_block_1, 
                                                        number_filt = 128, 
                                                        kernel_size = (3,3), 
                                                        strides = (1,1), 
                                                        padding = 'SAME',
                                                        activation = tf.nn.relu,
                                                        max_pool = True,
                                                        batch_norm = True)
        
        # Conv3
        conv_block_3, self.conv_3_features = conv_block(inputs = conv_block_2, 
                                                        number_filt = 256, 
                                                        kernel_size = (5,5), 
                                                        strides = (1,1), 
                                                        padding = 'SAME',
                                                        activation = tf.nn.relu,
                                                        max_pool = True,
                                                        batch_norm = True)
        
        # Conv 4
        conv_block_4, self.conv_4_features = conv_block(inputs = conv_block_3, 
                                                        number_filt = 512, 
                                                        kernel_size = (5,5), 
                                                        strides = (1,1), 
                                                        padding = 'SAME',
                                                        activation = tf.nn.relu,
                                                        max_pool = True,
                                                        batch_norm = True)
        
        # Flatten
        flat_layer = tf.layers.flatten(conv_block_4)
        
        # Dense 1
        dense_block_1, self.dense_1_features = dense_block(flat_layer,
                                                      units = 128, 
                                                      activation = tf.nn.relu,
                                                      dropout_rate = self.dropout_rate,
                                                      batch_norm = True)
        
        # Dense 2
        dense_block_2, self.dense_2_features = dense_block(dense_block_1,
                                                      units = 256, 
                                                      activation = tf.nn.relu,
                                                      dropout_rate = self.dropout_rate,
                                                      batch_norm = True)
        
        # Dense 3
        dense_block_3, self.dense_3_features = dense_block(dense_block_2,
                                                      units = 512, 
                                                      activation = tf.nn.relu,
                                                      dropout_rate = self.dropout_rate,
                                                      batch_norm = True)
        
        # Dense 4
        dense_block_4, self.dense_4_features = dense_block(dense_block_3,
                                                      units = 1024, 
                                                      activation = tf.nn.relu,
                                                      dropout_rate = self.dropout_rate,
                                                      batch_norm = True)
        
        logits = tf.layers.dense(inputs = dense_block_4, 
                                 units = number_of_classes, 
                                 activation = None)
        
        self.prediction = tf.nn.softmax(logits)
        
        self.loss, self.opt = opt_loss(logits, 
                                       targets = self.targets, 
                                       learning_rate = learning_rate)
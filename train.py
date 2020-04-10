"""
Train Script

Created by: @SteveyyP

"""

# Import Libraries
import tensorflow as tf
import numpy as np
import os
import pickle

import config as cfg
from model import ImageSearchModel

# Training Loop
def train(model, epochs, drop_rate, batch_size, data, save_dir, saver_delta = 0.15):
    """ 
    Training Function 
    
    Input
    model - object, model object
    epochs - int, number of iterations of the dataset to train on
    drop_rate - float, dropout_rate
    batch_size - int, number of training examples to train on at a time
    save_dir - string, filepath location to save model to
    saver_delta - float, maximum difference between training and validation set, to ensure no overfitting
    
    Output
    None
    """
    
    X_train, y_train, X_test, y_test = data
    
    # Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    # Saver
    saver = tf.train.Saver()
    
    best_test_accuracy = 0
    
    # Training Loop
    for epoch in range(epochs):
        train_accuracy = []
        train_loss = []
        for i in tqdm_notebook(range(len(X_train) // batch_size)):
            start_id = i * batch_size
            end_id = start_id + batch_size
            
            X_batch = X_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]
            
            feed_dict = {model.inputs:X_batch, 
                         model.targets:y_batch, 
                         model.dropout_rate:drop_rate}
            
            _, t_loss, preds_t = session.run([model.opt, model.loss, model.prediction], feed_dict=feed_dict)
            
            train_accuracy.append(sparse_accuracy(y_batch, preds_t))
            train_loss.append(t_loss)
            
        print("Epoch: {}/{}".format(epoch, epochs),
              " | Training accuracy: {}".format(np.mean(train_accuracy)),
              " | Training loss: {}".format(np.mean(train_loss)))
        
        test_accuracy = []
        for i in tqdm_notebook(range(len(X_test) // batch_size)):
            start_id = i * batch_size
            end_id = start_id + batch_size
            
            X_batch = X_test[start_id:end_id]
            y_batch = y_test[start_id:end_id]
            
            feed_dict = {model.inputs:X_batch, 
                         model.dropout_rate:0.0}
            
            preds_test = session.run(model.prediction, feed_dict=feed_dict)
            
            test_accuracy.append(sparse_accuracy(y_batch, preds_test))
            
        print("Test accuracy: {}".format(np.mean(test_accuracy)))
        
        # Save Model
        if np.mean(train_accuracy) > np.mean(test_accuracy):
            if np.abs(np.mean(train_accuracy) - np.mean(test_accuracy)) <= saver_delta:
                if np.mean(test_accuracy) >= best_test_accuracy:
                    best_test_accuracy = np.mean(test_accuracy)
                    saver.save(session, "{}/model_epoch_{}.ckpt".format(save_dir, epoch))
                    
    session.close()

# Create Training Set Vectors
def create_training_set_vectors(model, X_train, y_train, batch_size, checkpoint_path, image_size, distance='hamming'):
    """ 
    Create training set vectors and save them in a pickle file 
    
    Inputs
    model - object, model
    X_train - list of training data
    y_train - list of ground truths for training data
    batch_size - int, nunmber of examples that are trained on at a single time
    checkpoint_path - string, location of saved model
    image_size, tuple, size of image (w, h)
    distance - string, distance calculation function
    
    Output
    None
    """
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    
    dense_2_features = []
    dense_4_features = []        
    
    for i in tqdm_notebook(range(len(X_train)//batch_size)):
        start_id = i * batch_size
        end_id = start_id + batch_size
        
        X_batch = X_train[start_id:end_id]
        
        feed_dict = {model.inputs:X_batch, model.dropout_rate:0.0}
        
        dense_2, dense_4 = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)
        
        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)
        
    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        
        with open('hamming_training_vectors_pickle', 'wb') as f:
            pickle.dump(training_vectors, f)
            
    elif distance == 'cosine':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        
        with open('cosine_training_vectors_pickle', 'wb') as f:
            pickle.dump(training_vectors, f)
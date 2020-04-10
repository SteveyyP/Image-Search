"""
General Utils

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


# Cosine Distance
def cosine_distance(training_set_vectors, query_vector, top_n=50):
    """ 
    Calculate cosine distance between query iamge (vectory) and all training set images (vectors) 
    
    Inputs:
    training_set_vectors - list of vectors outputted from a model with inference done
    query_vector - vector you want to calculate distances to
    top_n - number of results to return

    Output:
    sorted list of top_n cosine distances
    """
    
    distances = []
    for i in range(len(training_set_vectors)):
        distances.append(cosine(training_set_vector[i], query_vector[0]))
        
    return np.argsort(distances)[:top_n]

# Hamming Distance
def hamming_distance(training_set_vector, query_vector, top_n=50):
    """ 
    Calculate hamming distance between query image (vector) and all training set images (vectors)

    Inputs:
    training_set_vectors - list of vectors outputted from a model with inference done
    query_vector - vector you want to calculate distances to
    top_n - number of results to return

    Output:
    sorted list of top_n hamming distances 
    """

    distances = []
    for i in range(len(training_set_vector)):
        distances.append(hamming(training_set_vector[i], query_vector[0]))
        
    return np.argsort(distances)[:top_n]  

# Sparse Accuracy
def sparse_accuracy(true_labels, predicted_labels):
    """ 
    Calculates accuracy of model based on softmax outputs 
    
    Inputs
    true_labels - Ground truth values from dataset
    predicted_labels - model predicted values from dataset
    
    Output
    float - correct values/total true values
    """
    
    assert len(true_labels) == len(predicted_labels)
    
    correct = 0
    for i in range(len(true_labels)):
        if np.argmax(predicted_labels[i]) == true_labels[i]:
            correct += 1
            
    return correct / len(true_labels)
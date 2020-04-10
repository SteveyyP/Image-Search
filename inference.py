"""
Inference Script

Created by: @SteveyyP

"""

# Import Libraries
import pickle
import numpy as np
import tensorflow as tf
import config as cfg

from utils.utils import *
from utils.dataset import image_loader

# Simple Inference
def simple_inference(model, session, training_set_vectors, uploaded_image_path, image_size, distance = 'hamming'):
    """ 
    Simple inference for a single image 
    
    Input
    model - object, ImageSearchModel
    session - tensorflow session
    training_set_vectors - vectors created from training set
    uploaded_image_path - string, file to run inference on
    image_size - tuple, size of image (w, h)
    distance - string, which distance metric to use
    
    Output
    closest_ids - list of most similar images
    """
    
    image = image_loader(uploaded_image_path, image_size)
    
    feed_dict = {model.inputs:[image], model.dropout_rate:0.0}
    dense_2_features, dense_4_features = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)
    
    closest_ids = None
    
    if distance == 'hamming':
        
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)  # Binarization 
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        uploaded_image_vector = np.hstack([dense_2_features, dense_4_features])

        closest_ids = hamming_distance(training_set_vectors, uploaded_image_vector)
        
    elif distnace == 'cosine':
        
        uploaded_image_vector = np.hstack([dense_2_features, dense_4_features])
        closest_ids = cosine_distance(training_set_vectors, uploaded_image_vector)
        
    return closest_ids
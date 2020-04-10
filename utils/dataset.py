"""
Dataset Utils

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

# Image Loader
def image_loader(image_path, size):
  """ 
  Load Image from Path 
  
  Inputs
  image_path - string path to image file
  size - tuple of size of image (w, h)
  
  Output
  image - color and size corrected image
  """
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (size), cv2.INTER_CUBIC)
  return image

# Dataset Preprocessing
def dataset_preprocessing(dataset_path, labels_file_path, size, image_path_pickle):
  """ 
  Load Images and labels from folder 
  
  Inputs
  dataset_path - folder location of dataset
  labels_file_path - file location of labels
  size - tuple of size of image (w, h)
  image_path_pickle - file write location of image pickle file
  
  Output
  images - numpy array of images
  labels - numpy array of labels
  """
  
  with open(labels_file_path, 'r') as f:
    classes = f.read().split('\n')[:-1]

  images = []
  labels = []
  image_paths = []

  for image_name in os.listdir(dataset_path):
    try:
      image_path = os.path.join(dataset_path, image_name)
      images.append(image_loader(image_path, size))
      image_paths.append(image_path)
      for i in range(len(classes)):
        if classes[i] in image_name:
          labels.append(i)
    except:
      pass

  with open(image_path_pickle + ".pickle", 'wb') as f:
    pickle.dump(image_paths, f)

  assert len(images) == len(labels)
  return np.array(images), np.array(labels)
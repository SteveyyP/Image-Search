"""
FLASK backend for image to image search
"""

# Import dependencies
import os
import tensorflow as tf
import numpy as np
import pickle
import config as cfg

from model import ImageSearchModel
from inference import simple_inference

# Flask Imports
from flask import Flask, request, render_template, send_from_directory

# Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define our model
model = ImageSearchModel(learning_rate=cfg.LEARNING_RATE, image_size=cfg.IMAGE_SIZE, number_of_classes=cfg.NUMBER_OF_CLASSES)

# Start tf.Session()
session = tf.Session()
session.run(tf.global_variables_initializer())

# Restore Session
saver = tf.train.Saver()
saver.restore(session, 'saver/model_epoch_5.ckpt')

# Load Training Set Vectors
with open('hamming_training_vectors_pickle', 'rb') as f:
    train_vectors = pickle.load(f)

# Load Training Set Paths
with open('training_images_pickle.pickle', 'rb') as f:
    train_paths = pickle.load(f)

# Define Flask App
app = Flask(__name__, static_url_path='/static')

# Define Apps Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Define Upload Function
@app.route("/upload", methods=["POST"])
def upload():
    
    upload_dir = os.path.join(APP_ROOT, "uploads/")
    
    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = "/".join([upload_dir, img_name])
        img.save(destination)

    # Inference
    result = np.array(train_paths)[simple_inference(model, session, train_vectors, os.path.join(upload_dir, img_name), cfg.IMAGE_SIZE)]

    result_final = []

    for img in result:
        result_final.append("images/" +img.split("/")[-1])

    return render_template("results.html", image_name = img_name, result_paths=result_final)
    
# Define Helper Function for Finding Image Paths
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("uploads", filename)

# Start the Application
if __name__ == "__main__":
    app.run(port=5000, debug=True)
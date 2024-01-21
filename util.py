# 1.load library
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st 
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=True)
import datetime

# set threshold > np.max(sample_losses)
threshold = 0.0049

# 2.load image dimension
file = "/home/byungsoo/Documents/fiber/src/2.jpg"
image = cv2.imread(file,0)
height, width = dim = image.shape

# 3.load model
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # input layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(height*width, activation='sigmoid'),
            layers.Reshape((height, width))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder = keras.models.load_model('/home/byungsoo/Documents/fiber/model/')

# 4.utility functions
@st.cache_resource()
def image_loss(image):
    normalized_data = image.astype('float32') / 255.
    # test an image
    encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return round(sample_loss,4)

@st.cache_resource()
def decoded_image(image):
    # generate decoded image
    normalized_data = image.astype('float32') / 255.
    # decode an image and calulate loss
    encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
    decoded = autoencoder.decoder(encoded)
    decoded_image = decoded.numpy()
    decoded_image = decoded_image.reshape(height, width)
    decoded_image = (decoded_image*255)
    # plt.imshow(decoded_image, cmap='gray')
    return decoded_image.astype(np.uint8)

@st.cache_resource()
def show_diff(a,b):
    diff_image = cv2.absdiff(a, b)
    thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)[1]  # Adjust threshold as needed
    blended = cv2.addWeighted(a, 0.4, thresh, 0.6, 0)  # Adjust weights for desired blending
    # plt.axis('off')
    # plt.imshow(blended, cmap='magma')
    # plt.savefig('diff_img.jpg')
    return blended

@st.cache_resource()
def diff_image(a, b):
    '''
    subtract differences between autoencoder and reconstructed image
    '''
    # autoencoder - reconstructed
    inv_01 = cv2.subtract(a, b)

    # reconstructed - autoencoder
    inv_02 = cv2.subtract(b, a)

    # combine differences
    combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
    return combined

@st.cache_resource()
def extract_matched_area(image, template):
  """Extract matched area

  Args:
    image: The image to be cropped.
    template: Reference image. 
  Returns:
    aligned and copped image.
  """

  ######## calculate_angle ##########
  # Initialize SIFT detector
  sift = cv2.SIFT_create()

  # Detect keypoints and compute descriptors
  kp_template, des_template = sift.detectAndCompute(template, None)
  kp_scene, des_scene = sift.detectAndCompute(image, None)

  # Initialize a BFMatcher with default parameters
  bf = cv2.BFMatcher()

  # Match descriptors
  matches = bf.knnMatch(des_template, des_scene, k=2)

  # Apply ratio test to filter good matches
  good_matches = []
  for m, n in matches:
      if m.distance < 0.75 * n.distance:
          good_matches.append(m)

  # Extract matched keypoints
  src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

  # Calculate homography matrix
  H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

  # Calculate rotation angle in radians
  rotation_rad = np.arctan2(H[1, 0], H[0, 0])

  # Convert radians to degrees
  rotation_deg = int(np.degrees(rotation_rad))

  ######## rotate_image ##########
  # Get the image size
  height, width = image.shape[:2]

  # Get the center of the image
  center = (width // 2, height // 2)

  # Create a rotation matrix
  rotation_matrix = cv2.getRotationMatrix2D(center, int(rotation_deg), 1.0)

  # Warp the image using the rotation matrix
  rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

  ######## template_matching ##########
  h, w = template.shape[::] 
  result = cv2.matchTemplate(rotated_image, template, cv2.TM_SQDIFF)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
  top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
  bottom_right = (top_left[0] + w, top_left[1] + h)
  # Crop the image
  matched_area = rotated_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
  return matched_area 

@st.cache_resource()
def formated_datetime():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    return date_time

if __name__ == "__main__":
    '''
    Make inferences for pass and fail
    '''
    # read a file
    file = "/home/byungsoo/Documents/fiber/images/0.jpg" # pass
    file = "/home/byungsoo/Documents/src/9.jpg"    #fail
  
    image = cv2.imread(file, 0)
    plt.imshow(image, cmap='gray')

    # calculate image loss
    loss = image_loss(image)
    print(loss)

    # calculate decoded image
    decoded = decoded_image(image)

    # show difference
    difference = diff_image(image,decoded)
    plt.imshow(difference, cmap='magma')

    '''
    Make inferences for pass and fail
    '''
    os.chdir('/home/byungsoo/Documents/fiber/images')
    sample_losses = []
    for file in os.listdir():
        if file.endswith("jpg"):
            image = cv2.imread(file, 0) 
            normalized_data = image.astype('float32') / 255.

            # decode an image and calulate loss
            encoded = autoencoder.encoder(normalized_data.reshape(-1, dim[0], dim[1]))
            decoded = autoencoder.decoder(encoded)
            loss = tf.keras.losses.mse(decoded, normalized_data)
            sample_loss = np.mean(loss) + np.std(loss)
            sample_loss = round(sample_loss,4)
            sample_losses.append(sample_loss)
            print(file, sample_loss)

    threshold = np.max(sample_losses)
    print(threshold)
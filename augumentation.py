import os
import glob
import util
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "/home/byungsoo/Documents/src"
os.chdir(path)
os.getcwd()

# Load the grayscale image
files = glob.glob("*jpg")
file = files[10]
image = cv2.imread(file, 0)
plt.imshow(image, cmap='gray')

# rename
for file in files:
    dst = 'src_' + file
    print (dst)
    os.rename(file, dst)

# crop
files = glob.glob("*jpg")
for file in files:
    image = cv2.imread(file,0)
    cropped = image[1000:1500,1500:2500]
    cv2.imwrite(file, cropped)

# flip
files = glob.glob("*jpg")
for file in files:
    filename = "flip_" + str(file)
    image = cv2.imread(file,0)
    flipped = cv2.flip(image, 0) 
    cv2.imwrite(filename, flipped)

# rotate
files = glob.glob("*jpg")
for file in files:
    filename = "rotated_" + str(file)
    image = cv2.imread(file,0)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(filename, rotated)


''' threshold '''
# Set the threshold value (adjust as needed)
threshold_value = 80  #127

# Apply thresholding
_, binary_image = cv2.threshold(cropped, threshold_value, 255, cv2.THRESH_BINARY)

# Display the binary image
plt.imshow(binary_image, cmap='gray')

# Find the total area of the foreground objects
background = np.count_nonzero(binary_image != 0)
foreground = np.count_nonzero(binary_image == 0)

# check segmented area
print('tatal area:', background + foreground)
print('dimension: ', cropped.shape[0] * cropped.shape[1])

# Print the total area
print("Total area of foreground objects:", background)


def get_foreground_area(image, threshold = 80):
    # Apply thresholding
    _, binary_image = cv2.threshold(cropped, threshold_value, 255, cv2.THRESH_BINARY)
    foreground_area = np.count_nonzero(binary_image == 0)
    return foreground_area


foreground = get_foreground_area(image)
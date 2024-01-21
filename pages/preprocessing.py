import streamlit as st  
from PIL import Image
import numpy as np
import cv2
import util

with st.sidebar:
        st.subheader("Preprocessing step aligns an image and extracts inspection area")

file = st.file_uploader("", type=["jpg", "png"])

# read template images
template = cv2.imread('/home/byungsoo/Documents/card/images/anomaly/00.jpg',0)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert("L")
    st.image(image, use_column_width=True)

    # convert image to array
    numpy_array = np.array(image)
    
    st.write("Extracted Inspection Area")
    # calculate extracted image
    extracted = util.extract_matched_area(numpy_array, template)

    # show extracted image
    st.image(extracted, use_column_width=True)

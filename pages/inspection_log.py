import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from datetime import date

with st.sidebar:
        # st.title("Defect Detection")
        st.subheader(" Detected defects by date")


with open('sample_losses.pkl', 'rb') as f:
    loss_data = pickle.load(f)


# input inspection date
date = st.date_input("What was the inspection date?", date.today())


# date = datetime.date(2024,1,6)

# # extract date to select only the input date
# df['date'] = pd.to_datetime(df['date_time']).dt.date

if date == pd.to_datetime(df['date_time']).dt.date: 
        st.write("Defect found at ", df)
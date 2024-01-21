import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

with st.sidebar:
        # st.title("Defect Detection")
        st.subheader(" Sample Losse shows distribution of sample losses and sets threshold")

df = pd.read_csv('sample_losses.csv')

fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(df['loss'])
plt.xlabel('loss')
plt.ylabel('number of samples')
st.pyplot(fig_mpl)

cnt = len(df.index)
st.write(f'number of samples: {cnt}')
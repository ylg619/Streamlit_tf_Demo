import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

@st.cache(allow_output_mutation=True)
def load_model_cache():
    PATH = os.path.join(os.path.curdir,'autoencoder.h5')
    model=None
    if os.path.exists(PATH):
        model = load_model(PATH)
    return model

model = load_model_cache()
if model is None:
    print("File not Found")

uploaded_file = st.file_uploader("Gimme  image", type=["png", "jpg", "jpeg"])
res = None
image=None

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    imgArray = np.array(image)

if st.button('Predict'):
    # Send to API, endpoint must accept POST
    if image is not None:
        np.array(imgArray.reshape((28,28))).shape
        res = model.predict(np.array([imgArray.reshape((28,28))])/255)
    if res is not None:
        
       #res_arr = Image.fromarray(res[0])
        st.image(res[0])

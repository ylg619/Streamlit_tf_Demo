import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_model_cache():
    model = load_model('autoencoder.h5')
    return model

model = load_model_cache()


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

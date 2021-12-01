import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
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
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)

if st.button('Predict'):
    # Send to API, endpoint must accept POST
    if image is not None:
        res = model.predict(np.array(imgArray)/255)
    if res:
        res = Image.fromarray(res[0])
        st.image(res)

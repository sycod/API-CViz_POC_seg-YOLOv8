"""Streamlit simple app to predict dog breed from an image"""

import os
import yaml
import logging
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# CONFIG
# local config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
APP_PATH = cfg["app_data"]["local_path"]
MODEL = os.path.join(APP_PATH, cfg["app_data"]["model"])
BREEDS = cfg["app_data"]["breeds"]
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@st.cache_resource
def load_model_cached():
    """Load and cache prediction model"""
    return load_model(MODEL)


def launch_api():
    """Launch API server"""
    # GUI
    st.set_page_config(
        page_title="Which breed is that dog?",
        page_icon="app_favicon.ico",
        layout="centered",
    )

    # create session states
    if "model" not in st.session_state:
        # load and cache model
        st.session_state.model = load_model_cached()
    if "image" not in st.session_state:
        st.session_state.image = None
    if "breed" not in st.session_state:
        st.session_state.breed = None

    st.write("# Send dog image to detect breed among 10")
    st.write(
        f"""
                > Model is trained upon the **Stanford Dogs Dataset**.  
                > It is able to detect **10 dogs breeds**: *{(', ').join(BREEDS)}*  
                > 
                > ➡️ For better results, use **1 dog per image** -- only **JPG and PNG** files allowed -- max size: 200MB"""
    )
    st.write("#### 👇 **Upload your dog image** to predict its breed 👇")

    # user input
    st.session_state.image = st.file_uploader(
        "",
        # "👇 Upload your dog image 👇",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
        # on_change=on_upload,
    )

    st.markdown("""---""")

    if st.session_state.image is not None:
        # preprocess image
        img = Image.open(st.session_state.image)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # predict dog breed
        predictions = st.session_state.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        st.session_state.breed = BREEDS[predicted_class]
        confidence = predictions[0][predicted_class]
        txt = f"Predicted breed: {st.session_state.breed} ({confidence :.2%})"

        st.write(f"#### What a beautiful :blue[{st.session_state.breed}]!")
        st.image(
            st.session_state.image,
            caption=txt,
            use_column_width=True,
        )


if __name__ == "__main__":
    launch_api()

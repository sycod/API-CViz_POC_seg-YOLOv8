"""Streamlit simple app to predict dog breed from an image"""

import os
import yaml
import logging
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# CONFIG
# local config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
APP_PATH = cfg["app_data"]["local_path"]
MODEL_URI = os.path.join(APP_PATH, cfg["app_data"]["model"])
BREEDS = cfg["app_data"]["breeds"]
CONFID = cfg["app_data"]["confidence"]
MAX_DETECT = cfg["app_data"]["max_detections"]
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@st.cache_resource
def load_model_cached():
    """Load and cache prediction model"""
    return YOLO(MODEL_URI)


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

    st.write("# Send dogs image to detect breeds")
    st.write(
        f"""
                > Model is trained upon the **Stanford Dogs Dataset**.  
                > ➡️ able to detect **10 dogs breeds**  
                > ➡️ for **up to 6 simultaneous dogs**: *{(', ').join(BREEDS)}*  
                > ⚠️ Only **JPG and PNG** files allowed -- max size: 200MB"""
    )
    st.write("#### 👇 **Upload your image** to predict dog(s) breed(s) 👇")

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
        img = Image.open(st.session_state.image)
        pred = st.session_state.model.predict(
            img,
            save=False,
            conf=CONFID,
            max_det=MAX_DETECT,
        )

        if len(pred[0].boxes) == 0:
            st.image(
                img,
                caption='Picture with no dog breed detected 😕',
                use_column_width=True
            )
        else:
            # get predicted image
            pred_plot = pred[0].plot()[:, :, ::-1]
            
            # create image caption
            caption = f"Picture of {len(pred[0])} dog(s) with following breeds: "
            for i, p in enumerate(pred[0]):
                # get prediction data
                conf = f"{p.boxes.conf[0].item() :0.0%}"
                cls = BREEDS[int(p.boxes.data[0][-1])]
                # create caption
                caption += f"{cls} ({conf})"
                # add separator
                if i != len(pred[0]) - 1:
                    caption += ", "
            st.image(pred_plot, caption=caption, use_column_width=True)


if __name__ == "__main__":
    launch_api()

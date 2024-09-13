"""Streamlit simple app to predict dog breed from an image"""

import os
import yaml
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from src.data import extract_data, create_img_db


# CONFIG
# *****************************************************************************
# local config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
RAW_DATA_URI = cfg["data"]["raw_data_uri"]
DATA_DIR = cfg["data"]["local_path"]
IMG_DIR = os.path.join(DATA_DIR, cfg["data"]["img_dir"])
ANNOT_DIR = os.path.join(DATA_DIR, cfg["data"]["annot_dir"])
IMG_DB_URI = os.path.join(DATA_DIR, cfg["data"]["img_db_uri"])
BREEDS = cfg["model"]["classes"]
APP_PATH = cfg["app_data"]["local_path"]
TRAIN_OUTPUTS_DIR = os.path.join(APP_PATH, cfg["app_data"]["train_outputs_dir"])
MODEL_URI = os.path.join(APP_PATH, cfg["app_data"]["model"])
CONFID = cfg["app_data"]["min_confidence"]
MAX_DETECT = cfg["app_data"]["max_detections"]

# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# LOAD DATA & CREATE DB
# *****************************************************************************
if os.path.exists(IMG_DB_URI):
    img_df = pd.read_csv(IMG_DB_URI, index_col=0)
else:
    # extract data
    logging.info("⚙️ Extracting data...")
    extract_data(RAW_DATA_URI, DATA_DIR)
    
    # create database
    logging.info("⚙️ Creating database...")
    img_df = create_img_db(IMG_DIR, ANNOT_DIR, IMG_DB_URI)

# create readable breeds list
breeds_read = [b.lower() for b in BREEDS]


# CACHE RESOURCES
# *****************************************************************************
@st.cache_resource
def load_img_db_cached():
    """Load and cache image database"""
    return img_df


@st.cache_resource
def load_model_cached():
    """Load and cache prediction model"""
    return YOLO(MODEL_URI)


# CALLBACKS
# *****************************************************************************
def compute_data_hist():
    # compute
    n_per_breed = st.session_state.img_db["class_label"].value_counts().values
    n_total = st.session_state.img_db.shape[0]
    n_min = n_per_breed.min()
    n_max = n_per_breed.max()
    n_med = np.median(n_per_breed)
    n_mean = n_per_breed.mean()

    # plot
    hist = plt.figure(figsize=(8, 3))
    ax = sns.histplot(data=img_df, x="class_label", kde=True)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    plt.title(
        f"Number of images per breed\n(min {n_min}, median {n_med :.0f}, mean {n_mean :.0f}, max {n_max})"
    )

    # caption
    st.session_state.img_db_hist_caption = f"Data histogram, showing distribution of {n_total} images (min {n_min}, median {n_med :.0f}, mean {n_mean :.0f}, max {n_max})"

    return hist


def erase_image():
    st.session_state.raw_breed_list = None
    st.session_state.raw_breed = None
    st.session_state.raw_image = None
    st.session_state.raw_image_caption = None


def display_img_from_raw_breed():
    # pick random image in database
    df = st.session_state.img_db
    df = df.loc[df["class_label"] == st.session_state.raw_breed]
    rdm_index = random.choice(df.index)
    rdm_img_data = df.loc[rdm_index]
    # get image data
    rdm_img_ID = rdm_img_data["ID"]
    rdm_img_uri = rdm_img_data["img_uri"]
    rdm_img_width = rdm_img_data["width"]
    rdm_img_height = rdm_img_data["height"]
    bb_xmin = rdm_img_data["bb_xmin"]
    bb_ymin = rdm_img_data["bb_ymin"]
    bb_xmax = rdm_img_data["bb_xmax"]
    bb_ymax = rdm_img_data["bb_ymax"]
    bb_width = bb_xmax - bb_xmin
    bb_height = bb_ymax - bb_ymin
    bb_w_ratio = f"{bb_width / rdm_img_width :.0%}"
    bb_h_ratio = f"{bb_height / rdm_img_height :.0%}"
    rdm_img_caption = f"{rdm_img_width}(w) x {rdm_img_height}(h) image showing {st.session_state.raw_breed} dog, occupying {bb_w_ratio} width and {bb_h_ratio} height"

    # render image
    fig, ax = plt.subplots()
    img = Image.open(rdm_img_uri)
    ax.imshow(img)
    ax.add_patch(
        patches.Rectangle(
            (bb_xmin, bb_ymin),
            bb_width,
            bb_height,
            linewidth=2,
            edgecolor='cyan',
            facecolor='none'
        )
    )
    ax.axis("off")
    ax.set_title(f"Image ID: {rdm_img_ID} ; size (W x H): {rdm_img_width} x {rdm_img_height}")
    plt.tight_layout()

    st.session_state.raw_image_caption = rdm_img_caption
    st.session_state.raw_image = fig
    plt.close()


def raw_breed_from_list():
    st.session_state.raw_breed = st.session_state.raw_breed_list
    display_img_from_raw_breed()


def prev_da_img():
    if st.session_state.da_img_index > 0:
        st.session_state.da_img_index -= 1


def next_da_img():
    if st.session_state.da_img_index < 2:
        st.session_state.da_img_index += 1


def val_img():
    if st.session_state.val_img == "labels":
        st.session_state.val_img = "pred"
    else:
        st.session_state.val_img = "labels"


def update_confid():
    st.session_state.confidence = st.session_state.confidence_slider


def update_max_detect():
    st.session_state.max_detections = st.session_state.max_detections_slider


# APP
# *****************************************************************************
def launch_api():
    """Launch API server"""
    # GUI
    # *************************************************************************
    st.set_page_config(
        page_title="Which breeds are these dogs?",
        page_icon="app_favicon.ico",
        layout="centered",
    )

    # load and cache raw images database
    if "img_db" not in st.session_state:
        st.session_state.img_db = load_img_db_cached()
    # compute data histogram
    if "img_db_hist" not in st.session_state:
        st.session_state.img_db_hist = compute_data_hist()
    if "img_db_hist_caption" not in st.session_state:
        st.session_state.img_db_hist_caption = None
    # load and cache model
    if "model" not in st.session_state:
        st.session_state.model = load_model_cached()
    # raw data, for EDA
    if "raw_breed_list" not in st.session_state:
        st.session_state.raw_breed_list = None
    if "raw_breed" not in st.session_state:
        st.session_state.raw_breed = None
    if "raw_image" not in st.session_state:
        st.session_state.raw_image = None
    if "raw_image_caption" not in st.session_state:
        st.session_state.raw_image_caption = None
    if "da_img_index" not in st.session_state:
        st.session_state.da_img_index = 0
    if "val_img" not in st.session_state:
        st.session_state.val_img = "labels"
    
    # for inference
    if "confidence" not in st.session_state:
        st.session_state.confidence = CONFID
    if "max_detections" not in st.session_state:
        st.session_state.max_detections = MAX_DETECT
    if "image" not in st.session_state:
        st.session_state.image = None
    if "breed" not in st.session_state:
        st.session_state.breed = None

    st.write("# Dogs breeds detector")

    # EDA
    # *************************************************************************
    st.write("## Original data")
    st.write("Model is trained upon the **[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)**, with a **selection of 10 dogs breeds** :")
    st.write(f"*{(', ').join(breeds_read)}*")

    # histogram
    st.write("### Distribution")
    st.write(f"It includes **{st.session_state.img_db.shape[0]} images**, distributed as follows:")
    st.pyplot(st.session_state.img_db_hist)
    st.write(f"{st.session_state.img_db_hist_caption}")

    # images visualizer
    st.write("### Exploration")
    st.selectbox(
        "Explore initial dataset by **choosing your favorite dog breed** and see a **random image**:",
        BREEDS,
        placeholder="Choose a dog breed",
        key="raw_breed_list",
        on_change=raw_breed_from_list,
    )

    if st.session_state.raw_image:
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "Reload image 🔄️",
                on_click=display_img_from_raw_breed,
                use_container_width=True,
                # type="primary",
            )
        with col2:
            st.button(
                "Reset ❌",
                on_click=erase_image,
                use_container_width=True,
            )

        st.pyplot(st.session_state.raw_image)
        st.write(f"{st.session_state.raw_image_caption}")

    st.markdown("""---""")

    # TRAINING
    # *************************************************************************
    st.write("## Training the model")

    # data augmentation
    st.write("## Training the model")
    st.write("Model was trained using **data augmentation on the training set**:")
    # train images carrousel
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        if st.session_state.da_img_index <= 0:
            st.button("⬅️", disabled=True)
        else:
            st.button("⬅️", disabled=False, on_click=prev_da_img)
    with col3:
        if st.session_state.da_img_index >= 2:
            st.button("➡️", disabled=True)
        else:
            st.button("➡️", disabled=False, on_click=next_da_img)
    st.image(
        os.path.join(
            TRAIN_OUTPUTS_DIR,
            "train_batch" + str(st.session_state.da_img_index) + ".jpg"
        ),
        caption="Augmented training images representing transformed raw images",
    )

    # validation set predictions
    st.write("And its weights were adjusted upon their **performance on the validation set**, for **class predictions** and **object detection**:")
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        if st.session_state.val_img == "labels":
            st.button("Classification", disabled=True)
        else:
            st.button("Classification", disabled=False, on_click=val_img)
    with col3:
        if st.session_state.val_img == "pred":
            st.button("Detection", disabled=True)
        else:
            st.button("Detection", disabled=False, on_click=val_img)
    st.image(
        os.path.join(
            TRAIN_OUTPUTS_DIR,
            "val_batch0_" + st.session_state.val_img + ".jpg"
        ),
        caption="Model predictions on validation set during training, showing dogs with breeds and eventual bounding box with confidence",
    )

    st.markdown("""---""")

    # INFERENCE
    # *************************************************************************
    st.write("## Test it")
    st.write("Guidelines:")
    st.write(
        f"""
        > ➡️ 🚧(upon selector) for **up to 6 simultaneous dogs** in following 10 breeds: *{(', ').join(breeds_read)}*  
        > ⚠️ Only **JPG and PNG** files allowed -- max size: 200MB"""
    )
    st.write("#### 👇 **Upload your image** to predict dog(s) breed(s) 👇")

    # user input
    st.session_state.image = st.file_uploader(
        "",
        # "👇 Upload your dog image 👇",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
    )

    if st.session_state.image is not None:
        # adjust confidence and maximum detections
        col1, col2 = st.columns(2)
        with col1:
            confidence_slider = st.slider(
                "Confidence",
                value=st.session_state.confidence,
                min_value=0.05,
                max_value=0.95,
                step=0.05,
                key="confidence_slider",
                on_change=update_confid,
            )
        with col2:
            max_detections_slider = st.slider(
                "Maximum detections",
                value=st.session_state.max_detections,
                min_value=1,
                max_value=20,
                step=1,
                key="max_detections_slider",
                on_change=update_max_detect,
            )

        # display image
        img = Image.open(st.session_state.image)
        pred = st.session_state.model.predict(
            img,
            save=False,
            conf=st.session_state.confidence,
            max_det=st.session_state.max_detections,
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

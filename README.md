Title | Icon | Licence
:---:|:---:|:---:
Dogs breeds detection | 🐶 | MIT

> 👉 App is also available online at [wholetthedogsoutagain.streamlit.app](https://wholetthedogsoutagain.streamlit.app/)

# General information

The application model is trained upon the **Stanford Dogs Dataset**, which is part of the ImageNet dataset.

This is an improvement of former model **EfficientNet B0 model** (available at [this URL](https://wholetthedogsout.streamlit.app/)), which was made to classify several dogs among these 10 dogs breeds:
- brabancon griffon
- cardigan
- leonberg
- basenji
- boxer
- chow
- dhole
- dingo
- malamute
- papillon

🦾 Based on the **YOLOv8 model**, this evolution is **now able to perform detection** among these breeds.  

# Installation

It assumes you have **Python 3.11 installed** on your machine but it may work with lower versions.

Once this git repository cloned on your computer, **use `make install`** to update PIP and install all requirements, located in the *requirements.txt* file.

# Run the app

To **run the app locally**, just use this command in your terminal: `streamlit run 4_dashboard.py`

# Usage

As precised on the app screen, you **just have to upload any dog image** (with at least one dog belonging to one of the 10 specified dogs breeds).  
To predict again, upload another one and it will replace the previous prediction.

🎛️ Update detection parameters for a **live interaction with model**:
- minimal **confidence** (from 0.05 to 0.95)
- **maximum detections** (up to 20 simultaneous detections)

➡️ For better results:
- only JPG files are allowed
- maximum image size: 200MB

🚱 To avoid data leakage, use prior :
- Internet images of any of these breeds
- avoid using any of the Stanford Dogs DataSet
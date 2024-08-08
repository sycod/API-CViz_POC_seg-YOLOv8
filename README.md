Title | Icon | Licence
:---:|:---:|:---:
Dogs breeds detection | 🐶 | MIT

> 👉 App is also available online at [wholetthedogsout.streamlit.app](https://wholetthedogsout.streamlit.app/)

# General information

The application model is trained upon the **Stanford Dogs Dataset**, which is part of the ImageNet dataset.

This is an **EfficientNet B0 model**, with transfer learning.

It is able to detect 10 dogs breeds:
- Brabancon_griffon
- Cardigan
- Leonberg
- basenji
- boxer
- chow
- dhole
- dingo
- malamute
- papillon

# Installation

It assumes you have **Python 3.11 installed** on your machine but it may work with lower versions.

Once this git repository cloned on your computer, **use `make install`** to update PIP and install all requirements, located in the *requirements.txt* file.

# Run the app

To **run the app locally**, just use this command in your terminal: `streamlit run 4_app.py`

# Usage

As precised on the app screen, you **just have to upload any dog image** (belonging to one of the 10 specified dogs breeds) so the model can predict the dog breed.  
A **confidence rate** is also displayed when predicting.

To predict again, upload another one and it will replace the previous prediction.

➡️ For better results:
- use 1 dog per image (however several dogs of the same breed don't seem to impact the predictions)
- only JPG and PNG files are allowed
- maximum image size: 200MB

🚱 To avoid data leakage, use prior :
- Internet images of any of these breeds
- avoid using any of the Stanford Dogs DataSet
- or **images included in the *app_data* folder** because even though they are part of the Stanford Dogs Dataset, this model wasn't trained over these images
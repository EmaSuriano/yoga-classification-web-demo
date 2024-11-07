import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import json
from util import classify
import pandas as pd

# set title
st.title("Yoga classification")

# set header
st.header("Please upload a yoga position image")

with open("./model/Poses.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    df = pd.DataFrame(data["Poses"])

with st.expander("See trained proses"):
    st.data_editor(
        df,
        column_config={
            "sanskrit_name": "Sankrit name",
            "english_name": "English name",
            "img_url": st.column_config.ImageColumn("Pose Image"),
        },
        hide_index=True,
    )


class_names = df.apply(
    lambda row: f"{row['sanskrit_name']} / {row['english_name']}",
    axis=1,
).tolist()

file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])

# load classifier
model = load_model("./model/vgg16-ft.h5")

# display image
if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

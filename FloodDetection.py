import cv2
from PIL import Image
import pandas as pd
import streamlit as st
from streamlit_image_comparison import image_comparison
import segmentation_models_pytorch as smp
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

model_path = 'Unet'
model_path1 = 'FPN'
legend = "Legend.png"
placeholder_img = '6397.jpg'

# Functions
def load_dummy_model():               # Alternative when the model is missing
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=10
    )
    return model

def load_model(smodel):

    if smodel == "Unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes)
        )
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    else:
        model = smp.FPN(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(classes)
        )
        state_dict = torch.load(model_path1, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
    return model

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def decode_segmap(mask, colors, classes, selected_classes):
    seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        class_name = classes[i]
        if class_name in selected_classes:
            seg_img[mask == i] = color
    return seg_img

def class_counter(mask, classes_palette):
    pixels = [tuple(pixel) for row in mask for pixel in row]
    pixel_counts = Counter(pixels)
    df = pd.DataFrame(list(pixel_counts.items()), columns=['RGB', 'Count'])
    df['Class'] = df['RGB'].map(classes_palette)
    df['Percentage'] = (df['Count'] / df['Count'].sum()) * 100
    return df

def plot_class_distribution(df):
    plt.figure(figsize=(8, 7), dpi=150)
    bars = plt.bar(df['Class'], df['Count'], color=[tuple(np.round(np.array(rgb, dtype=np.float32) / 255 , 5)) for rgb in df['RGB']])

    for bar, count in zip(bars, df['Count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{count}\n({count / df["Count"].sum() * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=8)

    plt.ylim(0, max(df['Count']) + max(df['Count']) * 0.15)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

def percentages(df, A, B):

    flooded = int(df.loc[df['Class'] == A, 'Count'].values[0]) if not df.loc[df['Class'] == A].empty else 0
    non_flooded = int(df.loc[df['Class'] == B, 'Count'].values[0]) if not df.loc[df['Class'] == B].empty else 0
    total = flooded + non_flooded

    if total == 0:
        return 0.0

    return round((flooded / total) * 100, 2)

def container_response(color, message):
    return  result_container.markdown(f"""
                                        <p style="background-color: {color};
                                          color:#000;
                                          text-align:center;
                                          font-size:18px;
                                          border-radius:0.5rem;
                                          padding:1% 2%;
                                          margin:10px 20px;">
                                        {message}
                            </p>""", unsafe_allow_html=True)

# Palette
classes = ['Background', 'Building Flooded', 'Building Non-Flooded', 'Road Flooded',
           'Road Non-Flooded',  'Water', 'Tree', 'Vehicle',  'Pool', 'Grass']
RGBs = [[0.0, 0.0, 0.0],[255.0, 0.0, 0.0], [180.0, 120.0, 120.0], [160.0, 150.0, 20.0], [140.0, 140.0, 140.0],
        [61.0, 230.0, 250.0], [0.0, 82.0, 255.0],[255.0, 0.0, 245.0], [255.0, 235.0, 0.0], [4.0, 250.0, 7.0]]
RGBs_tuples = [tuple(rgb) for rgb in RGBs]
classes_palette = dict(zip(RGBs_tuples, classes))


## Introduction

st.set_page_config(page_title='Flood Detection' ,layout="centered")
st.title("Flood Detection")
st.markdown("##### Advanced Computational Techniques for Big Imaging and Signal Data")

st.write("Upload an image captured by small unmanned aerial systems (UAS) to obtain a detailed 10-class segmentation "
"specifically designed to identify flood-induced damages in natural disaster scenarios. "
 "The segmentation models employed have been fine-tuned on the FloodNet dataset, which contains high-resolution"
 " aerial imagery with detailed semantic annotations. For additional information, please refer to the FloodNet GitHub repository "
 "[here](https://github.com/BinaLab/FloodNet-Supervised_v1.0?tab=readme-ov-file) and the related "
 "research paper [here](https://ieeexplore.ieee.org/document/9460988).")
st.write("")

# Instructions
with st.container(border = 1):
    st.markdown("##### Instructions:")
    st.markdown('''
    - Upload an aerial image.
    - The segmentation will process automatically.
    - Use the side panel to customize your view:
        - Choose between different visualization options.
        - Choose between different segmentation models.
        - Select specific classes to highlight.
    - Review the frequency histogram below for a statistical overview.
    ''')

# Legend
#with st.container(border = 1):
    #st.markdown('*Legend*')

with st.expander('##### Legend:'):
    st.image(legend)

## Sidebar

st.sidebar.write("## Upload Aerial Image")
my_upload = st.sidebar.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
st.sidebar.divider()

st.sidebar.write("## Choose Visualization Type")
visualization = st.sidebar.selectbox(
    "Visualization type:",
    ("Slider", "Comparison"), )

if visualization == "Slider":
    alpha = st.sidebar.slider("Transparence:", 0.0, 1.0, 0.5)
st.sidebar.divider()

st.sidebar.write("## Choose Segmentation Model")
smodel = st.sidebar.selectbox(
    "Segmentation model:",
    ("Unet", "FPN"), )
st.sidebar.divider()

st.sidebar.write("## Choose segments of interest")
selected_classes = st.sidebar.multiselect("Select classes:", classes, default=classes, placeholder="Choose an option")
st.sidebar.divider()

st.sidebar.info("Made by Davide Vettore - 868855")


## Main page

# Image loading
if my_upload is not None:
    image = np.array(Image.open(my_upload))
else:
    image = np.array(Image.open(placeholder_img))

# Preprocess the image
processed_image = preprocess_image(image)

# Convert the processed image to a tensor and add batch dimension
input_tensor = torch.tensor(processed_image).permute(2, 0, 1).unsqueeze(0).float()

# Segmentation
#model = load_dummy_model()
model = load_model(smodel)
model.eval()
with torch.no_grad():
    output = model(input_tensor)

output = output.argmax(dim=1).squeeze().cpu().numpy()
segmented_image = decode_segmap(output, RGBs, classes, selected_classes)

# For dummy model grayscale
#segmented_image = torch.argmax(output.squeeze(), dim=0).numpy()
#segmented_image = (segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min())

# Display images side by side

if visualization=="Comparison":
    st.markdown('##### Side-by-side comparison')
    if my_upload is None:
        st.markdown('*Placeholder Image*')

    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_image, caption='Preprocessed Image', use_column_width=True)

    with col2:
        st.image(segmented_image, caption='Segmented Image', use_column_width=True, clamp=True)
else:
    st.markdown('##### Slider of Uploaded Image and Segmented Image')
    if my_upload is None:
        st.markdown('*Placeholder Image*')

    overlay = cv2.addWeighted(cv2.resize(segmented_image, (image.shape[1], image.shape[0])),
                              alpha, image, 1-alpha, 0)

    image_comparison(
        img1=image,
        img2=overlay,
        label1="Actual Image",
        label2="Segmented Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

df = class_counter(segmented_image, classes_palette)

# Message
flooded_building_percentage = percentages(df, 'Building Flooded', 'Building Non-Flooded')
flooded_road_percentage = percentages(df, 'Road Flooded', 'Road Non-Flooded')
result_container = st.container()
if flooded_building_percentage > 20 or flooded_road_percentage > 20:
    container_response("rgba(255, 0, 0, 0.6)",
                       f"The automatically analyzed aerial image shows that {flooded_building_percentage:.2f}% of buildings and "
                        f"{flooded_road_percentage:.2f}% of roads are flooded. Quick action is needed to help these areas!")
else:
    container_response("rgba(0, 255, 0, 0.6)",
                       f"The automatically analyzed aerial image doesn't show any flood-induced damage")


## Statistics part

st.divider()
st.markdown('##### Frequency Histogram')

plot_class_distribution(df)


st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')

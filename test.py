import streamlit as st
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import torch


def load_dummy_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=10
    )
    return model


def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


def main():
    st.title("Simple Image Segmentation App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = np.array(Image.open(uploaded_file))
        processed_image = preprocess_image(image)

        # Load the dummy model
        model = load_dummy_model()

        # Convert the processed image to a tensor and add batch dimension
        input_tensor = torch.tensor(processed_image).permute(2, 0, 1).unsqueeze(0).float()

        # Dummy segmentation (the model isn't trained so this won't produce meaningful results)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        # Convert model output to a single-channel image by taking the argmax over the channel dimension
        segmented_image = torch.argmax(output.squeeze(), dim=0).numpy()

        # Normalize segmented image for display
        segmented_image = (segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min())

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(processed_image, caption='Preprocessed Image', use_column_width=True)

        with col2:
            st.image(segmented_image, caption='Segmented Image', use_column_width=True, clamp=True)


if __name__ == "__main__":
    main()

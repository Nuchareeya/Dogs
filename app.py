import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page title
st.title("YOLO Image Detection App 🐶")

# Load YOLO model
# You should replace "weights/your-model.pt" with the actual path to your model file.
# For example, it might be "best.pt" or "yolo11n.pt".
model = YOLO("yolo11n.pt") 

# Upload image
uploaded_image = st.file_uploader(
    "Upload an image (jpg, jpeg, png)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Convert the uploaded image to a format the model can use
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Run YOLO inference
    st.info("Running YOLO object detection...")
    results = model.predict(image_np, conf=0.4)

    # Get the image with bounding boxes and display it
    result_image = results[0].plot()
    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # Extract detection results and count people
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    # Count the number of dogs detected
    dog_count = class_names.count("dog")
    st.write(f"Number of dogs detected: **{dog_count}**")
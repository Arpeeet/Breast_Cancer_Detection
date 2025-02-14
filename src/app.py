import gradio as gr
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Ensure compatibility with custom deserialization if needed
try:
    model = load_model(
        'C:/P/bcd-src/saved_models/2024112-91925/model.keras',
        custom_objects={'Functional': tf.keras.models.Functional}  # Add this line
    )
except TypeError as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit if the model cannot be loaded

def predict_image(img):
    img = np.resize(img, (50, 50, 3))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    class_labels = ['IDC -ve', 'IDC +ve']
    label = class_labels[class_idx]

    return (label, confidence)

app = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type='numpy', label="Upload Breast Histopathology Image"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Number(label="Confidence Score")
    ],
    title="IDC Classifier",
    description="Upload a Breast Histopathology Image.",
    flagging_mode='never'
)

app.launch()

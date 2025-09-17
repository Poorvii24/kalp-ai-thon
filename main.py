import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load trained model
MODEL_PATH = "deepfake_xception.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # compile=False avoids metrics warnings

IMG_SIZE = (160, 160)

st.title("🛡️ Deepfake Defender")
st.write("Upload a video or image, and this tool will detect if it's REAL or FAKE.")

uploaded_file = st.file_uploader("Upload Video/Image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

def preprocess_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

if uploaded_file is not None:
    # If image
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image")

        pred = model.predict(preprocess_frame(image))[0][0]
        score = float(pred) * 100
        st.metric("Deepfake Confidence", f"{score:.2f}%")

        if score > 50:
            st.error("⚠️ Likely FAKE")
        else:
            st.success("✅ Likely REAL")

    # If video
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frames_pred = []
        count = 0
        while cap.isOpened() and count < 5:  # first 5 frames only
            ret, frame = cap.read()
            if not ret:
                break
            pred = model.predict(preprocess_frame(frame))[0][0]
            frames_pred.append(pred)
            count += 1
        cap.release()

        avg_pred = np.mean(frames_pred)
        score = float(avg_pred) * 100

        st.video(tfile.name)
        st.metric("Deepfake Confidence", f"{score:.2f}%")

        if score > 50:
            st.error("⚠️ Likely FAKE")
        else:
            st.success("✅ Likely REAL")

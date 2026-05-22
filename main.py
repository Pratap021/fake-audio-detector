import streamlit as st
import tensorflow as tf
import librosa
import numpy as np

# Global constant
max_length = 204

# Load the trained model
model = tf.keras.models.load_model("Model/ann.h5")

# Feature Extraction Process
def extract_features(audio_file, sr=16000):
    audio, _ = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # Padding or trimming the MFCCs to have a consistent shape
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]

    return mfccs

# Prediction function
def detect_fake_voice(audio_file, model):
    mfccs = extract_features(audio_file, sr=16000)
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    return prediction

# Feedback function (to interpret prediction)
def feedback(prediction):
    # Assuming the prediction is a score between 0 and 1
    # You can adjust the threshold based on your model's output
    threshold = 0.5
    if prediction[0][0] > threshold:
        return "REAL"
    else:
        return "FAKE"

# Streamlit App
st.title("Fake Voice Detector")
st.write("Upload an audio file to check if the voice is fake or real.")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    with open("temp_audio_file.wav", "wb") as f:
        f.write(audio_file.read())

    # Make prediction
    prediction = detect_fake_voice("temp_audio_file.wav", model)
    prediction_result = feedback(prediction)
    score = prediction[0][0]

    # Display results
    st.write(f"**Prediction:** {prediction_result}")
    st.write(f"**Prediction Score:** {score}")

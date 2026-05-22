import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------
# CONFIG
# -------------------------------
sr = 16000
max_length = 204
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ann_model = load_model(os.path.join(BASE_DIR, "model", "ann.h5"))
lstm_model = load_model(os.path.join(BASE_DIR, "model", "lstm.h5"))

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    mfcc = mfcc[:, :max_length]

    if mfcc.shape[1] < max_length:
        pad = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")

    mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)

    return mfcc.reshape(1, 20, max_length)

# -------------------------------
# WAVEFORM
# -------------------------------
def plot_waveform(file_path):
    audio, _ = librosa.load(file_path, sr=None)

    fig, ax = plt.subplots()
    ax.plot(audio, linewidth=0.6, color="#00E5FF")
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.set_title("Waveform", color="white")

    return fig

# -------------------------------
# UI CONFIG (PRO THEME)
# -------------------------------
st.set_page_config(
    page_title="AI Fake Audio Detector",
    page_icon="🎧",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (GLASSMORPHISM UI)
# -------------------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main-title {
    font-size: 42px;
    text-align: center;
    font-weight: bold;
    color: #00E5FF;
}

.sub-title {
    text-align: center;
    color: #aaa;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}

.metric-box {
    background: rgba(0,229,255,0.1);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="main-title">🎧 AI Fake Audio Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Deep Learning System (ANN + LSTM)</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ANN", "LSTM", "Ensemble"]
)

st.sidebar.info("Upload audio and detect fake or real speech using AI models.")

# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Audio File", type=["wav", "mp3", "m4a"])

# -------------------------------
# MAIN UI
# -------------------------------
if uploaded_file:

    st.success("File uploaded successfully!")

    file_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.audio(file_bytes)

    with col2:
        st.pyplot(plot_waveform(temp_path))

    st.divider()

    # BUTTON
    if st.button("🚀 Analyze Audio", use_container_width=True):

        with st.spinner("AI is analyzing audio pattern..."):

            features = extract_features(temp_path)

            ann_pred = float(ann_model.predict(features)[0][0])
            lstm_pred = float(lstm_model.predict(features)[0][0])

            if model_choice == "ANN":
                pred = ann_pred
            elif model_choice == "LSTM":
                pred = lstm_pred
            else:
                pred = (ann_pred + lstm_pred) / 2

        # -------------------------------
        # RESULT DASHBOARD
        # -------------------------------
        st.markdown("## 📊 Prediction Dashboard")

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown('<div class="metric-box">ANN<br><h2>{:.2f}</h2></div>'.format(ann_pred), unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="metric-box">LSTM<br><h2>{:.2f}</h2></div>'.format(lstm_pred), unsafe_allow_html=True)

        with colC:
            st.markdown('<div class="metric-box">FINAL<br><h2>{:.2f}</h2></div>'.format(pred), unsafe_allow_html=True)

        st.progress(min(max(pred, 0), 1))

        # STATUS
        st.markdown("## 🧠 AI Decision")

        if pred < 0.35:
            st.success("✅ REAL AUDIO DETECTED")

        elif pred > 0.65:
            st.error("❌ FAKE AUDIO DETECTED")

        else:
            st.warning("⚠️ UNCERTAIN RESULT")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("<center>🔥 Built for Major Project | AI Audio Detection System</center>", unsafe_allow_html=True)
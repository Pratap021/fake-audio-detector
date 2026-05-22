# Fake Voice Detector

A production-ready Flask frontend for a Fake Voice Detector ML project. The backend loads a trained model and renders audio analysis results using a polished Jinja2 interface.

## Project Structure

- `app.py` — Flask application with upload, prediction, and metadata extraction routes
- `templates/index.html` — Jinja2 template for the frontend UI
- `static/css/style.css` — Page styling for the dashboard, cards, and animations
- `static/js/app.js` — Client-side interaction, drag-and-drop, waveform rendering, and animations
- `requirements.txt` — Required Python dependencies for Flask, audio processing, and machine learning
- `model.h5` or `model.pkl` — Placeholder path for your trained model file (replace as needed)

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Place your trained model in the project root and update `MODEL_PATH` inside `app.py` if needed.

## Run the App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Plug in Your Own Model

1. Replace the `load_model_stub` implementation in `app.py` with your real model loader:

- For TensorFlow/Keras: `from tensorflow.keras.models import load_model`
- For scikit-learn: `import joblib`

2. Load your model from the saved file:

```python
model = load_model(MODEL_PATH)
```

or

```python
model = joblib.load(MODEL_PATH)
```

3. Update `predict_audio(features)` to match your model input shape and output format.

## Notes

- The UI renders waveform analysis and prediction results only after a successful upload.
- Supported audio formats: WAV, MP3, OGG.
- All styling and JavaScript are self-contained and work offline.

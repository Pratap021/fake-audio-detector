import sys
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------------
# FIX IMPORT PATH (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.feature_extractor import extract_features

# -------------------------------
sr = 16000
max_length = 204

# -------------------------------
def load_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]

# -------------------------------
def train_model():

    print("\n🚀 ANN Training Started\n")

    real_dir = os.path.join(BASE_DIR, "audio_dataset", "training", "real")
    fake_dir = os.path.join(BASE_DIR, "audio_dataset", "training", "fake")

    real_files = load_files(real_dir)
    fake_files = load_files(fake_dir)

    print("Real:", len(real_files), "Fake:", len(fake_files))

    X_real = np.array([extract_features(f) for f in real_files])
    X_fake = np.array([extract_features(f) for f in fake_files])

    y_real = np.ones(len(X_real))
    y_fake = np.zeros(len(X_fake))

    X = np.concatenate([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])

    # normalization
    X = X / (np.max(np.abs(X)) + 1e-6)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, max_length)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=16,
              validation_data=(X_test, y_test))

    model_path = os.path.join(BASE_DIR, "model", "ann.h5")
    model.save(model_path)

    print("ANN saved at:", model_path)

if __name__ == "__main__":
    train_model()
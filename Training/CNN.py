import tensorflow as tf
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split

sr = 16000
max_length = 204

def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    if mfcc.shape[1] < max_length:
        pad = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc

def load_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

real_path = os.path.join(BASE_DIR, "audio_dataset", "training", "real")
fake_path = os.path.join(BASE_DIR, "audio_dataset", "training", "fake")

real_files = load_files(real_path)
fake_files = load_files(fake_path)

X_real = np.array([extract_features(f) for f in real_files])
X_fake = np.array([extract_features(f) for f in fake_files])

y_real = np.ones(len(X_real))
y_fake = np.zeros(len(X_fake))

X = np.concatenate([X_real, X_fake])
y = np.concatenate([y_real, y_fake])

# CNN needs channel dimension
X = X.reshape(-1, 20, max_length, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(20, max_length, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

model.save(os.path.join(BASE_DIR, "model", "cnn.h5"))

print("✅ CNN Model saved!")
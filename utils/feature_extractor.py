import librosa
import numpy as np

SR = 16000
MAX_LEN = 128  # spectrogram width

def extract_features(file_path):

    audio, _ = librosa.load(file_path, sr=SR)

    # 🔥 MEL SPECTROGRAM (BEST FOR AUDIO DETECTION)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    mel = librosa.power_to_db(mel)

    # resize width
    mel = mel[:, :MAX_LEN]

    if mel.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)))

    # normalize
    mel = mel / (np.max(np.abs(mel)) + 1e-6)

    return mel
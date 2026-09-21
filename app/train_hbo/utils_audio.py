import librosa

def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

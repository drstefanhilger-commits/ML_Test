import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import shutil

SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

MODEL_PATH = Path("models/binary/model_binary_int8.tflite")

NO_DRONE_SRC = Path("data/ESC-50/audio")
FP_DST = Path("data/train/no_drone_hard")


def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < FRAME_LEN:
        y = np.pad(y, (0, FRAME_LEN - len(y)))
    else:
        y = y[:FRAME_LEN]

    window = np.hanning(FRAME_LEN)
    S = librosa.stft(
        y,
        n_fft=FRAME_LEN,
        hop_length=FRAME_LEN // 2,
        window=window,
        center=False
    )

    mag = np.abs(S)[:129, :]
    mel = MEL_FILTERBANK @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)


def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, x):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # INT8 quantization
    scale, zero = input_details[0]['quantization']
    x_q = x / scale + zero
    x_q = np.clip(x_q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], [x_q])
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # dequantize
    scale_o, zero_o = output_details[0]['quantization']
    return (out - zero_o) * scale_o


def run_fp_analysis():
    interpreter = load_tflite_model()

    FP_DST.mkdir(parents=True, exist_ok=True)

    fp_list = []

    for wav in sorted(NO_DRONE_SRC.glob("*.wav")):
        feat = extract_features(wav)
        score = predict(interpreter, feat)

        if score >= 0.5:
            fp_list.append((wav.name, score))
            shutil.copy(wav, FP_DST / wav.name)
            print(f"[FP] {wav.name} -> {score:.3f}")

    print("\n=== FP-Analyse abgeschlossen ===")
    print(f"FP-Samples: {len(fp_list)}")

    return fp_list


if __name__ == "__main__":
    run_fp_analysis()

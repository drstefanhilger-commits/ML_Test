import sys
from joblib import load

from app.train_bpp.bpp_dataset import extract_features_from_wav

MODEL_PATH = "./models/bpp/uav_bandpass_model.joblib"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 app/train_bpp/bpp_test.py <wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"[INFO] Teste Datei (BPP): {wav_path}")

    X = extract_features_from_wav(wav_path)

    clf = load(MODEL_PATH)
    prob = clf.predict_proba([X])[0][1]
    pred = int(prob >= 0.5)

    print("=== BPP TEST RESULT ===")
    print(f"Predicted class: {pred}")
    print(f"Probability (Drone): {prob:.3f}")


if __name__ == "__main__":
    main()

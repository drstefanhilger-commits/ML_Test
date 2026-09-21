import numpy as np
import scipy.signal as sig

SR = 48000

# UAV-Band (Beispiel): 200 Hz – 3000 Hz
BP_LOW = 200.0
BP_HIGH = 3000.0
BP_ORDER = 4


def design_bandpass(sr: int = SR):
    nyq = 0.5 * sr
    low = BP_LOW / nyq
    high = BP_HIGH / nyq
    b, a = sig.butter(BP_ORDER, [low, high], btype="bandpass")
    return b, a


def apply_bandpass(x: np.ndarray, sr: int = SR) -> np.ndarray:
    b, a = design_bandpass(sr)
    y = sig.filtfilt(b, a, x)
    return y

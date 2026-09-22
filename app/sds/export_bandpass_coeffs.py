import scipy.signal as sig
import numpy as np

SR = 48000
BP_LOW = 200.0
BP_HIGH = 3000.0
BP_ORDER = 4

nyq = SR * 0.5
low = BP_LOW / nyq
high = BP_HIGH / nyq

b, a = sig.butter(BP_ORDER, [low, high], btype="bandpass")

print("b =", b)
print("a =", a)

# Optional: Format for C++ array initialization
print("BP_B =", "{", ", ".join(f"{coef:.6f}f" for coef in b), "}")
print("BP_A =", "{", ", ".join(f"{coef:.6f}f" for coef in a), "}")

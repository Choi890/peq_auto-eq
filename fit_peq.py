import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from dsp import load_wav_mono, welch_spectrum_db, normalize_at, peq_sum_response_db

os.makedirs("results", exist_ok=True)

# 1) load and measure
x, fs = load_wav_mono("D:\Code\Codex\data\Licht.flac")
freqs, mag_db = welch_spectrum_db(x, fs, nperseg=8192)
mag_db = normalize_at(freqs, mag_db, f0=1000.0)

# analysis band (avoid extreme low/high bins)
band_mask = (freqs >= 30) & (freqs <= 16000)
f = freqs[band_mask]
y = mag_db[band_mask]

# 2) target = flat 0 dB
target = np.zeros_like(y)
err = y - target  # what we'd like to cancel

# 3) PEQ setup (fixed f0, fixed Q; optimize gains only)
centers = np.array([60, 150, 400, 1000, 3000], dtype=float)
Q = 1.2
gain_limit = 6.0  # dB

def residual(gains):
    bands = [(float(fc), Q, float(g)) for fc, g in zip(centers, gains)]
    eq_db = peq_sum_response_db(f, fs, bands)
    # applying EQ means y + eq_db should approach target(0)
    return (y + eq_db) - target

x0 = np.zeros(len(centers))
res = least_squares(
    residual, x0,
    bounds=(-gain_limit*np.ones_like(x0), gain_limit*np.ones_like(x0))
)
g_opt = res.x

# 4) report
print("=== Recommended PEQ (fixed Q=1.2) ===")
print("Preamp: -3.0 dB (suggested safety headroom)")
for fc, g in zip(centers, g_opt):
    print(f"PEQ: {fc:>6.0f} Hz  Q {Q:.2f}  Gain {g:+.2f} dB")

# 5) plot before/after
bands = [(float(fc), Q, float(g)) for fc, g in zip(centers, g_opt)]
eq_db_full = peq_sum_response_db(f, fs, bands)
after = y + eq_db_full

plt.figure()
plt.semilogx(f, y, label="Measured (norm @1kHz)")
plt.semilogx(f, after, label="After EQ (pred.)")
plt.axhline(0.0)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Before/After (Prediction)")
plt.grid(True, which="both")
plt.legend()
plt.savefig("results/peq_fit.png", dpi=200)
print("Saved: results/peq_fit.png")

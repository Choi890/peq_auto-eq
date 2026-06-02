import os
import matplotlib.pyplot as plt
from dsp import load_wav_mono, welch_spectrum_db, normalize_at

os.makedirs("results", exist_ok=True)

x, sr = load_wav_mono("D:\Code\Codex\data\Licht.flac")
freqs, mag_db = welch_spectrum_db(x, sr, nperseg=8192)
mag_db = normalize_at(freqs, mag_db, f0=1000.0)

plt.figure()
plt.semilogx(freqs[1:], mag_db[1:])  # skip DC
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB, normalized @1kHz)")
plt.title("Average Spectrum (Welch)")
plt.grid(True, which="both")
plt.savefig("results/pink_spectrum.png", dpi=200)
print("Saved: results/pink_spectrum.png")

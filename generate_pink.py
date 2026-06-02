import os
from dsp import pink_noise, save_wav

os.makedirs("data", exist_ok=True)

sr = 48000
seconds = 20
x = pink_noise(sr=sr, seconds=seconds, level_dbfs=-12.0, seed=0)
save_wav("data/pink.wav", x, sr)

print("Saved: data/pink.wav")
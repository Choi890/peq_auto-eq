import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.optimize import least_squares

import subprocess
import tempfile
import shutil

from typing import Optional


from dsp import (
    load_wav_mono, welch_spectrum_db, normalize_at,
    peq_sum_response_db,
    apply_peq_to_audio, limiter_softclip,
    smooth_mag_db_octave, target_tilt_curve
)

def find_ffmpeg() -> Optional[str]:
    # 1) PATH에서 찾기
    p = shutil.which("ffmpeg")
    if p:
        return p

    # 2) winget links(네가 보여준 경로)
    winget_link = os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")
    if os.path.exists(winget_link):
        return winget_link

    # 3) 흔한 설치 위치(혹시 모를 대비)
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return None

def decode_to_temp_wav(input_path: str, target_sr: int = 48000) -> str:
    # Convert compressed inputs through ffmpeg so the analysis and full render use one PCM path.
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg not found for Python. "
            "Either add ffmpeg to PATH or hardcode ffmpeg.exe path."
        )

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="auto_eq_tmp_")
    os.close(fd)

    cmd = [
        ffmpeg_path, "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", "2",
        tmp_wav
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{proc.stderr[:2000]}")
    return tmp_wav

def load_audio(path):
    x, fs = sf.read(path, always_2d=True)
    # keep stereo if exists; also keep float64 for processing
    return x.astype(np.float64), fs

def save_audio(path, x, fs):
    # hard safety: softclip then cast float32
    x = limiter_softclip(x, drive=1.3)
    sf.write(path, x.astype(np.float32), fs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input music file path (FLAC/WAV etc.)")
    p.add_argument("--outdir", default="results", help="Output directory")
    p.add_argument("--preamp", type=float, default=-3.0, help="Preamp dB applied before EQ")
    p.add_argument("--tilt", type=float, default=-1.0, help="Target tilt dB per octave (negative = downward)")
    p.add_argument("--pivot", type=float, default=1000.0, help="Pivot frequency for target curve")
    p.add_argument("--gain_limit", type=float, default=4.0, help="Max abs gain per band (dB)")
    p.add_argument("--q", type=float, default=1.2, help="Fixed Q for all bands")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    in_path = args.input
    tmp_path = None

    try:
        # FLAC is normalized through ffmpeg first to avoid backend differences across platforms.
        # (3번) FLAC이면 무조건 ffmpeg로 임시 WAV 생성 후 처리 (환경 차이 제거)
        if os.path.splitext(in_path)[1].lower() == ".flac":
            tmp_path = decode_to_temp_wav(in_path, target_sr=48000)
            in_path = tmp_path

        # 1) Measure spectrum from mono mixdown (analysis only)
        x_mono, fs = load_wav_mono(in_path)
        freqs, mag_db = welch_spectrum_db(x_mono, fs, nperseg=8192)

        # normalize at pivot for stability
        mag_db = normalize_at(freqs, mag_db, f0=args.pivot)

        # analysis band
        band_mask = (freqs >= 40) & (freqs <= 12000)
        f = freqs[band_mask]
        y = mag_db[band_mask]

        # smooth to avoid overfitting musical content
        y_s = smooth_mag_db_octave(f, y, frac_oct=6)

        # 2) Target curve (gentle tilt)
        target = target_tilt_curve(f, tilt_db_per_oct=args.tilt, pivot_hz=args.pivot)

        # 3) Choose fixed band centers (music-friendly)
        centers = np.array([60, 150, 400, 1000, 3000], dtype=float)
        Q = float(args.q)

        # 4) Fit gains only
        def residual(gains):
            # The optimizer changes band gains while fixed centers/Q keep the EQ musical and stable.
            bands = [(float(fc), Q, float(g)) for fc, g in zip(centers, gains)]
            eq_db = peq_sum_response_db(f, fs, bands)
            after = y_s + eq_db
            return after - target

        x0 = np.zeros(len(centers))
        bounds = (-args.gain_limit*np.ones_like(x0), args.gain_limit*np.ones_like(x0))
        res = least_squares(residual, x0, bounds=bounds)
        g_opt = res.x

        bands = [(float(fc), Q, float(g)) for fc, g in zip(centers, g_opt)]

        # 5) Report
        print("=== Auto EQ for MUSIC (gentle) ===")
        print(f"Input: {args.input}")
        print(f"fs: {fs} Hz")
        print(f"Target: tilt {args.tilt:+.2f} dB/oct, pivot {args.pivot:.0f} Hz")
        print(f"Preamp: {args.preamp:+.2f} dB")
        for fc, g in zip(centers, g_opt):
            print(f"PEQ: {fc:>6.0f} Hz  Q {Q:.2f}  Gain {g:+.2f} dB")

        # 6) Plot (before/after prediction)
        eq_db = peq_sum_response_db(f, fs, bands)
        after = y_s + eq_db

        plt.figure()
        plt.semilogx(f, y, alpha=0.4, label="Measured (raw, norm@pivot)")
        plt.semilogx(f, y_s, label="Measured (smoothed)")
        plt.semilogx(f, target, label="Target (tilt)")
        plt.semilogx(f, after, label="After EQ (pred.)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Auto EQ Fit (Music)")
        plt.grid(True, which="both")
        plt.legend()
        plot_path = os.path.join(args.outdir, "auto_eq_fit_music.png")
        plt.savefig(plot_path, dpi=200)
        print(f"Saved plot: {plot_path}")

        # 7) Apply EQ to original audio (keep stereo)
        # Analysis is mono for robustness, but rendering preserves the original channel layout.
        x_full, fs2 = load_audio(in_path)
        if fs2 != fs:
            raise RuntimeError(f"Sample rate mismatch: analysis fs={fs}, full fs={fs2}")

        y_full = apply_peq_to_audio(x_full, fs, bands, preamp_db=args.preamp)

        base = os.path.splitext(os.path.basename(args.input))[0]
        out_wav = os.path.join(args.outdir, f"{base}_auto_eq.wav")
        save_audio(out_wav, y_full, fs)
        print(f"Saved EQ applied wav: {out_wav}")

    finally:
        # cleanup temp wav
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()

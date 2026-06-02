import numpy as np
import soundfile as sf
from scipy import signal

def save_wav(path, x, sr):
    # prevent clipping
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 1.0:
        x = x / peak
    sf.write(path, x.astype(np.float32), sr)

def load_wav_mono(path):
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)  # mono
    return x.astype(np.float64), sr

def pink_noise(sr=48000, seconds=20, level_dbfs=-12.0, seed=0):
    """
    Frequency-domain pink noise: magnitude proportional to 1/sqrt(f)
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)

    # rfft size n -> bins 0..n/2
    freqs = np.fft.rfftfreq(n, 1/sr)
    # random complex spectrum
    real = rng.standard_normal(len(freqs))
    imag = rng.standard_normal(len(freqs))
    spec = real + 1j * imag

    # 1/sqrt(f) shaping (avoid f=0)
    shaping = np.ones_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    spec *= shaping

    # back to time
    x = np.fft.irfft(spec, n=n)

    # normalize to desired level
    x = x / (np.max(np.abs(x)) + 1e-12)
    amp = 10 ** (level_dbfs / 20.0)
    x *= amp
    return x

def welch_spectrum_db(x, sr, nperseg=8192):
    """
    Returns (freqs, mag_db) using Welch PSD.
    """
    freqs, psd = signal.welch(
        x, fs=sr, window="hann",
        nperseg=nperseg, noverlap=nperseg//2,
        scaling="density"
    )
    mag_db = 10.0 * np.log10(psd + 1e-20)
    return freqs, mag_db

def normalize_at(freqs, mag_db, f0=1000.0):
    """
    Shift so that magnitude at f0 becomes 0 dB (nearest bin).
    """
    idx = np.argmin(np.abs(freqs - f0))
    return mag_db - mag_db[idx]

def peq_peak_mag_db(freqs, f0, Q, gain_db):
    """
    Approximate magnitude response (in dB) of a peak EQ biquad.
    We compute digital biquad freq response via scipy.signal.freqz.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0
    # convert Hz -> normalized rad/sample later; we will design via RBJ cookbook using digital w0
    # Using RBJ formulas with w0 in rad/sample:
    # w0 = 2*pi*f0/fs  (we’ll pass fs later in wrapper)
    raise NotImplementedError("Use peq_sum_response_db wrapper with fs.")
    
def peq_sum_response_db(freqs, fs, bands):
    """
    bands: list of (f0, Q, gain_db)
    Returns total magnitude response in dB across given freqs.
    """
    # Multiply each biquad response so the optimizer sees the same cascade that rendering applies.
    # We'll compute response on a dense grid using freqz at those freqs
    w = 2*np.pi*freqs/fs
    H = np.ones_like(w, dtype=np.complex128)

    for (f0, Q, gain_db) in bands:
        A = 10 ** (gain_db / 40.0)
        w0 = 2*np.pi*f0/fs
        alpha = np.sin(w0)/(2*Q)

        b0 = 1 + alpha*A
        b1 = -2*np.cos(w0)
        b2 = 1 - alpha*A
        a0 = 1 + alpha/A
        a1 = -2*np.cos(w0)
        a2 = 1 - alpha/A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1/a0, a2/a0])

        _, h = signal.freqz(b, a, worN=w)
        H *= h

    mag_db = 20*np.log10(np.abs(H) + 1e-20)
    return mag_db





import numpy as np
from scipy import signal

def design_peaking_eq(fs, f0, Q, gain_db):
    """
    RBJ cookbook peaking EQ biquad.
    Returns (b, a) for lfilter.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cosw0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

def apply_peq_cascade(x, fs, bands):
    """
    bands: list of (f0, Q, gain_db)
    Applies cascade of biquads to mono signal x.
    """
    y = x.astype(np.float64)
    for (f0, Q, gain_db) in bands:
        b, a = design_peaking_eq(fs, f0, Q, gain_db)
        y = signal.lfilter(b, a, y)
    return y

def apply_peq_to_audio(x, fs, bands, preamp_db=-3.0):
    """
    x: mono or stereo numpy array shape (n,) or (n, ch)
    Applies preamp then cascade PEQ.
    Returns float64 audio.
    """
    # Apply preamp before EQ so positive filters have headroom before the final soft limiter.
    pre = 10 ** (preamp_db / 20.0)
    y = x.astype(np.float64) * pre

    if y.ndim == 1:
        y = apply_peq_cascade(y, fs, bands)
    else:
        # apply per channel
        out = []
        for ch in range(y.shape[1]):
            out.append(apply_peq_cascade(y[:, ch], fs, bands))
        y = np.stack(out, axis=1)

    return y

def limiter_softclip(x, drive=1.0):
    """
    Simple safety soft-clip (tanh). Keeps output in [-1, 1] softly.
    """
    return np.tanh(drive * x) / np.tanh(drive)

def smooth_mag_db_octave(freqs, mag_db, frac_oct=6):
    """
    Simple log-frequency moving average smoothing.
    frac_oct=6 means ~1/6 octave smoothing.
    """
    # Log-frequency smoothing prevents the solver from chasing narrow notes instead of mix balance.
    f = freqs.copy()
    y = mag_db.copy()
    out = np.empty_like(y)

    logf = np.log2(np.maximum(f, 1e-9))
    # window size in log2 scale: 1/frac_oct octave
    half_w = 0.5 * (1.0 / frac_oct)

    for i in range(len(f)):
        lo = logf[i] - half_w
        hi = logf[i] + half_w
        m = (logf >= lo) & (logf <= hi)
        out[i] = np.mean(y[m]) if np.any(m) else y[i]
    return out

def target_tilt_curve(freqs, tilt_db_per_oct=-1.0, pivot_hz=1000.0):
    """
    A gentle target curve:
    tilt_db_per_oct = -1.0 means 1 octave up -> -1 dB (downward tilt).
    pivot_hz: where target = 0 dB.
    """
    return tilt_db_per_oct * np.log2(np.maximum(freqs, 1e-9) / pivot_hz)

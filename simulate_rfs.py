"""Simulate complex RFS data and a synchronous ECG on the same time grid.

The simulated RFS signal is the sum of
    a static complex offset,
    a cardiac waveform (volume like, zero at the R peak) along its own IQ direction,
        with an amplitude that is modulated by respiration,
    a respiratory component along a different IQ direction,
    slow drift, 50 Hz powerline interference, complex white noise,
    and a few short motion artifacts.

Ground truth (R peak times, per beat cardiac scale, cardiac direction) is returned
so the pipeline output can be validated.
"""
import numpy as np

# ECG waves relative to the R peak: (time offset [s], amplitude [mV], width [s])
ECG_WAVES = [
    (-0.16, 0.15, 0.025),   # P
    (-0.03, -0.10, 0.008),  # Q
    (0.00, 1.00, 0.010),    # R
    (0.03, -0.25, 0.008),   # S
    (0.28, 0.30, 0.045),    # T
]


def smooth_step(phi, a, b):
    """Cosine ramp from 0 (phi <= a) to 1 (phi >= b)."""
    s = np.clip((phi - a) / (b - a), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def cardiac_template(phi):
    """Complex cardiac waveform over one cycle, phi in [0, 1], zero at phi = 0 and 1.

    Real part: ejection (0.05 to 0.35), rapid filling (0.45 to 0.65), atrial kick (0.85 to 0.97).
    Imaginary part: small quadrature component so the IQ trajectory forms a loop.
    """
    v = (-0.60 * smooth_step(phi, 0.05, 0.35)
         + 0.45 * smooth_step(phi, 0.45, 0.65)
         + 0.15 * smooth_step(phi, 0.85, 0.97))
    return v + 0.25j * np.sin(2 * np.pi * phi)


def breathing_waveform(psi):
    """Asymmetric breathing: peak (end inspiration) at psi = 0, long flat end expiration."""
    return ((1 + np.cos(psi)) / 2) ** 2


def simulate(duration=180.0, fs=500.0, seed=0,
             hr_bpm=65.0, cardiac_amp=2e-3, cardiac_angle=0.4,
             resp_amp=1e-2, resp_angle=2.0, resp_modulation=0.3,
             artifact_times=(40.0, 95.0, 150.0), artifact_amp=3e-2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs

    # Respiration with slowly varying rate around 0.2 Hz
    f_resp = 0.2 + 0.02 * np.sin(2 * np.pi * t / 60)
    psi = 2 * np.pi * np.cumsum(f_resp) / fs
    resp = breathing_waveform(psi)

    # Beat times with respiratory sinus arrhythmia and random RR variation.
    # Start before t = 0 and end after the last sample so every sample lies inside a beat.
    r_times = []
    tb = -2.0
    while tb < duration + 2.0:
        r_times.append(tb)
        rr = 60.0 / hr_bpm - 0.05 * np.interp(tb, t, resp) + 0.02 * rng.standard_normal()
        tb += rr
    r_times = np.array(r_times)

    # Cardiac amplitude per beat: reduced during inspiration
    beat_scale = 1.0 - resp_modulation * np.interp(r_times, t, resp)

    # Cardiac waveform per sample
    idx = np.searchsorted(r_times, t, side="right") - 1
    phi = (t - r_times[idx]) / (r_times[idx + 1] - r_times[idx])
    cardiac = beat_scale[idx] * cardiac_template(phi)

    # ECG: Gaussian P, QRS and T waves per beat, baseline wander and noise
    ecg = np.zeros_like(t)
    for tr in r_times:
        win = (t > tr - 0.5) & (t < tr + 0.6)
        for off, amp, width in ECG_WAVES:
            ecg[win] += amp * np.exp(-0.5 * ((t[win] - tr - off) / width) ** 2)
    ecg += 0.1 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(t.size)

    # Complex RFS signal
    s0 = 0.8 * np.exp(1j * 0.6)
    drift = (3e-3 * (t / duration) * np.exp(1j * 1.0)
             + 1e-3 * np.sin(2 * np.pi * t / 90) * np.exp(1j * 2.5))
    powerline = 4e-4 * np.cos(2 * np.pi * 50 * t + 0.3) * (1 + 0.5j)
    noise = 1.5e-4 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
    artifacts = np.zeros_like(t, dtype=complex)
    for ta in artifact_times:
        artifacts += (artifact_amp * np.exp(1j * rng.uniform(0, 2 * np.pi))
                      * np.exp(-0.5 * ((t - ta) / 0.15) ** 2))

    rfs = (s0
           + cardiac_amp * np.exp(1j * cardiac_angle) * cardiac
           + resp_amp * np.exp(1j * resp_angle) * (resp - resp.mean())
           + drift + powerline + noise + artifacts)

    in_range = (r_times >= 0) & (r_times < duration)
    return {
        "t": t,
        "fs": fs,
        "rfs": rfs,
        "ecg": ecg,
        "true_r_times": r_times[in_range],
        "true_beat_scale": beat_scale[in_range],
        "cardiac_amp": cardiac_amp,
        "cardiac_angle": cardiac_angle,
        "artifact_times": np.array(artifact_times),
    }


if __name__ == "__main__":
    data = simulate()
    np.savez("simulated_rfs.npz", **data)
    print(f"Saved simulated_rfs.npz: {data['t'].size} samples at {data['fs']} Hz, "
          f"{data['true_r_times'].size} beats")

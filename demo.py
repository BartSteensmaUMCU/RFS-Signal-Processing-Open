"""Run the minimal RFS pipeline on simulated data and plot every step.

Usage
    python demo.py                   # simulate data on the fly
    python demo.py my_data.npz       # own data: keys 't', 'rfs' (complex), 'ecg'
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

import rfs_pipeline as rp
from simulate_rfs import simulate, cardiac_template

N_BINS = 3
N_OUT = 100


data = simulate()
truth = data

t, rfs, ecg = data["t"], data["rfs"], data["ecg"]
fs = 1.0 / np.median(np.diff(t))

# Pipeline -------------------------------------------------------------------
cardiac, resp = rp.highpass_filter(rfs, fs)
r_peaks = rp.detect_r_peaks(ecg, fs)
beats = rp.segment_beats(cardiac, r_peaks, n_out=N_OUT)
artifact_mask = rp.detect_artifacts(cardiac, fs)
valid = rp.reject_beats(artifact_mask, r_peaks)
beat_bins, end_exp_bin, breathing, phase = rp.respiratory_bins(
    resp, fs, r_peaks, n_bins=N_BINS, artifact_mask=artifact_mask)
avg, counts = rp.average_beats(beats, beat_bins, valid, N_BINS)
avg = rp.subtract_baseline(avg)

print(f"Detected {len(r_peaks)} R peaks, {len(beats)} beats, {np.sum(~valid)} rejected")
print(f"Beats per respiratory bin: {counts.tolist()} (end expiration bin: {end_exp_bin})")

# Validation against ground truth (simulation only) --------------------------
truth_avg = None
if truth is not None:
    true_r = truth["true_r_times"]
    nearest = np.abs(t[r_peaks][:, None] - true_r[None, :]).argmin(axis=1)
    err_ms = 1e3 * np.abs(t[r_peaks] - true_r[nearest])
    print(f"R peak timing error: median {np.median(err_ms):.1f} ms, max {err_ms.max():.1f} ms")

    art_hit = [artifact_mask[np.argmin(np.abs(t - ta))] for ta in truth["artifact_times"]]
    print(f"Simulated artifacts detected: {sum(art_hit)}/{len(art_hit)}")

    # Expected average per bin: mean true cardiac scale of the included beats x template
    scale = truth["true_beat_scale"][nearest[:-1]]
    direction = truth["cardiac_amp"] * np.exp(1j * truth["cardiac_angle"])
    template = cardiac_template(np.linspace(0, 1, N_OUT))
    truth_avg = np.array([scale[valid & (beat_bins == b)].mean() * direction * template
                          for b in range(N_BINS)])
    for b in range(N_BINS):
        rel = np.linalg.norm(avg[b] - truth_avg[b]) / np.linalg.norm(truth_avg[b])
        print(f"  bin {b}: relative RMS error vs ground truth {100 * rel:.1f} %")

# Figures --------------------------------------------------------------------
colors = plt.cm.viridis(np.linspace(0, 0.85, N_BINS))
fig, ax = plt.subplots(3, 2, figsize=(13, 11))

def shade_artifacts(a):
    edges = np.flatnonzero(np.diff(artifact_mask.astype(int)))
    starts, stops = edges[::2] + 1, edges[1::2] + 1
    for s, e in zip(starts, stops):
        a.axvspan(t[s], t[e], color="r", alpha=0.2, lw=0)

a = ax[0, 0]
a.plot(t, rfs.real - rfs.real.mean(), lw=0.5, label="I")
a.plot(t, rfs.imag - rfs.imag.mean(), lw=0.5, label="Q")
shade_artifacts(a)
a.set(title="Raw RFS (mean removed), artifacts shaded", xlabel="Time (s)")
a.legend(loc="upper right")

a = ax[0, 1]
zoom = t < 10
a.plot(t[zoom], ecg[zoom], lw=0.8, color="k")
pk = r_peaks[r_peaks < zoom.sum()]
a.plot(t[pk], ecg[pk], "ro", label="Detected R peaks")
if truth is not None:
    for tr in truth["true_r_times"][truth["true_r_times"] < 10]:
        a.axvline(tr, color="g", ls=":", lw=1)
a.set(title="ECG with R peaks (dotted: ground truth)", xlabel="Time (s)")
a.legend(loc="upper right")

a = ax[1, 0]
a.plot(t, cardiac.real, lw=0.5, label="I")
a.plot(t, cardiac.imag, lw=0.5, label="Q")
shade_artifacts(a)
a.set(title="High pass filtered (cardiac) RFS", xlabel="Time (s)", ylim=np.array([-1, 1]) * 5e-3)
a.legend(loc="upper right")

a = ax[1, 1]
zoom = t < 30
a.plot(t[zoom], breathing[zoom], color="k", lw=1)
beat_t = t[r_peaks[:-1]]
for b in range(N_BINS):
    sel = (beat_bins == b) & (beat_t < 30)
    a.plot(beat_t[sel], breathing[r_peaks[:-1][sel]], "o", color=colors[b], label=f"bin {b}")
a.set(title="Respiratory PC1 with bin per beat", xlabel="Time (s)")
a.legend(loc="upper right")

a = ax[2, 0]
for b in range(N_BINS):
    a.plot(avg[b].real, avg[b].imag, color=colors[b], label=f"bin {b} (n={counts[b]})")
    if truth_avg is not None:
        a.plot(truth_avg[b].real, truth_avg[b].imag, "--", color=colors[b], lw=1)
a.plot(0, 0, "k+", ms=12)
a.set(title="Averaged beats in IQ plane (dashed: truth)", xlabel="I", ylabel="Q", aspect="equal")
a.legend(loc="best")

a = ax[2, 1]
cycle = np.linspace(0, 1, N_OUT)
for b in range(N_BINS):
    a.plot(cycle, np.abs(avg[b]), color=colors[b], label=f"bin {b}")
    if truth_avg is not None:
        a.plot(cycle, np.abs(truth_avg[b]), "--", color=colors[b], lw=1)
a.set(title="|Averaged beat - value at R peak|", xlabel="Normalized cardiac cycle")
a.legend(loc="best")

fig.tight_layout()
fig.savefig("demo_output.png", dpi=150)
plt.show()

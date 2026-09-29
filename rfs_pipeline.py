"""RFS processing pipeline.

Input: complex RFS signal and ECG sampled on the same time grid.

Steps
    1. highpass_filter      split into cardiac and respiratory components, remove powerline
    2. detect_r_peaks       R peak detection on the ECG (neurokit2)
    3. segment_beats        cut R to R and resample every beat to a fixed number of samples
    4. detect_artifacts     sample mask of motion artifacts; reject_beats flags affected beats
    5. respiratory_bins     PCA + Hilbert phase of the respiratory signal, one bin per beat
    6. average_beats        complex average per respiratory bin
    7. subtract_baseline    subtract the complex value at the R peak from the averaged waveform
"""
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import skew


# 1. Filtering ---------------------------------------------------------------

def _filtfilt_complex(b, a, x):
    """Zero phase filtering of a complex signal (real and imaginary parts separately)."""
    return filtfilt(b, a, x.real) + 1j * filtfilt(b, a, x.imag)


def highpass_filter(x, fs, cutoff=0.6, order=5, resp_cutoff=0.1, powerline=50.0):
    """Split a complex RFS signal into cardiac and respiratory components.

    cardiac     = high pass (cutoff) of the powerline filtered signal
    respiratory = remainder (below cutoff), high passed at resp_cutoff to remove drift

    Set powerline=None to skip the powerline band stop filter.
    """
    x = np.asarray(x, dtype=complex)
    if powerline is not None:
        b, a = butter(3, [powerline - 1, powerline + 1], btype="bandstop", fs=fs)
        x = _filtfilt_complex(b, a, x)

    b, a = butter(order, cutoff, btype="high", fs=fs)
    cardiac = _filtfilt_complex(b, a, x)

    b, a = butter(3, resp_cutoff, btype="high", fs=fs)
    respiratory = _filtfilt_complex(b, a, x - cardiac)
    return cardiac, respiratory


# 2. R peak detection --------------------------------------------------------

def detect_r_peaks(ecg, fs):
    """Return R peak sample indices."""
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, correct_artifacts=True)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


# 3. Beat segmentation -------------------------------------------------------

def _resample_beat(segment, n_out):
    """Linear resampling of a (complex) segment to n_out samples."""
    x_in = np.linspace(0, 1, len(segment))
    x_out = np.linspace(0, 1, n_out)
    return np.interp(x_out, x_in, segment.real) + 1j * np.interp(x_out, x_in, segment.imag)


def segment_beats(x, r_peaks, n_out=100):
    """Cut the signal from each R peak up to and including the next R peak.

    Returns an array of shape (n_beats, n_out) with n_beats = len(r_peaks) - 1.
    Sample 0 corresponds to the R peak, sample n_out - 1 to the next R peak.
    """
    return np.stack([_resample_beat(x[a:b + 1], n_out)
                     for a, b in zip(r_peaks[:-1], r_peaks[1:])])


# 4. Artifact detection ------------------------------------------------------

def detect_artifacts(x, fs, threshold=5.0, buffer_s=1.0):
    """Boolean mask, True where the Hilbert envelope exceeds threshold times its median.
 
    For complex input the envelope is computed from the analytic signals of the
    real and imaginary parts, sqrt(|H(I)|^2 + |H(Q)|^2), which is invariant to a
    rotation of the IQ plane. Every outlier is extended by buffer_s seconds on both sides.
    """
    x = np.asarray(x) - np.mean(x)
    if np.iscomplexobj(x):
        envelope = np.sqrt(np.abs(hilbert(x.real)) ** 2 + np.abs(hilbert(x.imag)) ** 2)
    else:
        envelope = np.abs(hilbert(x))
    outliers = envelope > threshold * np.median(envelope)
    buffer = int(round(buffer_s * fs))
    return np.convolve(outliers.astype(float), np.ones(2 * buffer + 1), mode="same") > 0


def reject_beats(artifact_mask, r_peaks):
    """True for every beat (R to R) that contains no artifact samples."""
    return np.array([not artifact_mask[a:b + 1].any()
                     for a, b in zip(r_peaks[:-1], r_peaks[1:])])


# 5. Respiratory binning -----------------------------------------------------

def pca_2ch(X, artifact_mask=None):
    """PCA on a (2, N) observation matrix.

    artifact_mask: boolean array of shape (N,), True for samples excluded from the fit.

    Returns
    -------
    scores : ndarray, shape (2, N)
        PC1 in row 0, PC2 in row 1 (sorted by descending eigenvalue).
    components : ndarray, shape (2, 2)
        Eigenvectors as rows (sorted by descending eigenvalue).
    center : ndarray, shape (2,)
        Per channel mean of the fitted samples, subtracted before projection.
    """
    X = np.asarray(X, dtype=float)
    X_fit = X if artifact_mask is None else X[:, ~artifact_mask]
    center = X_fit.mean(axis=1)
    eigvals, eigvecs = np.linalg.eigh(np.cov(X_fit))
    components = eigvecs[:, np.argsort(eigvals)[::-1]].T
    scores = components @ (X - center[:, None])
    return scores, components, center


def respiratory_bins(resp, fs, r_peaks, n_bins=3, artifact_mask=None, smooth_s=1.0):
    """Assign every beat to a respiratory phase bin.

    The complex respiratory signal is projected on its first principal component,
    smoothed, clipped to its 5th and 95th percentiles, and oriented so that
    inspiration is a positive peak. Hilbert phase 0 is then end inspiration and
    phase pi is end expiration. Each beat gets the bin of the phase at its R peak.

    Returns
    -------
    beat_bins : ndarray of int, shape (len(r_peaks) - 1,)
    end_exp_bin : int, the bin whose centre is closest to pi
    breathing : ndarray, the real respiratory trace used for the phase
    phase : ndarray, Hilbert phase in [0, 2 pi)
    """
    scores, _, _ = pca_2ch(np.vstack([resp.real, resp.imag]), artifact_mask)
    w = int(round(smooth_s * fs))
    breathing = np.convolve(scores[0], np.ones(w) / w, mode="same")
    breathing = np.clip(breathing, *np.percentile(breathing, [5, 95]))
    if skew(breathing) < 0:
        breathing = -breathing

    phase = np.mod(np.angle(hilbert(breathing - breathing.mean())), 2 * np.pi)
    bin_width = 2 * np.pi / n_bins
    sample_bins = np.minimum((phase // bin_width).astype(int), n_bins - 1)
    centres = bin_width * (np.arange(n_bins) + 0.5)
    end_exp_bin = int(np.argmin(np.abs(centres - np.pi)))
    return sample_bins[r_peaks[:-1]], end_exp_bin, breathing, phase


# 6. Complex averaging -------------------------------------------------------

def average_beats(beats, beat_bins, valid, n_bins):
    """Complex mean beat per respiratory bin, using valid beats only.

    Returns averages of shape (n_bins, n_samples) and the number of beats per bin.
    """
    avg = np.full((n_bins, beats.shape[1]), np.nan, dtype=complex)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = valid & (beat_bins == b)
        counts[b] = sel.sum()
        if counts[b] > 0:
            avg[b] = beats[sel].mean(axis=0)
    return avg, counts


# 7. Baseline subtraction ----------------------------------------------------

def subtract_baseline(avg):
    """Subtract the complex value at the R peak (sample 0) from averaged waveform(s)."""
    return avg - avg[..., :1]
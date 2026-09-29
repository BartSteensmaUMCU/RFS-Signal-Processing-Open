#RFS-Demo
Signal processing pipeline for radio frequency sensing of cardiac motion

This repository contains the signal processing pipeline described in:
Steensma et al. Understanding the physiological basis of cardiac radio frequency sensing . xxx, xxx (2026). https://doi.org/10.xxxx/xxxxx

Note: this repository works with simulated data only. No measured subject data is included. The simulation provides a complex RFS signal and a synchronous ECG with known ground truth, so every step can be run and validated.

FILES

rfs_pipeline.py: processing functions
simulate_rfs.py: generates simulated complex RFS data and ECG
demo.py: runs the pipeline, compares the result with the ground truth and plots each step
requirements.txt: Python dependencies

PIPELINE

Input is a complex RFS signal (real and imaginary parts) and an ECG on the same time grid. In the demo script, simulated RFS and ECG data is used as input. 

1. Filtering: a 50 Hz band stop filter removes powerline interference. A zero phase Butterworth high pass filter (0.6 Hz) separates the cardiac component. The remainder, high pass filtered at 0.1 Hz, forms the respiratory component.
2. R peak detection: R peaks are detected in the ECG with NeuroKit2.
3. Beat segmentation: the cardiac component is cut from R peak to R peak, and each beat is resampled to 100 samples.
4. Artifact removal: samples where the Hilbert envelope exceeds 7 times its median are marked as artifacts, with a 1s buffer on both sides. Beats containing artifacts are rejected.
5. Respiratory binning: the respiratory component is projected on its first principal component in the complex plane, spanned by the real and imaginary parts. Its Hilbert phase divides the breathing cycle into 3 bins, and each beat is assigned to a bin based on the phase at its R peak.
6. Complex averaging: valid beats are averaged per respiratory bin.
7. Baseline subtraction: the complex value at the R peak is subtracted from each averaged waveform.

SIMULATION

simulate_rfs.py generates 180 s of data at 500 Hz. The RFS signal combines a static offset, a cardiac waveform modulated by respiration, a respiratory signal, drift, powerline interference, noise and three motion artifacts. The ECG consists of synthetic P, QRS and T waves at known beat times. This is a simplified signal model, not a physical model of RF interaction with the body.

USAGE

pip install -r requirements.txt
python demo.py

The demo prints a summary and saves demo_output.png. To use your own data, provide a .npz file with the arrays t (time in s), rfs (complex) and ecg, and run:
python demo.py my_data.npz

LIMITATIONS

This code is intended for research and reproducibility only.

CITATION

If you use this code, please cite the paper above.

LICENSE

MIT License, see LICENSE.

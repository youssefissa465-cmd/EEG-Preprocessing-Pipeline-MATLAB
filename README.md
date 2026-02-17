# EEG Preprocessing Pipeline

A modular, research-oriented EEG preprocessing pipeline designed for
Brain–Computer Interface (BCI) and neuro-rehabilitation applications.

This project transforms raw, artifact-contaminated EEG recordings into
clean, normalized signals suitable for signal analysis and machine learning.

-Objective

To implement a four-phase EEG preprocessing framework that:
- Removes physiological and environmental noise
- Preserves neural timing (zero-phase distortion)
- Produces standardized signals for feature extraction and classification


Preprocessing Pipeline

Phase 1 – Signal Standardization
- Common Average Referencing (CAR)
- Reduces global noise shared across channels

Phase 2 – Spectral Cleaning
- High-pass filter: 0.5 Hz (baseline drift removal)
- Low-pass filter: 45 Hz (EMG noise suppression)
- Notch filter: 50 / 60 Hz (power-line interference)
- Implemented using zero-phase `filtfilt`

Phase 3 – Artifact Removal (ICA)
- Independent Component Analysis (ICA)
- Suppression of:
  - Eye blink artifacts (EOG)
  - Muscle activity (EMG)
  - Cardiac interference (ECG)

Phase 4 – Normalization
- Channel-wise Z-score normalization
- Ensures numerical stability for machine learning models

Repository Structure

```text
src/        → Signal processing algorithms  
data/       → EEG datasets (excluded from version control)  
utils/      → Visualization and evaluation tools  
docs/       → Academic report  
results/    → Output figures  

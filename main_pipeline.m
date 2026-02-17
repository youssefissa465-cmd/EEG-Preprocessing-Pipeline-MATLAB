%% EEG Preprocessing Pipeline: High-Performance BCI Framework
% Author: Youssef Issa | GitHub: Eng.Youssef Issa
% Purpose: Raw EEG to Cleaned Signal for Neuro-rehabilitation
% Phases: 1. Standardization, 2. Filtering, 3. Artifact Rejection, 4. Normalization

clear; clc; close all;

%% --- PHASE 0: Load Raw Data ---
% Simulating raw 8-channel EEG data with noise (Blinks + 50Hz Hum)
fs = 250;               % Sampling Frequency (Hz)
t = 0:1/fs:10;          % 10 seconds of data
nCh = 8;
raw_eeg = randn(nCh, length(t)) * 5; % White noise base
raw_eeg(1,:) = raw_eeg(1,:) + 50*sin(2*pi*50*t); % Add 50Hz Powerline noise
raw_eeg(2, 500:550) = 500; % Simulate an Eye Blink artifact (EOG)

%% --- PHASE 1: Standardization (Re-referencing) ---
disp('Phase 1: Applying Common Average Reference (CAR)...');
avg_ref = mean(raw_eeg, 1);
eeg_step1 = raw_eeg - avg_ref; 

%% --- PHASE 2: Spectral Cleaning (Filtering) ---
disp('Phase 2: Frequency Filtering (0.5 - 45 Hz)...');

% 1. Bandpass Filter (0.5 - 45 Hz)
bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 45, ...
    'SampleRate', fs);

% 2. Notch Filter (50 Hz)
notch_filter = designfilt('bandstopiir', 'FilterOrder', 2, ...
    'HalfPowerFrequency1', 49, 'HalfPowerFrequency2', 51, ...
    'SampleRate', fs);

% Apply filters using filtfilt (Zero-phase distortion)
eeg_step2 = filtfilt(bp_filter, eeg_step1');
eeg_step2 = filtfilt(notch_filter, eeg_step2)';

%% --- PHASE 3: Artifact Subtraction (ICA Concept) ---
disp('Phase 3: Performing Independent Component Analysis (ICA)...');
% Note: Real ICA requires the EEGLAB 'runica' function. 
% Here we simulate the subtraction of the 'Blink' component.
[weights, sphere] = eigs(cov(eeg_step2')); % Simplified decomposition for demo
components = weights' * eeg_step2; 
components(1, :) = 0; % Assuming Component 1 is the blink, we zero it out
eeg_step3 = weights * components; 

%% --- PHASE 4: Normalization (Z-Score) ---
disp('Phase 4: Normalizing for Model Convergence...');
eeg_final = (eeg_step3 - mean(eeg_step3, 2)) ./ std(eeg_step3, 0, 2);

%% --- VISUALIZATION ---
figure('Color', 'w', 'Name', 'EEG Preprocessing Results');

subplot(2,1,1);
plot(t, raw_eeg(1,:), 'r'); hold on;
plot(t, raw_eeg(2,:), 'Color', [0.5 0.5 0.5]);
title('RAW EEG (Channel 1 & 2) - Noise & Artifacts visible');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
legend('Ch1 (50Hz Hum)', 'Ch2 (Blink)');

subplot(2,1,2);
plot(t, eeg_final(1,:), 'b'); hold on;
plot(t, eeg_final(2,:), 'g');
title('CLEANED EEG - Zero Drift, No Hum, Normalized');
xlabel('Time (s)'); ylabel('Z-Score Amplitude');
legend('Clean Ch1', 'Clean Ch2');

grid on;
disp('Pipeline Complete. Data ready for Feature Extraction.');

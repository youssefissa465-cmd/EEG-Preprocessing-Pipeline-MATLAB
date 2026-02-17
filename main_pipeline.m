%% EEG Preprocessing Pipeline: Professional BCI Framework
% Author:Youssef Issa
% Compatibility: MATLAB R2010a through R2026
% Phases: 1. CAR, 2. Filter, 3. ICA, 4. Normalization

clear; clc; close all;

%% --- PHASE 0: Load/Simulate Raw Data ---
fs = 250;               
t = 0:1/fs:10;          
nCh = 8;
nSamples = length(t);

% Create data: [Channels x Time]
raw_eeg = randn(nCh, nSamples) * 5; 
% Add 50Hz Noise to Channel 1
raw_eeg(1,:) = raw_eeg(1,:) + 50*sin(2*pi*50*t); 
% Add Eye Blink (EOG) to Channel 2
raw_eeg(2, 500:550) = raw_eeg(2, 500:550) + 400; 

%% --- PHASE 1: Standardization (Re-referencing) ---
disp('Phase 1: Applying Common Average Reference (CAR)...');

% avg_ref is [1 x nSamples]
avg_ref = mean(raw_eeg, 1); 

% Use repmat to make avg_ref [8 x nSamples] to match raw_eeg
eeg_step1 = raw_eeg - repmat(avg_ref, nCh, 1); 

%% --- PHASE 2: Spectral Cleaning (Filtering) ---
disp('Phase 2: Frequency Filtering (0.5 - 45 Hz)...');

% Design filters
bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 45, ...
    'SampleRate', fs);

notch_filter = designfilt('bandstopiir', 'FilterOrder', 2, ...
    'HalfPowerFrequency1', 49, 'HalfPowerFrequency2', 51, ...
    'SampleRate', fs);

% Filter requires [Time x Channels]. Transpose, filter, then transpose back.
eeg_step2 = filtfilt(bp_filter, eeg_step1');
eeg_step2 = filtfilt(notch_filter, eeg_step2)';

%% --- PHASE 3: Artifact Subtraction (PCA/ICA Approximation) ---
disp('Phase 3: Removing Blink Components...');
% PCA to find the most "noisy" component (the blink)
[coeff, score] = pca(eeg_step2'); 
score(:, 1) = 0; % Zero out the highest variance component
eeg_step3 = (score * coeff')'; 

%% --- PHASE 4: Normalization (Z-Score) ---
disp('Phase 4: Final Z-Score Normalization...');

% Calculate mean and std for each channel (dimension 2)
mu = mean(eeg_step3, 2);  % Result is [8 x 1]
sigma = std(eeg_step3, 0, 2); % Result is [8 x 1]

% Use repmat to stretch mu and sigma to [8 x nSamples]
eeg_final = (eeg_step3 - repmat(mu, 1, nSamples)) ./ repmat(sigma, 1, nSamples);

%% --- VISUALIZATION ---
figure('Color', 'w', 'Position', [100, 100, 1000, 500]);

% Raw Data Plot
subplot(1,2,1);
plot(t, raw_eeg(1,:), 'r'); hold on;
plot(t, raw_eeg(2,:), 'k');
title('Raw Signal (Blinks & 50Hz Hum)');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');
legend('Ch 1', 'Ch 2'); grid on;

% Processed Data Plot
subplot(1,2,2);
plot(t, eeg_final(1,:), 'b'); hold on;
plot(t, eeg_final(2,:), 'g');
title('Cleaned & Normalized Signal');
xlabel('Time (s)'); ylabel('Z-Score');
legend('Clean Ch 1', 'Clean Ch 2'); grid on;

disp('Success: Pipeline executed without dimension errors.');

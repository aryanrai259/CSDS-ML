# Product Requirements Document (PRD)
**Project Name**: Smart Stethoscope: AI-Powered Device for Cardiac and Pulmonary Sound Analysis
**Version**: 2.0 (Detailed Lifecycle Specification)

## 1. Executive Summary & Vision
The Smart Stethoscope project bridges the gap between traditional mechanical auscultation and modern AI-driven diagnostics. This document defines a distributed cyber-physical pipeline designed to acquire, digitize, process, and classify cardiopulmonary sounds. This is treated as a multi-disciplinary engineering product, progressing strictly from signal feasibility to embedded edge acquisition, culminating in a web-based diagnostic dashboard.

## 2. Product Lifecycle Flow
The product cycle follows a rigid, phase-gated approach to isolate engineering domains and de-risk the hardware-software integration:
- **Gate 1: Feasibility & DSP Verification (Phase 1)** - Can we process digital audio accurately without hardware variables? 
- **Gate 2: ML Model Validation (Phase 2)** - Can a CNN generalize on standardized acoustic datasets (e.g., PhysioNet)?
- **Gate 3: End-to-End Software Pipeline (Phase 3)** - Can the backend stitch live mock audio to the CNN in real-time?
- **Gate 4: Hardware-in-the-Loop Integration (Phase 4)** - Can embedded sensors replace the mock audio without degrading signal integrity?
- **Gate 5: Productization (Phase 5)** - Is the system robust against disconnects, noise artifacts, and user errors?

## 3. Detailed User Personas
### 3.1. The Clinical Researcher
- **Goal**: Needs to visualize heart/lung sounds via spectrograms to annotate data.
- **Pain Point**: Traditional stethoscopes leave no digital trace.
- **Requirement**: Real-time visualization and recording capabilities.

### 3.2. The Embedded Engineer
- **Goal**: Ensure the acoustic signal is captured without clipping or aliasing.
- **Pain Point**: Arduino UNO has extremely limited memory (2KB RAM) and slow ADC conversion times if not optimized.
- **Requirement**: Lightweight C++ firmware focused *only* on deterministic sampling and serial transmission.

### 3.3. The Machine Learning Engineer
- **Goal**: Train a CNN that generalizes beyond lab environments.
- **Pain Point**: Medical audio datasets are noisy and heterogeneous.
- **Requirement**: A strictly standardized Digital Signal Processing (DSP) pipeline that is identical in both training and live inference.

## 4. Intricate Feature Breakdown
### 4.1. Hardware Acquisition Module
- **Transducer Interface**: Chestpiece mechanically coupled to the MAX9814 microphone.
- **Auto Gain Control (AGC)**: The microphone must dynamically adjust gain to prevent clipping during loud respiratory sounds (coughs) while boosting faint murmurs.
- **Analog-to-Digital Conversion**: Must sample exactly at the Nyquist frequency required for clinical audio (minimum 4000 Hz for human auscultation ranges up to 2000 Hz).

### 4.2. Data Transmission Pipeline
- **Continuous Serial Stream**: The hardware must stream raw integer values over USB.
- **Lossless Buffering**: The host system must queue incoming bytes using a thread-safe ring buffer to prevent data loss during CPU spikes.

### 4.3. Digital Signal Processing (DSP) Engine
- **Noise Suppression**: Algorithmic removal of 50/60Hz AC mains hum and ambient room noise.
- **Segmentation**: Sliding window mechanism. The continuous stream is sliced into 3.0-second discrete "frames" with a 1.0-second overlap (stride) to ensure transient anomalies (like a crackle) are not split across frame boundaries.
- **Feature Extraction**: Transformation of the 1D time-series array into 2D Mel-Frequency Cepstral Coefficients (MFCCs) and Mel-Spectrograms.

### 4.4. AI Inference Engine
- **Model Loader**: Singleton class that loads the `.h5` or `.onnx` model into memory precisely once on server boot.
- **Asynchronous Predictor**: The inference function must not block the serial reading thread. 

### 4.5. Visualization Dashboard
- **Web Interface**: A single-page application (SPA).
- **Live Oscilloscope**: A scrolling waveform viewer mimicking an ECG machine.
- **Live Spectrogram**: A scrolling heatmap displaying frequency intensity over time.
- **Diagnostic Ticker**: A real-time log of the model's confidence scores and predicted class (e.g., "75% Wheeze detected").

## 5. Granular Use Cases (User Journeys)
### Use Case 1: System Boot & Calibration
1. User connects Arduino via USB and launches `python backend/app.py`.
2. Backend initializes ML weights in RAM (approx. 2-5 seconds).
3. Backend scans available COM ports and establishes a connection at 115200 baud.
4. Backend flushes the serial buffer to clear stale data.
5. UI displays a green "Connected" status indicator.

### Use Case 2: Live Diagnostic Session
1. User places the chestpiece on the patient.
2. The UI waveform immediately begins scrolling, displaying the periodic heartbeat.
3. Every 2 seconds, the backend extracts the last 3 seconds of audio, generates an MFCC, and runs a prediction.
4. If a severe abnormality is detected, the UI flashes a visual alert alongside the confidence interval.
5. User clicks "Stop Session", and the raw audio + JSON metadata is saved to `logs/`.

## 6. Non-Functional Requirements (NFRs)
- **Latency**: The time from an acoustic event occurring to the UI displaying the prediction must be `< 1.5 seconds`.
- **Throughput**: The serial connection must sustain `> 8,000 bytes/sec` continuously.
- **Reliability**: If the Arduino is unplugged, the backend must not crash. It must gracefully catch the `SerialException` and display "Hardware Disconnected" on the UI.
- **Reproducibility**: The exact `scipy` / `librosa` versions must be pinned to prevent DSP math differences between training and inference.

## 7. Success Metrics & KPIs
1. **Signal-to-Noise Ratio (SNR)**: > 15 dB achieved after the DSP filtering phase.
2. **Buffer Overruns**: Zero buffer drops over a continuous 10-minute session.
3. **Model Accuracy**: Validation F1-score > 0.85 on holdout datasets (e.g., ICBHI 2017 or PhysioNet).
4. **End-to-End Latency**: Measured latency < 1.5s from transducer to frontend visualization.

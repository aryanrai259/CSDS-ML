import numpy as np
import os

def generate_mock_audio_dataset(num_samples=100, fs=4000, duration=3.0):
    """
    Generates a synthetic dataset of audio waveforms to validate the ML pipeline.
    This simulates loading real WAV files (e.g., from PhysioNet).
    
    Classes:
    0: Normal (Clean low frequency sine waves, simulating normal heartbeat/breathing)
    1: Abnormal (Added high frequency noise and transients, simulating wheezes/murmurs)
    """
    X = []
    y = []
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    for i in range(num_samples):
        # Base normal heartbeat (e.g. 60 BPM -> 1 Hz base frequency)
        # We'll use a mix of low frequencies to simulate "Normal"
        sig = np.sin(2 * np.pi * 1.5 * t) * 0.5 + np.sin(2 * np.pi * 50 * t) * 0.2
        
        # 50% chance to be abnormal
        if i % 2 == 1:
            # Add "abnormal" high frequency components (simulating a wheeze)
            wheeze = np.sin(2 * np.pi * 400 * t) * 0.4
            # Add random transients (simulating crackles)
            crackles = np.random.normal(0, 0.5, len(t)) * (np.random.rand(len(t)) > 0.99)
            sig = sig + wheeze + crackles
            y.append(1)
        else:
            y.append(0)
            
        # Add slight ambient white noise to all samples
        sig += np.random.normal(0, 0.05, len(t))
        X.append(sig)
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X, y = generate_mock_audio_dataset(num_samples=10)
    print(f"Generated {len(X)} samples. Shape: {X.shape}, Labels: {y.shape}")

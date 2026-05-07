import os
import glob
import numpy as np
import librosa
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dsp.filters import apply_bandpass, normalize_signal
from src.dsp.extractors import generate_spectrogram

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def process_and_save_dataset():
    """
    Reads all .wav files from data/raw/normal and data/raw/abnormal,
    runs them through the DSP pipeline, and saves the 2D spectrograms
    as .npy files in data/processed/.
    
    Why we do this: Loading 4GB of raw audio into RAM will crash your computer.
    By converting them to .npy features first, we can train the CNN using 
    batch generators without running out of memory.
    """
    classes = ['normal', 'abnormal']
    
    for cls_idx, cls_name in enumerate(classes):
        input_folder = os.path.join(RAW_DIR, cls_name)
        output_folder = os.path.join(PROCESSED_DIR, cls_name)
        os.makedirs(output_folder, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
        print(f"Found {len(wav_files)} files in {input_folder}")
        
        for i, wav_path in enumerate(wav_files):
            try:
                # 1. Load audio (resample to our exact hardware target: 4000Hz)
                raw_audio, _ = librosa.load(wav_path, sr=4000)
                
                # We need exactly 3 seconds (12000 samples). Truncate or pad.
                if len(raw_audio) > 12000:
                    raw_audio = raw_audio[:12000]
                else:
                    raw_audio = np.pad(raw_audio, (0, 12000 - len(raw_audio)))
                
                # 2. Exact same DSP Pipeline as live inference
                filtered = apply_bandpass(raw_audio, lowcut=20.0, highcut=1800.0, fs=4000)
                normalized = normalize_signal(filtered)
                spec = generate_spectrogram(normalized, fs=4000)
                
                # 3. Save feature to disk
                filename = os.path.basename(wav_path).replace('.wav', '.npy')
                save_path = os.path.join(output_folder, filename)
                np.save(save_path, spec)
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(wav_files)} {cls_name} samples...")
                    
            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")

if __name__ == "__main__":
    print("Starting Offline Dataset Preprocessing...")
    process_and_save_dataset()
    print("Done. Ready for training.")

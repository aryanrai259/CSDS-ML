import os
import glob
import numpy as np
import librosa
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dsp.filters import apply_bandpass, normalize_signal
from src.dsp.extractors import generate_spectrogram

RAW_DIR = os.path.join("data", "raw", "Respiratory_Sound_Database")
AUDIO_DIR = os.path.join(RAW_DIR, "audio_and_txt_files")
CSV_PATH = os.path.join(RAW_DIR, "patient_diagnosis.csv")
PROCESSED_DIR = os.path.join("data", "processed")

def load_patient_diagnoses():
    """Reads the ICBHI CSV file and creates a mapping of Patient_ID -> Diagnosis"""
    diagnosis_map = {}
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Could not find {CSV_PATH}")
        return diagnosis_map
        
    with open(CSV_PATH, 'r') as f:
        # The ICBHI csv format is simply: 101,URTI
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                patient_id = row[0].strip()
                diagnosis = row[1].strip()
                diagnosis_map[patient_id] = diagnosis
    return diagnosis_map

def process_and_save_dataset():
    """
    Parses the ICBHI clinical dataset. Maps patient IDs to Normal/Abnormal labels,
    processes the audio through the DSP pipeline, and saves ML-ready features.
    """
    diagnosis_map = load_patient_diagnoses()
    if not diagnosis_map:
        return
        
    os.makedirs(os.path.join(PROCESSED_DIR, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, 'abnormal'), exist_ok=True)
    
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    print(f"Found {len(wav_files)} clinical audio files. Processing...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            filename = os.path.basename(wav_path)
            # ICBHI filename format: 101_1b1_Al_sc_Meditron.wav
            patient_id = filename.split('_')[0]
            
            # Look up diagnosis
            diagnosis = diagnosis_map.get(patient_id, "Unknown")
            
            # Map clinical diagnosis to binary classification
            if diagnosis == "Healthy":
                category = "normal"
            elif diagnosis == "Unknown":
                continue # Skip files we can't label
            else:
                category = "abnormal" # COPD, URTI, Asthma, etc.
                
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
            save_path = os.path.join(PROCESSED_DIR, category, filename.replace('.wav', '.npy'))
            np.save(save_path, spec)
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(wav_files)} files...")
                
        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")

if __name__ == "__main__":
    print("Starting Clinical Dataset Preprocessing...")
    process_and_save_dataset()
    print("Done. Ready for training.")

import librosa
import numpy as np

def generate_mfcc(data, fs=4000, n_mfcc=13):
    """
    Generates Mel-Frequency Cepstral Coefficients (MFCCs).
    MFCCs compress the audio waveform into a set of acoustic features
    that represent the power spectrum of the sound.
    """
    # Ensure data is float32 for librosa compatibility
    data = data.astype(np.float32)
    # n_fft is the window size for the FFT. 512 at 4kHz is ~128ms.
    # hop_length is the stride. 128 is ~32ms.
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    return mfccs

def generate_spectrogram(data, fs=4000):
    """
    Generates a Mel-scaled spectrogram (time-frequency image).
    This is what the CNN will typically ingest.
    """
    data = data.astype(np.float32)
    S = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=128, n_fft=512, hop_length=128)
    # Convert power to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

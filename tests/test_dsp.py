import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dsp.filters import apply_bandpass, normalize_signal
from src.dsp.extractors import generate_mfcc, generate_spectrogram

@pytest.fixture
def dummy_audio():
    # 3 seconds of a pure 440Hz sine wave at 4000Hz sampling rate
    t = np.linspace(0, 3.0, 12000, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t)
    return sig

def test_normalization(dummy_audio):
    # Add a huge DC offset and amplify
    dirty_sig = (dummy_audio * 5.0) + 10.0
    clean = normalize_signal(dirty_sig)
    
    assert np.isclose(np.mean(clean), 0.0, atol=1e-5), "DC offset not removed"
    assert np.max(np.abs(clean)) <= 1.0, "Amplitude not normalized"

def test_mfcc_shape(dummy_audio):
    mfcc = generate_mfcc(dummy_audio, fs=4000, n_mfcc=13)
    # n_fft=512, hop_length=128 for 12000 samples -> 94 frames
    assert mfcc.shape == (13, 94), f"Unexpected MFCC shape: {mfcc.shape}"

def test_spectrogram_shape(dummy_audio):
    spec = generate_spectrogram(dummy_audio, fs=4000)
    assert spec.shape == (128, 94), f"Unexpected Spectrogram shape: {spec.shape}"

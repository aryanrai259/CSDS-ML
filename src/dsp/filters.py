import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Calculates Butterworth filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=20.0, highcut=1800.0, fs=4000, order=5):
    """
    Applies a bandpass filter to the audio data.
    Typical heart sounds are 20-150Hz. Lung sounds are 100-2000Hz.
    (Note: highcut must be strictly less than the Nyquist frequency, fs/2)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_signal(data):
    """Removes DC offset and normalizes amplitude between -1.0 and 1.0."""
    # DC offset removal
    data = data - np.mean(data)
    # Amplitude normalization (Prevent amplifying pure background noise)
    max_val = np.max(np.abs(data))
    
    # If the room is completely silent, max_val is tiny (e.g. 0.001).
    # If we divide by 0.001, silence becomes giant spikes.
    # We set a noise floor threshold of 0.05.
    noise_floor = 0.05
    scaling_factor = max(max_val, noise_floor)
    
    data = data / scaling_factor
    return data

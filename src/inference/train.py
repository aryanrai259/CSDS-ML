import os
import sys
import numpy as np

# Suppress TensorFlow logging to keep output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split

# Add src to python path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.dataset import generate_mock_audio_dataset
from src.dsp.filters import apply_bandpass, normalize_signal
from src.dsp.extractors import generate_spectrogram

def preprocess_dataset(X_raw):
    """
    Passes the raw audio through the exact same DSP pipeline used in Phase 1.
    This guarantees that the ML model is trained on the same math it will see in live inference.
    """
    X_processed = []
    print(f"Preprocessing {len(X_raw)} audio samples...")
    for i, audio in enumerate(X_raw):
        # 1. Bandpass filter
        filtered = apply_bandpass(audio, lowcut=20.0, highcut=1800.0, fs=4000)
        # 2. Normalize
        normalized = normalize_signal(filtered)
        # 3. Generate Spectrogram
        spec = generate_spectrogram(normalized, fs=4000)
        
        # Expand dims to add the channel dimension required by CNNs: (Height, Width) -> (Height, Width, 1)
        spec = np.expand_dims(spec, axis=-1)
        X_processed.append(spec)
        
    return np.array(X_processed)

def build_cnn(input_shape):
    """Builds a lightweight CNN architecture suitable for spectrogram classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Binary classification (Normal vs Abnormal)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("--- Phase 2: Offline ML Verification ---")
    
    # 1. Load Data
    print("Generating synthetic audio dataset...")
    X_raw, y = generate_mock_audio_dataset(num_samples=200, fs=4000, duration=3.0)
    
    # 2. Preprocess Data (Enforcing DSP consistency)
    X_features = preprocess_dataset(X_raw)
    print(f"Feature shape after DSP: {X_features.shape}") # Should be (200, 128, 94, 1)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # 4. Build Model
    input_shape = X_train.shape[1:]
    model = build_cnn(input_shape)
    model.summary()
    
    # 5. Train Model
    print("Starting training loop...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 6. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nBaseline Validation Accuracy: {accuracy * 100:.2f}%")
    
    # 7. Save Artifact
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'classifier.h5')
    model.save(model_path)
    print(f"Model saved successfully to: {model_path}")

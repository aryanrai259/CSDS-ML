import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROCESSED_DIR = os.path.join("data", "processed")

class AudioDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Data Generator. Instead of loading 4GB of audio into RAM,
    this reads the pre-processed .npy Spectrogram files batch-by-batch
    during training. This prevents Out-Of-Memory (OOM) crashes.
    """
    def __init__(self, file_paths, labels, batch_size=32):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load the .npy files and add the CNN channel dimension
        batch_x = []
        for p in batch_x_paths:
            spec = np.load(p)
            spec = np.expand_dims(spec, axis=-1)
            batch_x.append(spec)

        return np.array(batch_x), np.array(batch_y)

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
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("--- Phase 2: Offline ML Training (Disk-Backed) ---")
    
    # 1. Gather all file paths
    normal_files = glob.glob(os.path.join(PROCESSED_DIR, 'normal', '*.npy'))
    abnormal_files = glob.glob(os.path.join(PROCESSED_DIR, 'abnormal', '*.npy'))
    
    all_files = normal_files + abnormal_files
    all_labels = [0]*len(normal_files) + [1]*len(abnormal_files)
    
    if len(all_files) == 0:
        print("ERROR: No .npy files found. Did you run prepare_data.py first?")
        exit(1)
        
    print(f"Found {len(all_files)} total preprocessed samples.")
    
    # 2. Train/Test Split
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # 3. Create Generators
    train_gen = AudioDataGenerator(X_train_paths, y_train, batch_size=32)
    test_gen = AudioDataGenerator(X_test_paths, y_test, batch_size=32)
    
    # 4. Build Model (Determine shape from first file)
    sample_shape = np.load(X_train_paths[0]).shape + (1,)
    model = build_cnn(sample_shape)
    model.summary()
    
    # 5. Train Model
    print("Starting training loop...")
    model.fit(
        train_gen,
        epochs=10,
        validation_data=test_gen,
        verbose=1
    )
    
    # 6. Save Artifact
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'classifier.h5')
    model.save(model_path)
    print(f"Model saved successfully to: {model_path}")

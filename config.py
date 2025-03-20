import os

def set_dataset_paths():
    base_path = os.path.dirname(os.path.abspath(__file__))  # Path lokasi file config.py
    train_dataset_path = os.path.join(base_path, "TRAINING")
    test_dataset_path = os.path.join(base_path, "TESTING")
    
    # Cek apakah path ada
    if not os.path.isdir(train_dataset_path):
        raise FileNotFoundError(f"Training path not found: {train_dataset_path}")
    if not os.path.isdir(test_dataset_path):
        raise FileNotFoundError(f"Testing path not found: {test_dataset_path}")
    
    return train_dataset_path, test_dataset_path

# Hyperparameter
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 20
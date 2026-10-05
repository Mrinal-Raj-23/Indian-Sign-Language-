"""
Configuration file for Indian Sign Language (ISL) detection project.
Contains path definitions, hyperparameters, and model configurations.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Path aliases (for backward compatibility across modules)
RAW_DATA_PATH = str(DATA_RAW_DIR)
PROCESSED_DATA_PATH = str(DATA_PROCESSED_DIR)
MODELS_PATH = str(MODELS_DIR)

# Auto-create directories on import
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Gesture Classes
STATIC_CLASSES = [
    'Hello', 'Namaste', 'India', 'Language', 'Bye',
    'Thank You', 'Welcome', 'Please', 'Sorry', 'Practice'
]
DYNAMIC_CLASSES = [
    'Bye', 'Food', 'Hello', 'Help', 'Namaste', 'Sorry', 'Thank You', 'Want', 'Water'
]

# Landmark & Feature Constants
NUM_HAND_LANDMARKS = 21
NUM_COORDS = 3  # (x, y, z)
NUM_FEATURES_PER_HAND = NUM_HAND_LANDMARKS * NUM_COORDS  # 63
NUM_FEATURES_BOTH_HANDS = NUM_FEATURES_PER_HAND * 2     # 126

# Data Collection Config
SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS = 500
SEQUENCES_PER_CLASS = 50

# Data Collection aliases
NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
NUM_SEQUENCES = SEQUENCES_PER_CLASS

# MediaPipe Config
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Training Config
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Inference Config
PREDICTION_THRESHOLD = 0.6
STABILITY_FRAMES = 5

# Gesture Lifecycle & Debounce Config (Phase 1)
GESTURE_RESET_FRAMES = 8          # Number of consecutive neutral frames to conclude a gesture stroke
REST_POSITION_Y_THRESHOLD = 0.85   # Normalized Y coordinate threshold (wrist Y > 0.85 is considered resting)

# Personalization Config
PERSONAL_SAMPLES_PER_CLASS = 25
PERSONAL_DATA_PATH = str(BASE_DIR / "data" / "personal")
PERSONAL_CLASSES = [
    'Hello', 'Namaste', 'I', 'You', 'Bye', 'Thank You', 'Sorry'
]

# Text-to-Speech (TTS) Config
TTS_RATE = 150
TTS_VOLUME = 1.0


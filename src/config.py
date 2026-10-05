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

# ── Dynamic Gesture Classes (Preserved Untouched) ──────────────────────────
DYNAMIC_CLASSES = [
    'Bye', 'Food', 'Hello', 'Help', 'Namaste', 'Sorry', 'Thank You', 'Want', 'Water'
]

# ── 32 Static Classes (6 Words + 26 Alphabets) with Unique Internal Identifiers ─
WORD_CLASSES = [
    'WORD_HELLO',
    'WORD_NAMASTE',
    'WORD_I',
    'WORD_BYE',
    'WORD_SORRY',
    'WORD_THANK_YOU'
]

ALPHABET_CLASSES = [
    f'LETTER_{chr(c)}' for c in range(ord('A'), ord('Z') + 1)
]

# Exact 32 classes
STATIC_CLASSES = WORD_CLASSES + ALPHABET_CLASSES

# Mapping from internal identifier to user-facing display / text representation
DISPLAY_NAME_MAP = {
    'WORD_HELLO': 'Hello',
    'WORD_NAMASTE': 'Namaste',
    'WORD_I': 'I',
    'WORD_BYE': 'Bye',
    'WORD_SORRY': 'Sorry',
    'WORD_THANK_YOU': 'Thank You',
}
for c in range(ord('A'), ord('Z') + 1):
    char = chr(c)
    DISPLAY_NAME_MAP[f'LETTER_{char}'] = char

# Quick lookup sets
ALPHABET_CLASS_SET = set(ALPHABET_CLASSES)
WORD_CLASS_SET = set(WORD_CLASSES)

# Landmark & Feature Constants
NUM_HAND_LANDMARKS = 21
NUM_COORDS = 3  # (x, y, z)
NUM_FEATURES_PER_HAND = NUM_HAND_LANDMARKS * NUM_COORDS  # 63
NUM_FEATURES_BOTH_HANDS = NUM_FEATURES_PER_HAND * 2     # 126

# Data Collection Config
SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS = 200          # default for static classes
ALPHABET_SAMPLES_PER_CLASS = 200 # target samples for each alphabet
NUM_SAMPLES_STATIC = SAMPLES_PER_CLASS
SEQUENCES_PER_CLASS = 50
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

# Gesture Lifecycle & Debounce Config
GESTURE_RESET_FRAMES = 3          # Consecutive neutral frames to conclude a gesture stroke
REST_POSITION_Y_THRESHOLD = 0.85  # Normalized Y threshold (wrist Y > 0.85 = resting)

# Fingerspelling / Letter-buffering Config
LETTER_STABILITY_FRAMES = 10
LETTER_POSE_CHANGE_THRESHOLD = 0.10  # Mean normalized landmark displacement required for a new letter
INTER_LETTER_GAP_SECONDS = 0.8   # max silence between letters in the same word
WORD_END_PAUSE_SECONDS   = 2.0   # silence after last letter that finalizes the word

# Personalization Config
PERSONAL_SAMPLES_PER_CLASS = 25
PERSONAL_DATA_PATH = str(BASE_DIR / "data" / "personal")
PERSONAL_CLASSES = [
    'Hello', 'Namaste', 'I', 'You', 'Bye', 'Thank You', 'Sorry'
]

# Text-to-Speech (TTS) Config
TTS_RATE = 150
TTS_VOLUME = 1.0

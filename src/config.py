"""
Central configuration for the Car Damage Assessment project.
Edit these values to match your dataset and hardware.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "car_damage_model.keras")
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# ---------------------------------------------------------------------
# Class labels
# Folder names inside data/train, data/val, data/test must match these
# exactly (Keras infers labels from subfolder names).
# ---------------------------------------------------------------------
CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------
# Image / model settings
# ---------------------------------------------------------------------
IMG_SIZE = (224, 224)          # required input size for EfficientNetB0
BATCH_SIZE = 32
SEED = 42

# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------
INITIAL_EPOCHS = 15            # train the classification head with base frozen
FINE_TUNE_EPOCHS = 10          # additional epochs after unfreezing top layers
FINE_TUNE_AT_LAYER = 100       # unfreeze layers from this index onward
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

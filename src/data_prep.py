"""
Data loading utilities for the Car Damage Assessment project.

Expected folder layout:

    data/
        train/
            minor/      *.jpg
            moderate/   *.jpg
            severe/     *.jpg
        val/
            minor/ moderate/ severe/
        test/
            minor/ moderate/ severe/

Dataset suggestion: search Kaggle for "car damage severity" or
"vehicle damage detection" datasets, then sort images into the
folders above (roughly 70/15/15 train/val/test split).
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES


def _make_dataset(directory, shuffle):
    """Load images from a directory into a tf.data.Dataset."""
    return image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def _directory_has_images(directory):
    if not tf.io.gfile.exists(directory):
        return False
    for _, _, files in tf.io.gfile.walk(directory):
        if any(file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")) for file in files):
            return True
    return False


def get_datasets():
    """
    Returns (train_ds, val_ds, test_ds) as tf.data.Dataset objects,
    already batched, cached, and prefetched.
    """
    train_ds = _make_dataset(TRAIN_DIR, shuffle=True)
    val_ds = _make_dataset(VAL_DIR, shuffle=False)
    test_ds = _make_dataset(TEST_DIR, shuffle=False) if _directory_has_images(TEST_DIR) else None

    # Light augmentation applied only to the training set to reduce
    # overfitting on a relatively small dataset.
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print(f"Classes: {CLASS_NAMES}")
    print(f"Train batches: {len(train_ds)}")
    print(f"Val batches:   {len(val_ds)}")
    print(f"Test batches:  {len(test_ds)}")

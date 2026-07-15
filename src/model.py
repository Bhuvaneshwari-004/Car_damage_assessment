"""
Model definition: EfficientNetB0 backbone (ImageNet weights) with a
custom classification head for car damage severity.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE, NUM_CLASSES, FINE_TUNE_AT_LAYER


def build_model():
    """
    Builds a transfer-learning model:
      - EfficientNetB0 base (frozen initially)
      - Global average pooling
      - Dropout for regularization
      - Dense softmax head over NUM_CLASSES

    Returns (model, base_model) so the caller can later unfreeze
    layers in base_model for fine-tuning.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
    )
    base_model.trainable = False  # freeze for initial training phase

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    # EfficientNet expects inputs preprocessed with its own scheme
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="car_damage_efficientnet")
    return model, base_model


def unfreeze_for_fine_tuning(base_model):
    """
    Unfreezes layers from FINE_TUNE_AT_LAYER onward so the top of the
    backbone can adapt to car-damage-specific features. Batch-norm
    layers are kept frozen to preserve stable statistics.
    """
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    for layer in base_model.layers[FINE_TUNE_AT_LAYER:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return base_model


if __name__ == "__main__":
    model, base = build_model()
    model.summary()

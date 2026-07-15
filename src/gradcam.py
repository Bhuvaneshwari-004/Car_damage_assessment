"""
Grad-CAM implementation for explainable predictions.

Produces a heatmap highlighting the image regions that most
influenced the model's damage-severity prediction, and overlays it
on the original image for stakeholder-friendly visualization.
"""

import numpy as np
import cv2
import tensorflow as tf


def find_last_conv_layer(base_model):
    """Auto-detects the name of the last 4D (conv-like) output layer
    inside the EfficientNet base model."""
    for layer in reversed(base_model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = getattr(layer.output, "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found in the base model.")


def make_gradcam_heatmap(img_array, model, base_model_layer_name="efficientnetb0",
                          last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3), raw pixel values
               (0-255) — preprocessing is handled inside the model graph.
    model: the full trained Keras model (input -> preprocess -> base -> head)
    base_model_layer_name: name of the base model layer inside `model`
                            (as created in model.py, this is "efficientnetb0"
                            by default — check with model.summary() if unsure)
    last_conv_layer_name: name of the last conv layer inside the base model.
                           If None, it is auto-detected.

    Returns (heatmap, predicted_class_index).
    """
    base_model = model.get_layer(base_model_layer_name)

    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(img_array)
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        conv_output = base_model(x, training=False)

        preds = conv_output
        base_model_index = model.layers.index(base_model)
        for layer in model.layers[base_model_index + 1:]:
            if isinstance(layer, tf.keras.layers.Dropout):
                preds = layer(preds, training=False)
            else:
                preds = layer(preds)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    original_img: numpy array (H, W, 3), values 0-255, uint8, RGB
    heatmap: 2D array, values 0-1
    Returns the overlaid image as a uint8 numpy array (RGB).
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = heatmap_color * alpha + original_img * (1 - alpha)
    return np.uint8(overlaid)

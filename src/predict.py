"""
Run inference on a single image and save a Grad-CAM overlay showing
which regions of the car the model focused on for its prediction.

Usage:
    python src/predict.py path/to/car_image.jpg
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES, OUTPUT_DIR
from gradcam import make_gradcam_heatmap, overlay_heatmap


def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    return img_array


def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    img_array = load_image(image_path)
    batch = np.expand_dims(img_array, axis=0).astype("float32")  # raw 0-255

    preds = model.predict(batch, verbose=0)[0]
    pred_index = int(np.argmax(preds))
    confidence = float(preds[pred_index])

    print(f"\nPrediction: {CLASS_NAMES[pred_index]}  (confidence: {confidence:.2%})")
    for cls, prob in zip(CLASS_NAMES, preds):
        print(f"  {cls:10s}: {prob:.2%}")

    heatmap, _ = make_gradcam_heatmap(batch, model, pred_index=pred_index)
    overlaid = overlay_heatmap(img_array, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlaid)
    axes[1].set_title(f"Grad-CAM: {CLASS_NAMES[pred_index]} ({confidence:.1%})")
    axes[1].axis("off")
    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR, f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved Grad-CAM visualization to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])

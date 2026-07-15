"""
Training script for the Car Damage Assessment model.

Run:
    python src/train.py

Two-phase training:
  Phase 1: train only the classification head (backbone frozen)
  Phase 2: unfreeze top backbone layers and fine-tune at a low LR

Outputs:
  outputs/car_damage_model.keras   - trained model
  outputs/training_history.png     - accuracy/loss curves
  outputs/confusion_matrix.png     - test-set confusion matrix
    outputs/confusion_matrix_normalized.png - row-normalized confusion matrix
    outputs/confusion_matrix_class_accuracy.png - per-class accuracy bar chart
    outputs/confusion_matrix_classification_report.txt - precision/recall/F1 report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    OUTPUT_DIR, MODEL_PATH, HISTORY_PLOT_PATH, CONFUSION_MATRIX_PATH,
    CLASS_NAMES, INITIAL_EPOCHS, FINE_TUNE_EPOCHS, LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
)
from data_prep import get_datasets
from model import build_model, unfreeze_for_fine_tuning


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
    ]


def plot_history(history_1, history_2, save_path):
    acc = history_1.history["accuracy"] + history_2.history["accuracy"]
    val_acc = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]
    loss = history_1.history["loss"] + history_2.history["loss"]
    val_loss = history_1.history["val_loss"] + history_2.history["val_loss"]
    fine_tune_start = len(history_1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(fine_tune_start, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(fine_tune_start, color="gray", linestyle="--", label="fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def evaluate_and_plot_confusion(model, eval_ds, save_path, split_name="test"):
    y_true, y_pred = [], []
    for images, labels in eval_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(f"\nClassification Report ({split_name} split):\n")
    print(report)

    report_path = save_path.replace(".png", "_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification report ({split_name} split)\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

    # Row-normalized confusion matrix (each row sums to 1)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype("float32") / row_sums
    norm_path = save_path.replace(".png", "_normalized.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Oranges", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Normalized Confusion Matrix ({split_name})")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=150)
    plt.close()
    print(f"Saved normalized confusion matrix to {norm_path}")

    # Per-class accuracy from the normalized diagonal.
    class_acc = np.diag(cm_norm)
    class_acc_path = save_path.replace(".png", "_class_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, class_acc, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy ({split_name})")
    for bar, val in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(class_acc_path, dpi=150)
    plt.close()
    print(f"Saved per-class accuracy plot to {class_acc_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets()

    print("Building model...")
    model, base_model = build_model()
    compile_model(model, LEARNING_RATE)

    print("\n--- Phase 1: training classification head (backbone frozen) ---")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=get_callbacks(MODEL_PATH),
    )

    print("\n--- Phase 2: fine-tuning top backbone layers ---")
    unfreeze_for_fine_tuning(base_model)
    compile_model(model, FINE_TUNE_LEARNING_RATE)

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_1.epoch[-1] + 1,
        callbacks=get_callbacks(MODEL_PATH),
    )

    model.save(MODEL_PATH)
    print(f"\nFinal model saved to {MODEL_PATH}")

    plot_history(history_1, history_2, HISTORY_PLOT_PATH)

    print("\n--- Evaluating model ---")
    if test_ds is None:
        print("data/test is empty; evaluating on validation split instead.")
        eval_ds = val_ds
        eval_name = "validation"
    else:
        eval_ds = test_ds
        eval_name = "test"

    eval_loss, eval_acc = model.evaluate(eval_ds)
    print(f"{eval_name.title()} accuracy: {eval_acc:.4f} | {eval_name.title()} loss: {eval_loss:.4f}")

    evaluate_and_plot_confusion(model, eval_ds, CONFUSION_MATRIX_PATH, split_name=eval_name)


if __name__ == "__main__":
    main()

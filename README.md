# Car Damage Assessment — CNN + Transfer Learning + Explainable AI

Image classification project that detects and categorizes vehicle
damage severity (minor / moderate / severe) using transfer learning
on EfficientNetB0, with Grad-CAM heatmaps for explainability.

## 1. Setup

```bash
cd car_damage_assessment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

You need car images sorted into damage-severity folders. Options:

- Kaggle: search **"car damage severity dataset"** or **"vehicle
  damage detection"** — several public datasets exist with
  minor/moderate/severe or similar labels.
- Kaggle: `anujms/car-damage-detection` (binary damaged/whole — good
  starting point if you'd rather do 2-class instead of 3-class;
  just edit `CLASS_NAMES` in `src/config.py`).
- Build your own small set by scraping/labeling a few hundred images
  per class — transfer learning works reasonably well even with
  ~150-300 images/class.

Arrange the images like this:

```
data/
  train/
    01-minor/      *.jpg
    02-moderate/   *.jpg
    03-severe/     *.jpg
  val/
    01-minor/ 02-moderate/ 03-severe/
  test/
    01-minor/ 02-moderate/ 03-severe/
```

Aim for roughly a 70 / 15 / 15 split between train/val/test.

If your classes are different (e.g. just "damaged" / "whole"),
update `CLASS_NAMES` in `src/config.py` to match your folder names —
everything else adapts automatically.

If `data/test/` is empty, training will still run, but the final test-set
evaluation and confusion matrix step will be skipped until you add test images.

## 3. Train

```bash
python src/train.py
```

This runs two-phase transfer learning:
1. Trains a new classification head with the EfficientNetB0 backbone
   frozen.
2. Unfreezes the top backbone layers and fine-tunes at a low
   learning rate.

Outputs land in `outputs/`:
- `car_damage_model.keras` — the trained model
- `training_history.png` — accuracy/loss curves
- `confusion_matrix.png` — test-set performance breakdown

## 4. Predict on a new image (with Grad-CAM)

```bash
python src/predict.py path/to/some_car_photo.jpg
```

Prints the predicted class + confidence, and saves a side-by-side
original vs. heatmap-overlay image to `outputs/gradcam_<name>.png` —
this is the "transparent, auditable model output" piece for your
resume bullet.

## Project structure

```
car_damage_assessment/
├── data/                  # your images go here (not included)
├── outputs/                # trained model + generated plots
├── requirements.txt
└── src/
    ├── config.py           # all settings in one place
    ├── data_prep.py         # dataset loading + augmentation
    ├── model.py              # EfficientNetB0 transfer-learning model
    ├── train.py                # two-phase training + evaluation
    ├── gradcam.py                # Grad-CAM heatmap generation
    └── predict.py                 # single-image inference + visualization
```

## Notes / things you can tune

- `src/config.py` — image size, batch size, epochs, learning rates,
  which backbone layers to unfreeze.
- Swap `EfficientNetB0` for `ResNet50` or `MobileNetV2` in
  `model.py` if you want to compare backbones (useful talking point
  in interviews).
- If accuracy plateaus, try: more training images, stronger
  augmentation, or a lower fine-tuning learning rate.

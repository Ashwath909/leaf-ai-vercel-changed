"""Evaluate a trained model on a separate test directory.

Expected structure: test_data/<class_name>/*.jpg
The class folder names must match the training class names.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "leaf_lens_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.txt"
TEST_DIR = ROOT / "test_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Trained model missing. Run python train.py first.")
    if not TEST_DIR.exists():
        raise SystemExit("Create test_data/<class_name>/ folders with held-out images.")
    model = tf.keras.models.load_model(MODEL_PATH)
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    class_names = CLASS_PATH.read_text(encoding="utf-8").splitlines()
    if ds.class_names != class_names:
        print("Warning: test folder class order/names differ from the training class list.")
        print("Training classes:", class_names)
        print("Test classes:", ds.class_names)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probabilities = model.predict(ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    print(classification_report(y_true, y_pred, target_names=ds.class_names, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()

"""Train a transfer-learning image classifier from folders in dataset/.

Expected structure:
dataset/
  Apple___healthy/
    image1.jpg
  Apple___scab/
    image2.jpg
  Tomato___healthy/
    image3.jpg

Each immediate subfolder is treated as a class label. Use a balanced, legally
obtained dataset and keep the test images separate when reporting final results.
"""
from pathlib import Path
import json
import tensorflow as tf

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 12

def main():
    if not DATA_DIR.exists() or not any(p.is_dir() for p in DATA_DIR.iterdir()):
        raise SystemExit(
            "Dataset folders were not found. Create dataset/<class_name>/image.jpg folders. "
            "See README.md for instructions."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    if num_classes < 2:
        raise SystemExit("At least two class folders are required for classification.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    # MobileNetV2 expects pixels in [-1, 1].
    x = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "leaf_lens_model.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "training_log.csv"))
    ]
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"Classes ({num_classes}): {class_names}")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    (MODEL_DIR / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    summary = {
        "classes": class_names,
        "epochs_run": len(history.history.get("loss", [])),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
    }
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Training complete. Start the app with: streamlit run app.py")

if __name__ == "__main__":
    main()

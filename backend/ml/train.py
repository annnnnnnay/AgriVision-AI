"""
AgriVision AI — Model Training Script

Trains a MobileNetV2-based crop disease classifier using transfer learning
on the PlantVillage dataset.

Usage:
    python ml/train.py --data-dir ./datasets/plantvillage --epochs 25
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
# Match the actual trained model input size to avoid mis-sizing and poor accuracy.
IMAGE_SIZE = 160
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_data_augmentation() -> keras.Sequential:
    """Build data augmentation pipeline for training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int) -> Tuple[keras.Model, keras.Model]:
    """
    Build the transfer learning model.

    Architecture:
    - MobileNetV2 base (frozen → partially unfrozen for fine-tuning)
    - Global Average Pooling
    - Dense + BatchNorm + Dropout classifier head
    """
    # Input
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Data augmentation
    augmentation = build_data_augmentation()
    x = augmentation(inputs)

    # Preprocessing for MobileNetV2
    x = layers.Rescaling(1.0 / 255)(x)  # Already normalized, but ensure

    # MobileNetV2 base
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model initially
    base_model.trainable = False
    x = base_model(x, training=False)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="AgriVision_MobileNetV2")

    return model, base_model


def load_dataset(data_dir: str, validation_split: float = 0.2):
    """Load and prepare training and validation datasets."""
    logger.info(f"Loading dataset from: {data_dir}")

    # Load training set
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    # Load validation set
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    logger.info(f"Found {num_classes} classes: {class_names[:5]}...")

    # Performance optimization
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def train(
    data_dir: str,
    epochs: int = 25,
    fine_tune_epochs: int = 10,
    learning_rate: float = 1e-3,
    fine_tune_lr: float = 1e-5,
    output_dir: str = "ml/models",
    fine_tune_layers: int = 20,
):
    """Full training pipeline."""
    # --- Load data ---
    train_ds, val_ds, class_names, num_classes = load_dataset(data_dir)

    # --- Build model ---
    logger.info("Building model...")
    model, base_model = build_model(num_classes)
    if not 1 <= fine_tune_layers <= len(base_model.layers):
        raise ValueError(
            f"fine_tune_layers must be between 1 and {len(base_model.layers)}."
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Train classifier head (frozen base) ---
    logger.info("Phase 1: Training classifier head (base frozen)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # --- Phase 2: Fine-tune top layers of base model ---
    logger.info(
        f"Phase 2: Fine-tuning last {fine_tune_layers} base-model layers "
        f"for up to {fine_tune_epochs} epochs..."
    )
    base_model.trainable = True

    # Freeze all layers except the last N
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
    )

    model_path = output_path / "best_model.keras"
    logger.info(f"Best validation model saved to: {model_path}")

    # --- Save class mapping ---
    class_mapping = {
        i: {"name": name, "crop": name.split("___")[0] if "___" in name else name}
        for i, name in enumerate(class_names)
    }
    mapping_path = Path(__file__).resolve().parents[1] / "data" / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    logger.info(f"Class mapping saved to: {mapping_path}")

    # --- Print results ---
    final_train_acc = history1.history["accuracy"][-1]
    final_val_acc = history1.history["val_accuracy"][-1]
    best_val_acc = max(history1.history["val_accuracy"] + history2.history["val_accuracy"])

    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Total Classes: {num_classes}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 50)

    return model, history1, history2


def main():
    parser = argparse.ArgumentParser(description="Train AgriVision AI crop disease model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./datasets/plantvillage",
        help="Path to PlantVillage dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (Phase 1)")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=10, help="Fine-tuning epochs (Phase 2)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--fine-tune-lr", type=float, default=1e-5, help="Fine-tuning learning rate"
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=20,
        help="Number of MobileNetV2 base layers to unfreeze during fine-tuning",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/models",
        help="Directory to save the trained model",
    )

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        output_dir=args.output_dir,
        fine_tune_layers=args.fine_tune_layers,
    )


if __name__ == "__main__":
    main()

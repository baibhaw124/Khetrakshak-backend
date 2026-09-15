from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# KHETRAKSHAK AI
# EfficientNetB0 Data Pipeline - MEMORY SAFE
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

TRAIN_CSV = BASE_DIR / "ml" / "training" / "train.csv"
VAL_CSV = BASE_DIR / "ml" / "training" / "validation.csv"

IMAGE_SIZE = (224, 224)

# Reduced for CPU/RAM safety
BATCH_SIZE = 16

# Avoid aggressive parallel processing on native Windows CPU
NUM_PARALLEL_CALLS = 2

# Small prefetch instead of AUTOTUNE
PREFETCH_SIZE = 1

SEED = 42


# ============================================================
# LOAD CSV
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV,
    usecols=["image_path", "class_name"]
)

val_df = pd.read_csv(
    VAL_CSV,
    usecols=["image_path", "class_name"]
)

print("=" * 60)
print("KHETRAKSHAK AI - EFFICIENTNETB0 PIPELINE")
print("=" * 60)

print(f"\nTraining images   : {len(train_df):,}")
print(f"Validation images : {len(val_df):,}")


# ============================================================
# CLASS LABELS
# ============================================================

class_names = sorted(
    train_df["class_name"].unique()
)

NUM_CLASSES = len(class_names)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

train_df["label"] = (
    train_df["class_name"]
    .map(class_to_index)
    .astype(np.int32)
)

val_df["label"] = (
    val_df["class_name"]
    .map(class_to_index)
    .astype(np.int32)
)

print(f"Number of classes : {NUM_CLASSES}")

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"{index:2d} -> {class_name}")


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        ),

        tf.keras.layers.RandomContrast(
            0.1
        ),
    ],
    name="data_augmentation"
)


def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    df,
    training=False
):

    paths = df["image_path"].to_numpy(
        dtype=str
    )

    labels = df["label"].to_numpy(
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=NUM_PARALLEL_CALLS
    )

    if training:

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=NUM_PARALLEL_CALLS
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        PREFETCH_SIZE
    )

    return dataset


# ============================================================
# CREATE TRAINING / VALIDATION DATASETS
# ============================================================

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].to_numpy()
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}


print("\n" + "=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

for index, class_name in enumerate(class_names):

    print(
        f"{class_name:<50} "
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# PIPELINE TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING DATA PIPELINE")
print("=" * 60)

images, labels = next(
    iter(train_dataset)
)

print(
    f"\nBatch image shape : {images.shape}"
)

print(
    f"Batch label shape : {labels.shape}"
)

print(
    f"Image data type   : {images.dtype}"
)

print(
    f"Pixel range       : "
    f"{tf.reduce_min(images).numpy():.2f} "
    f"to "
    f"{tf.reduce_max(images).numpy():.2f}"
)


# ============================================================
# FINAL CHECKS
# ============================================================

assert NUM_CLASSES == 38

assert images.shape[1:] == (
    224,
    224,
    3
)

assert labels.shape[0] <= BATCH_SIZE


print("\n" + "=" * 60)
print("✅ PIPELINE TEST SUCCESSFUL")
print("=" * 60)

print("\nEfficientNetB0 input:")
print("224 × 224 × 3 RGB")

print("\nMemory-safe configuration:")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Parallel operations : {NUM_PARALLEL_CALLS}")
print(f"Prefetch            : {PREFETCH_SIZE}")

print("\nReady for model construction.")
print("🚀 Khetrakshak ML pipeline is ready!")
import os
# ENVIRONMENT SETTINGS
# ============================================================
# Disable oneDNN optimizations.
# This helps avoid the MKL memory-object error seen earlier.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# KHETRAKSHAK AI
# EfficientNetB0 Training
# ============================================================
BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

TRAIN_CSV = BASE_DIR / "ml" / "training" / "train.csv"
VAL_CSV = BASE_DIR / "ml" / "training" / "validation.csv"

MODEL_DIR = BASE_DIR / "ml" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_SIZE = (224, 224)
# Memory-safe CPU training
# Reduced from 8 to 4 after CPU OOM during EfficientNet convolution.
BATCH_SIZE = 2
# Limit TensorFlow parallelism to reduce RAM usage
NUM_PARALLEL_CALLS = 1
PREFETCH_SIZE = 1

SEED = 42
# Number of EfficientNet layers to keep frozen
FINE_TUNE_AT = 200

# Training epochs
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

# Number of classes in PlantVillage dataset
NUM_CLASSES = 38


# ============================================================
# CPU THREAD LIMIT
# ============================================================

# Your machine is running TensorFlow on CPU.
# Limiting threads helps reduce RAM pressure.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ============================================================
# RANDOM SEED
# ============================================================
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# CHECK FILES
# ============================================================

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("KHETRAKSHAK AI - EFFICIENTNETB0 TRAINING")
print("=" * 60)

print("\nLoading CSV files...")

train_df = pd.read_csv(
    TRAIN_CSV,
    usecols=["image_path", "class_name"]
)

val_df = pd.read_csv(
    VAL_CSV,
    usecols=["image_path", "class_name"]
)

print("\nCSV files loaded successfully.")

print(
    f"Training images   : {len(train_df):,}"
)

print(
    f"Validation images : {len(val_df):,}"
)


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = sorted(
    train_df["class_name"].unique()
)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )

class_to_index = {
    class_name: index
    for index, class_name in enumerate(
        class_names
    )
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


print(
    f"\nNumber of classes : {len(class_names)}"
)

print("\nClass mapping:")

for index, class_name in enumerate(
    class_names
):

    print(
        f"{index:2d} -> {class_name}"
    )


# ============================================================
# CHECK LABELS
# ============================================================

if train_df["label"].isna().any():
    raise ValueError(
        "Training dataset contains unknown class labels."
    )

if val_df["label"].isna().any():
    raise ValueError(
        "Validation dataset contains unknown class labels."
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path, label):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

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


def augment(
    image,
    label
):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# DATASET CREATION
# ============================================================
def create_dataset(df, training=False):

    paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # Shuffle only training data
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 2000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # Load images
    dataset = dataset.map(
        load_image,
        num_parallel_calls=NUM_PARALLEL_CALLS
    )

    # Augmentation
    if training:
        dataset = dataset.map(
            augment,
            num_parallel_calls=NUM_PARALLEL_CALLS
        )

    # Small batch to prevent CPU RAM exhaustion
    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Only prefetch one batch
    dataset = dataset.prefetch(
        PREFETCH_SIZE
    )

    return dataset
# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

print("TensorFlow datasets ready.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].values
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}


print(
    "\nClass weights calculated successfully."
)


# ============================================================
# BUILD EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")

base_model = tf.keras.applications.EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )

)


# ============================================================
# PHASE 1
# TRANSFER LEARNING
# ============================================================

base_model.trainable = False


inputs = tf.keras.Input(

    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    name="image"

)


# EfficientNetB0 contains its own
# input preprocessing/rescaling.

x = base_model(
    inputs,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


x = tf.keras.layers.Dropout(
    0.3
)(
    x
)


outputs = tf.keras.layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    name="disease_prediction"

)(
    x
)


model = tf.keras.Model(

    inputs,
    outputs,

    name="Khetrakshak_EfficientNetB0"

)


# ============================================================
# COMPILE PHASE 1
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


print("\n" + "=" * 60)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR /
    "khetrakshak_efficientnet_best.keras"
)


callbacks_phase1 = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            best_model_path
        ),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    ),


    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1

    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1

    )

]
# ============================================================
# TRAIN PHASE 1
# ============================================================

print("\nStarting Phase 1 training...")

history_phase1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase1

)
# ============================================================
# CLEANUP
# ============================================================
# Release Python-side temporary objects before fine-tuning.
gc.collect()

# ============================================================
# PHASE 2
# PARTIAL FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)


base_model.trainable = True


# Freeze early layers.
for layer in base_model.layers[
    :FINE_TUNE_AT
]:

    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

# ============================================================
# RECOMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


print(
    f"\nFine-tuning from layer "
    f"{FINE_TUNE_AT} onward."
)

print(
    "Learning rate: 1e-5"
)


# ============================================================
# PHASE 2 CALLBACKS
# ============================================================

callbacks_phase2 = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            best_model_path
        ),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    ),


    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True,

        verbose=1

    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1

    )

]


# ============================================================
# TRAIN PHASE 2
# ============================================================

print("\nStarting Phase 2 training...")

history_phase2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase2

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (

    MODEL_DIR /
    "khetrakshak_efficientnet_final.keras"

)


model.save(
    final_model_path
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (

    MODEL_DIR /
    "class_names.txt"

)


with open(

    class_names_path,

    "w",

    encoding="utf-8"

) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

training_info_path = (

    MODEL_DIR /
    "training_info.txt"

)


with open(

    training_info_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "Khetrakshak AI - EfficientNetB0\n"
    )

    file.write(
        "================================\n"
    )

    file.write(
        f"Number of classes: {NUM_CLASSES}\n"
    )

    file.write(
        f"Image size: {IMAGE_SIZE}\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Phase 1 epochs: {INITIAL_EPOCHS}\n"
    )

    file.write(
        f"Phase 2 epochs: {FINE_TUNE_EPOCHS}\n"
    )

    file.write(
        f"Fine tune from layer: {FINE_TUNE_AT}\n"
    )

    file.write(
        "Dataset: PlantVillage\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("🎉 KHETRAKSHAK TRAINING COMPLETE")
print("=" * 60)


print(
    f"\nBest model:"
    f"\n{best_model_path}"
)


print(
    f"\nFinal model:"
    f"\n{final_model_path}"
)


print(
    f"\nClass names:"
    f"\n{class_names_path}"
)


print(
    f"\nTraining information:"
    f"\n{training_info_path}"
)


print(
    "\nIMPORTANT:"
    "\nTest dataset has NOT been used."
)


print(
    "\nNext step:"
    "\nEvaluate the best model on the untouched test set."
)


print(
    "\n🚀 Khetrakshak EfficientNetB0 training pipeline finished."
)
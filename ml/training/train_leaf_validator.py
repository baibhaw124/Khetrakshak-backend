from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


# ============================================================
# KHETRAKSHAK AI
# LEAF VALIDATOR TRAINING
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

DATASET_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "leaf_validator"
    / "prepared"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "khetrakshak_leaf_validator.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 2

EPOCHS = 10

SEED = 42


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print("=" * 70)
print("KHETRAKSHAK AI - LEAF VALIDATOR TRAINING")
print("=" * 70)

print()
print("Dataset :", DATASET_DIR)
print("Image size :", IMAGE_SIZE)
print("Batch size :", BATCH_SIZE)
print("Epochs :", EPOCHS)
print()


# ============================================================
# VERIFY DATASET
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VALIDATION_DIR}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

print()
print("Classes:")
print(train_dataset.class_names)

print()

if len(train_dataset.class_names) != 2:
    raise ValueError(
        "Leaf validator must contain exactly 2 classes."
    )


# ============================================================
# PERFORMANCE PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.1
        ),

        layers.RandomZoom(
            0.1
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print()
print("Loading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# Freeze base model initially

base_model.trainable = False


# ============================================================
# BUILD VALIDATOR MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.3
)(x)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="KhetrakshakLeafValidator"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING LEAF VALIDATOR TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "khetrakshak_leaf_validator_final.keras"
)

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("LEAF VALIDATOR TRAINING COMPLETE")
print("=" * 70)

print()
print("Best model:")
print(MODEL_PATH)

print()
print("Final model:")
print(FINAL_MODEL_PATH)

print()
print("Ready for testing.")
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# KHETRAKSHAK AI
# PHASE 2 ONLY - CONTINUE FROM SAVED MODEL
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

TRAIN_CSV = BASE_DIR / "ml" / "training" / "train.csv"
VAL_CSV = BASE_DIR / "ml" / "training" / "validation.csv"

MODEL_DIR = BASE_DIR / "ml" / "models"

# Existing Phase 1 best model
SOURCE_MODEL = (
    MODEL_DIR /
    "khetrakshak_efficientnet_best.keras"
)

# New Phase 2 checkpoint
PHASE2_BEST_MODEL = (
    MODEL_DIR /
    "khetrakshak_efficientnet_phase2_best.keras"
)

# Final Phase 2 model
FINAL_MODEL = (
    MODEL_DIR /
    "khetrakshak_efficientnet_final.keras"
)

CLASS_NAMES_FILE = (
    MODEL_DIR /
    "class_names.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

# VERY IMPORTANT:
# Your PC previously crashed with batch sizes 32, 8 and 4.
# We are using 2.
BATCH_SIZE = 2

# Keep CPU memory usage low
NUM_PARALLEL_CALLS = 1
PREFETCH_SIZE = 1

SEED = 42

# Same fine-tuning strategy as original training
FINE_TUNE_AT = 200

# Phase 2 epochs
PHASE2_EPOCHS = 8

LEARNING_RATE = 1e-5


# ============================================================
# CPU MEMORY CONTROL
# ============================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("KHETRAKSHAK AI - PHASE 2 ONLY")
print("=" * 60)

print("\nChecking required files...")

if not SOURCE_MODEL.exists():
    raise FileNotFoundError(
        f"\nSaved Phase 1 model not found:\n{SOURCE_MODEL}"
    )

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"\nTraining CSV not found:\n{TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"\nValidation CSV not found:\n{VAL_CSV}"
    )

print("\n✓ Saved Phase 1 model found")
print("✓ Training CSV found")
print("✓ Validation CSV found")


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading CSV files...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

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

NUM_CLASSES = len(class_names)

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

train_df["label"] = (
    train_df["class_name"]
    .map(class_to_index)
)

val_df["label"] = (
    val_df["class_name"]
    .map(class_to_index)
)

print(
    f"Number of classes : {NUM_CLASSES}"
)

if NUM_CLASSES != 38:
    raise ValueError(
        f"Expected 38 classes, found {NUM_CLASSES}"
    )


# ============================================================
# IMAGE LOADER
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
    name="phase2_augmentation"
)


def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    df,
    training=False
):

    paths = df[
        "image_path"
    ].values

    labels = df[
        "label"
    ].values

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(df),
                2000
            ),
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
# CREATE DATASETS
# ============================================================

print("\nCreating memory-safe datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

print("✓ Training dataset ready")
print("✓ Validation dataset ready")


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"]
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}

print("\nClass weights calculated.")


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING EXISTING PHASE 1 MODEL")
print("=" * 60)

print(
    f"\nLoading:\n{SOURCE_MODEL}"
)

model = tf.keras.models.load_model(
    SOURCE_MODEL,
    compile=False
)

print("\n✓ MODEL LOADED SUCCESSFULLY")

print(
    f"Input shape  : {model.input_shape}"
)

print(
    f"Output shape : {model.output_shape}"
)


# ============================================================
# VERIFY MODEL
# ============================================================

if model.input_shape != (
    None,
    224,
    224,
    3
):

    raise ValueError(
        "Unexpected model input shape."
    )

if model.output_shape[-1] != 38:

    raise ValueError(
        "Unexpected number of output classes."
    )


# ============================================================
# GET EFFICIENTNET BACKBONE
# ============================================================

print("\nFinding EfficientNetB0 backbone...")

try:

    base_model = model.get_layer(
        "efficientnetb0"
    )

except ValueError:

    print(
        "\nCould not find layer named "
        "'efficientnetb0'."
    )

    print(
        "\nAvailable top-level layers:"
    )

    for layer in model.layers:
        print(
            f" - {layer.name}"
        )

    raise


print(
    f"\nBackbone found: {base_model.name}"
)

print(
    f"Total backbone layers: "
    f"{len(base_model.layers)}"
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)

print(
    f"\nFine-tuning from layer "
    f"{FINE_TUNE_AT} onward."
)

print(
    f"Learning rate : {LEARNING_RATE}"
)

print(
    f"Batch size    : {BATCH_SIZE}"
)


# Enable backbone training
base_model.trainable = True


# Freeze early layers
for layer in base_model.layers[
    :FINE_TUNE_AT
]:

    layer.trainable = False


# Keep BatchNormalization frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_count = 0
non_trainable_count = 0

for layer in model.layers:

    if layer.trainable:

        trainable_count += (
            layer.count_params()
        )

    else:

        non_trainable_count += (
            layer.count_params()
        )

print(
    f"\nTrainable parameters    : "
    f"{trainable_count:,}"
)

print(
    f"Non-trainable parameters: "
    f"{non_trainable_count:,}"
)


# ============================================================
# RECOMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            PHASE2_BEST_MODEL
        ),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),


    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=2,

        restore_best_weights=True,

        verbose=1
    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=1,

        min_lr=1e-7,

        verbose=1
    )
]


# ============================================================
# PHASE 2 TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING PHASE 2 TRAINING")
print("=" * 60)

print(
    "\nIMPORTANT:"
)

print(
    "Phase 1 will NOT run."
)

print(
    "We are loading the already-trained "
    "Phase 1 model."
)

print(
    f"\nPhase 2 epochs: {PHASE2_EPOCHS}"
)

print(
    "\n🚀 Starting...\n"
)


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=PHASE2_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL MODEL")
print("=" * 60)

model.save(
    FINAL_MODEL
)

print(
    f"\n✓ Final model saved:\n"
    f"{FINAL_MODEL}"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_FILE,
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )

print(
    f"\n✓ Class names saved:\n"
    f"{CLASS_NAMES_FILE}"
)


# ============================================================
# CLEANUP
# ============================================================

gc.collect()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("🎉 PHASE 2 TRAINING COMPLETE")
print("=" * 60)

print(
    "\nBest Phase 2 model:"
)

print(
    PHASE2_BEST_MODEL
)

print(
    "\nFinal model:"
)

print(
    FINAL_MODEL
)

print(
    "\nNext step:"
)

print(
    "Evaluate the final/best model "
    "on the untouched test dataset."
)

print(
    "\n🚀 Khetrakshak ML model is ready "
    "for evaluation."
)
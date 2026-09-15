from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# KHETRAKSHAK AI
# FINAL MODEL EVALUATION
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "khetrakshak_efficientnet_best.keras"
)

TEST_CSV = (
    BASE_DIR
    / "ml"
    / "training"
    / "test.csv"
)

IMAGE_SIZE = (224, 224)

# IMPORTANT:
# Keep this small because we are running CPU-only.
BATCH_SIZE = 2

NUM_CLASSES = 38


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("KHETRAKSHAK AI - FINAL MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD TEST CSV
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(f"Test images : {len(test_df):,}")


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = sorted(
    test_df["class_name"].unique()
)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}"
    )

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

test_df["label"] = (
    test_df["class_name"]
    .map(class_to_index)
)


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
# CREATE TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["label"].values
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=1
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    1
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print(f"Input shape  : {model.input_shape}")
print(f"Output shape : {model.output_shape}")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ON TEST DATA")
print("=" * 70)

results = model.evaluate(
    test_dataset,
    verbose=1
)

test_loss = results[0]
test_accuracy = results[1]


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4%}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)

true_labels = test_df["label"].values


# ============================================================
# VERIFY PREDICTION COUNT
# ============================================================

if len(predicted_labels) != len(true_labels):

    raise ValueError(
        "Prediction count does not match test dataset size."
    )


# ============================================================
# ACCURACY VERIFICATION
# ============================================================

calculated_accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n" + "=" * 70)
print("ACCURACY VERIFICATION")
print("=" * 70)

print(
    f"\nKeras Accuracy     : {test_accuracy:.4%}"
)

print(
    f"Calculated Accuracy: {calculated_accuracy:.4%}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES)
)

print("\nShape:", cm.shape)

print("\nConfusion matrix generated successfully.")


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for index, class_name in enumerate(class_names):

    total = cm[index].sum()

    correct = cm[index, index]

    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0

    print(
        f"{index:2d} | "
        f"{accuracy:.2%} | "
        f"{class_name}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

REPORT_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "evaluation_report.txt"
)

CM_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "confusion_matrix.npy"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "KHETRAKSHAK AI - MODEL EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Test Images : {len(test_df):,}\n"
    )

    file.write(
        f"Classes     : {NUM_CLASSES}\n"
    )

    file.write(
        f"Test Loss   : {test_loss:.6f}\n"
    )

    file.write(
        f"Test Accuracy : {test_accuracy:.6%}\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(report)


np.save(
    CM_PATH,
    cm
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("✅ MODEL EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(
    f"Evaluation report : {REPORT_PATH}"
)

print(
    f"Confusion matrix  : {CM_PATH}"
)

print("\n🚀 Ready for inference testing.")
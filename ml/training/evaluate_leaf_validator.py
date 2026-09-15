from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# KHETRAKSHAK AI
# LEAF VALIDATOR EVALUATION
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

TEST_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "leaf_validator"
    / "prepared"
    / "test"
)

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "khetrakshak_leaf_validator.keras"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 2


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("KHETRAKSHAK AI - LEAF VALIDATOR EVALUATION")
print("=" * 70)


# ============================================================
# VERIFY FILES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print()
print("Classes:")
print(class_names)

print()

if len(class_names) != 2:
    raise ValueError(
        f"Expected 2 classes, found {len(class_names)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading leaf validator model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print(
    "Input shape :",
    model.input_shape
)

print()


# ============================================================
# COMPILE FOR EVALUATION
# ============================================================

model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("=" * 70)
print("EVALUATING ON UNSEEN TEST DATA")
print("=" * 70)

results = model.evaluate(
    test_dataset,
    verbose=1
)

test_loss = results[0]

test_accuracy = results[1]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.4f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("Generating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()


# ============================================================
# CONVERT PROBABILITIES TO CLASS LABELS
# ============================================================

predicted_labels = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# TRUE LABELS
# ============================================================

true_labels = np.concatenate(
    [
        labels.numpy().flatten()
        for images, labels in test_dataset
    ]
).astype(int)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print()
print(cm)

print()
print("Matrix format:")
print(
    f"Rows    = Actual classes"
)

print(
    f"Columns = Predicted classes"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for index, class_name in enumerate(class_names):

    total = cm[index].sum()

    correct = cm[index, index]

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:12s} : "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("LEAF VALIDATOR EVALUATION COMPLETE")
print("=" * 70)


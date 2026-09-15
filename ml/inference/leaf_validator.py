from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# KHETRAKSHAK AI
# LEAF VALIDATOR - SINGLE IMAGE INFERENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "khetrakshak_leaf_validator.keras"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Leaf validator model not found:\n{MODEL_PATH}"
    )


print("Loading Khetrakshak Leaf Validator...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Leaf validator loaded successfully.")

print(
    f"Input shape : {model.input_shape}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = tf.io.read_file(
        str(image_path)
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

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_leaf(image_path):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image,
        verbose=0
    )[0][0]

    # Model output:
    # 0 → leaf
    # 1 → non_leaf
    #
    # However, because directory-based training assigns
    # classes alphabetically:
    #
    # ['leaf', 'non_leaf']
    #
    # therefore:
    # prediction < 0.5 → leaf
    # prediction >= 0.5 → non_leaf

    if prediction < 0.5:

        label = "leaf"

        confidence = (
            1.0 - prediction
        ) * 100

    else:

        label = "non_leaf"

        confidence = prediction * 100

    return {
        "is_leaf": label == "leaf",
        "label": label,
        "confidence": float(confidence),
        "raw_prediction": float(prediction)
    }


# ============================================================
# TERMINAL TEST MODE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "KHETRAKSHAK AI - LEAF VALIDATOR"
    )
    print("=" * 60)

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    image_file = Path(
        image_path
    )

    if not image_file.exists():

        print(
            "\n❌ Image file not found."
        )

        raise SystemExit(1)

    result = validate_leaf(
        image_file
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(
        f"\nClassification : "
        f"{result['label']}"
    )

    print(
        f"Confidence     : "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Is Leaf        : "
        f"{result['is_leaf']}"
    )

    print("\n" + "=" * 60)
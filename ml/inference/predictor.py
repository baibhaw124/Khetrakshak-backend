from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# KHETRAKSHAK AI
# SINGLE IMAGE INFERENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "khetrakshak_efficientnet_final.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "class_names.txt"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"Class names file not found:\n"
        f"{CLASS_NAMES_PATH}"
    )


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


if len(class_names) != 38:

    raise ValueError(
        f"Expected 38 classes, "
        f"found {len(class_names)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Khetrakshak EfficientNetB0...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print(
    f"Input shape : {model.input_shape}"
)

print(
    f"Classes     : {len(class_names)}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path
):

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

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_prediction(
    class_name
):

    parts = class_name.split(
        "___",
        1
    )

    if len(parts) == 2:

        crop = parts[0]

        disease = parts[1]

    else:

        crop = "Unknown"

        disease = class_name

    return crop, disease


# ============================================================
# PREDICT
# ============================================================

def predict_image(
    image_path,
    top_k=1
):

    image = preprocess_image(
        image_path
    )

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    # Top K predictions
    top_indices = np.argsort(
        probabilities
    )[-top_k:][::-1]

    predictions = []

    for index in top_indices:

        class_name = class_names[
            index
        ]

        confidence = (
            float(
                probabilities[index]
            )
            * 100
        )

        crop, disease = (
            format_prediction(
                class_name
            )
        )

        predictions.append(
            {
                "class_name": class_name,
                "crop": crop,
                "disease": disease,
                "confidence": confidence
            }
        )

    return predictions


# ============================================================
# TERMINAL TEST MODE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "KHETRAKSHAK AI - IMAGE INFERENCE"
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

    predictions = predict_image(
        image_file,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("TOP PREDICTIONS")
    print("=" * 60)

    for position, prediction in enumerate(
        predictions,
        start=1
    ):

        print(
            f"\n{position}. "
            f"{prediction['disease']}"
        )

        print(
            f"   Crop       : "
            f"{prediction['crop']}"
        )

        print(
            f"   Confidence : "
            f"{prediction['confidence']:.2f}%"
        )

    print("\n" + "=" * 60)
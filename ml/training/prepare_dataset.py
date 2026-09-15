from pathlib import Path
import csv
import random

# ============================================================
# KHETRAKSHAK AI - DATASET SPLITTER
# ============================================================

# Dataset location
DATASET_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend\ml\dataset\plantvillage dataset\color"
)

# Output directory
OUTPUT_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend\ml\training"
)

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducible randomization
random.seed(42)

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(class_folder):
    """Return all valid image files inside a class folder."""

    return [
        file
        for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():

    print("=" * 60)
    print("KHETRAKSHAK AI - DATASET SPLITTER")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset folder not found!")
        print(DATASET_DIR)
        return

    # Find class folders
    class_folders = sorted(
        [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    print(f"\nTotal classes found: {len(class_folders)}")

    if len(class_folders) != 38:
        print("WARNING: Expected 38 classes.")

    train_rows = []
    val_rows = []
    test_rows = []

    total_images = 0

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_folder in class_folders:

        class_name = class_folder.name

        images = get_images(class_folder)

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        # Add to CSV rows
        for image in train_images:
            train_rows.append(
                [str(image), class_name]
            )

        for image in val_images:
            val_rows.append(
                [str(image), class_name]
            )

        for image in test_images:
            test_rows.append(
                [str(image), class_name]
            )

        total_images += total

        print(
            f"{class_name:<50} "
            f"{len(train_images):>5} train | "
            f"{len(val_images):>5} val | "
            f"{len(test_images):>5} test"
        )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    train_file = OUTPUT_DIR / "train.csv"
    val_file = OUTPUT_DIR / "validation.csv"
    test_file = OUTPUT_DIR / "test.csv"

    def write_csv(file_path, rows):

        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["image_path", "class_name"]
            )

            writer.writerows(rows)

    write_csv(train_file, train_rows)
    write_csv(val_file, val_rows)
    write_csv(test_file, test_rows)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(f"\nTotal images : {total_images:,}")

    print(f"\nTraining     : {len(train_rows):,}")
    print(f"Validation   : {len(val_rows):,}")
    print(f"Testing      : {len(test_rows):,}")

    print("\nCSV files created:")

    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")

    print("\nRandom seed: 42")

    print("\nReady for EfficientNet training. 🚀")


if __name__ == "__main__":
    main() 
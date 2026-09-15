from pathlib import Path
import random
import shutil


# ============================================================
# KHETRAKSHAK AI
# LEAF VALIDATOR DATASET PREPARATION
# ============================================================

BASE_DIR = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend"
)

# Original datasets
PLANTVILLAGE_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "plantvillage dataset"
    / "color"
)

NON_LEAF_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "leaf_validator"
    / "natural_images_raw"
    / "natural_images"
)

# New prepared dataset
OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "leaf_validator"
    / "prepared"
)


# ============================================================
# SETTINGS
# ============================================================

LEAF_COUNT = 5000
NON_LEAF_COUNT = 5000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ============================================================
# VALIDATE PATHS
# ============================================================

print("=" * 70)
print("KHETRAKSHAK AI - LEAF VALIDATOR DATASET PREPARATION")
print("=" * 70)

if not PLANTVILLAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nPlantVillage directory not found:\n"
        f"{PLANTVILLAGE_DIR}"
    )

if not NON_LEAF_DIR.exists():
    raise FileNotFoundError(
        f"\nNon-leaf directory not found:\n"
        f"{NON_LEAF_DIR}"
    )

print("\nOriginal datasets found successfully.")


# ============================================================
# COLLECT LEAF IMAGES
# ============================================================

print("\nCollecting PlantVillage leaf images...")

leaf_images = [
    path
    for path in PLANTVILLAGE_DIR.rglob("*")
    if path.is_file()
    and path.suffix in IMAGE_EXTENSIONS
]

print(f"Total leaf images found : {len(leaf_images)}")

if len(leaf_images) < LEAF_COUNT:
    raise ValueError(
        f"Only {len(leaf_images)} leaf images found. "
        f"Need {LEAF_COUNT}."
    )


# ============================================================
# COLLECT NON-LEAF IMAGES
# ============================================================

print("\nCollecting Natural Images non-leaf images...")

non_leaf_images = [
    path
    for path in NON_LEAF_DIR.rglob("*")
    if path.is_file()
    and path.suffix in IMAGE_EXTENSIONS
]

print(
    f"Total non-leaf images found : "
    f"{len(non_leaf_images)}"
)

if len(non_leaf_images) < NON_LEAF_COUNT:
    raise ValueError(
        f"Only {len(non_leaf_images)} non-leaf images found. "
        f"Need {NON_LEAF_COUNT}."
    )


# ============================================================
# RANDOM SAMPLING
# ============================================================

print("\nSelecting balanced samples...")

random.seed(RANDOM_SEED)

selected_leaf = random.sample(
    leaf_images,
    LEAF_COUNT
)

selected_non_leaf = random.sample(
    non_leaf_images,
    NON_LEAF_COUNT
)

print(f"Selected leaf images     : {len(selected_leaf)}")
print(f"Selected non-leaf images : {len(selected_non_leaf)}")


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(selected_leaf)
random.shuffle(selected_non_leaf)


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_dataset(images):

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train = images[:train_end]

    validation = images[
        train_end:val_end
    ]

    test = images[
        val_end:
    ]

    return train, validation, test


leaf_train, leaf_val, leaf_test = (
    split_dataset(selected_leaf)
)

non_leaf_train, non_leaf_val, non_leaf_test = (
    split_dataset(selected_non_leaf)
)


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

print("\nDataset split:")
print("-" * 50)

print(
    f"Leaf      → "
    f"Train: {len(leaf_train)}, "
    f"Validation: {len(leaf_val)}, "
    f"Test: {len(leaf_test)}"
)

print(
    f"Non-leaf  → "
    f"Train: {len(non_leaf_train)}, "
    f"Validation: {len(non_leaf_val)}, "
    f"Test: {len(non_leaf_test)}"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

directories = [
    OUTPUT_DIR / "train" / "leaf",
    OUTPUT_DIR / "train" / "non_leaf",

    OUTPUT_DIR / "validation" / "leaf",
    OUTPUT_DIR / "validation" / "non_leaf",

    OUTPUT_DIR / "test" / "leaf",
    OUTPUT_DIR / "test" / "non_leaf",
]

print("\nCreating output directories...")

for directory in directories:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(
    images,
    destination,
    prefix
):

    for number, source in enumerate(
        images,
        start=1
    ):

        extension = source.suffix.lower()

        destination_file = (
            destination
            / f"{prefix}_{number:05d}{extension}"
        )

        shutil.copy2(
            source,
            destination_file
        )

        if number % 500 == 0:
            print(
                f"  Copied {number} images..."
            )


# ============================================================
# COPY LEAF DATA
# ============================================================

print("\nPreparing LEAF training data...")

copy_images(
    leaf_train,
    OUTPUT_DIR / "train" / "leaf",
    "leaf_train"
)

print("Preparing LEAF validation data...")

copy_images(
    leaf_val,
    OUTPUT_DIR / "validation" / "leaf",
    "leaf_val"
)

print("Preparing LEAF test data...")

copy_images(
    leaf_test,
    OUTPUT_DIR / "test" / "leaf",
    "leaf_test"
)


# ============================================================
# COPY NON-LEAF DATA
# ============================================================

print("\nPreparing NON-LEAF training data...")

copy_images(
    non_leaf_train,
    OUTPUT_DIR / "train" / "non_leaf",
    "nonleaf_train"
)

print("Preparing NON-LEAF validation data...")

copy_images(
    non_leaf_val,
    OUTPUT_DIR / "validation" / "non_leaf",
    "nonleaf_val"
)

print("Preparing NON-LEAF test data...")

copy_images(
    non_leaf_test,
    OUTPUT_DIR / "test" / "non_leaf",
    "nonleaf_test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print("\nFinal dataset:")

print(
    f"\nTRAIN"
    f"\n  Leaf      : {len(leaf_train)}"
    f"\n  Non-leaf  : {len(non_leaf_train)}"
)

print(
    f"\nVALIDATION"
    f"\n  Leaf      : {len(leaf_val)}"
    f"\n  Non-leaf  : {len(non_leaf_val)}"
)

print(
    f"\nTEST"
    f"\n  Leaf      : {len(leaf_test)}"
    f"\n  Non-leaf  : {len(non_leaf_test)}"
)

print(
    "\nOriginal datasets were NOT modified."
)

print("\nReady for validator training.")
print("=" * 70)
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# Khetrakshak AI - PlantVillage Dataset Analyzer
# --------------------------------------------------

DATASET_PATH = Path(
    r"D:\SIH\Khetrakshak\khetrakshak-backend\ml\dataset\plantvillage dataset\color"
)

# .\ml\ml_venv\Scripts\Activate.ps1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
total_classes = 0
class_counts = {}

corrupted_images = 0


print("=" * 60)
print("       KHETRAKSHAK AI - DATASET ANALYZER")
print("=" * 60)

if not DATASET_PATH.exists():
    print("\n❌ Dataset folder not found!")
    print(DATASET_PATH)
    exit()

# Find class folders
class_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

total_classes = len(class_folders)

print(f"\nDataset path:")
print(DATASET_PATH)

print(f"\nTotal Classes: {total_classes}")
print("\nImages per class:")
print("-" * 60)

for class_folder in class_folders:

    count = 0

    for image_file in class_folder.iterdir():

        if image_file.suffix.lower() in IMAGE_EXTENSIONS:

            count += 1

            # Check whether image can actually be opened
            try:
                with Image.open(image_file) as img:
                    img.verify()
            except Exception:
                corrupted_images += 1

    class_counts[class_folder.name] = count
    total_images += count

    print(f"{class_folder.name:<40} {count:>6}")

# --------------------------------------------------
# Statistics
# --------------------------------------------------

counts = list(class_counts.values())

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total Classes       : {total_classes}")
print(f"Total Images        : {total_images:,}")

if counts:
    print(f"Largest Class       : {max(counts):,}")
    print(f"Smallest Class      : {min(counts):,}")
    print(f"Average/Class       : {total_images / total_classes:,.2f}")

print(f"Corrupted Images    : {corrupted_images}")

# --------------------------------------------------
# Largest and smallest classes
# --------------------------------------------------

if class_counts:

    print("\n" + "=" * 60)
    print("LARGEST CLASSES")
    print("=" * 60)

    for name, count in sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:
        print(f"{name:<40} {count:>6}")

    print("\n" + "=" * 60)
    print("SMALLEST CLASSES")
    print("=" * 60)

    for name, count in sorted(
        class_counts.items(),
        key=lambda x: x[1]
    )[:5]:
        print(f"{name:<40} {count:>6}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
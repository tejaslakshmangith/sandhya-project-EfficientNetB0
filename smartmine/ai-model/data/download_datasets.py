"""
Download and organize mineral/rock/ore datasets from Kaggle.

Prerequisites:
    pip install kaggle
    Place your Kaggle API credentials at ~/.kaggle/kaggle.json
    (or set KAGGLE_USERNAME / KAGGLE_KEY environment variables)

Usage:
    python download_datasets.py
"""

import os
import shutil
import random
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR                          # smartmine/ai-model/data/
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

CLASSES = ["mineral", "rock", "ore"]
VAL_SPLIT = 0.20          # 20 % of images go to validation
RANDOM_SEED = 42

# Kaggle dataset slugs
MINERAL_DATASET = "floriangeillon/mineral-photos"
ROCK_DATASET = "salmaneunus/rock-classification"

# Temporary extraction directories
TMP_MINERAL = os.path.join(DATA_DIR, "_tmp_mineral")
TMP_ROCK = os.path.join(DATA_DIR, "_tmp_rock")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def is_image(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTENSIONS


def collect_images(root: str) -> list:
    """Recursively collect all image file paths under *root*."""
    images = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_image(fname):
                images.append(os.path.join(dirpath, fname))
    return images


def copy_images(image_paths: list, dest_dir: str, label: str) -> None:
    """Copy *image_paths* into *dest_dir*, renaming to avoid collisions."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(image_paths):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(dest_dir, f"{label}_{i:05d}{ext}")
        shutil.copy2(src, dst)


def split_and_copy(image_paths: list, class_name: str, seed: int = RANDOM_SEED) -> tuple:
    """Split images into train/val and copy them.  Returns (n_train, n_val)."""
    random.seed(seed)
    shuffled = list(image_paths)
    random.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - VAL_SPLIT)))
    train_imgs = shuffled[:split]
    val_imgs = shuffled[split:]

    train_dest = os.path.join(TRAIN_DIR, class_name)
    val_dest = os.path.join(VAL_DIR, class_name)

    copy_images(train_imgs, train_dest, class_name)
    copy_images(val_imgs, val_dest, class_name)

    return len(train_imgs), len(val_imgs)


def download_kaggle_dataset(slug: str, dest: str) -> None:
    """Download and unzip a Kaggle dataset into *dest*."""
    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError:
        raise SystemExit(
            "kaggle package not found. Install it with: pip install kaggle"
        )

    os.makedirs(dest, exist_ok=True)
    print(f"  Downloading {slug} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"],
        check=True,
    )
    print(f"  Downloaded and extracted to: {dest}")


# ---------------------------------------------------------------------------
# Dataset-specific extraction logic
# ---------------------------------------------------------------------------

def extract_mineral_images() -> list:
    """Return image paths from the mineral-photos dataset."""
    images = collect_images(TMP_MINERAL)
    print(f"  Found {len(images)} mineral images.")
    return images


def extract_rock_and_ore_images() -> tuple:
    """Return (rock_images, ore_images) from the rock-classification dataset.

    The rock-classification dataset may use sub-folders named after rock /
    ore types.  We use a simple keyword heuristic to separate ores from
    rocks.  All images in sub-folders whose name contains an ore-related
    keyword are mapped to the 'ore' class; the remainder go to 'rock'.
    """
    ore_keywords = {
        "ore", "iron", "copper", "gold", "silver", "zinc", "lead",
        "nickel", "bauxite", "chromite", "magnetite", "hematite",
        "chalcopyrite", "galena", "sphalerite", "pyrite",
    }

    rock_images = []
    ore_images = []

    for dirpath, dirnames, filenames in os.walk(TMP_ROCK):
        folder_name = os.path.basename(dirpath).lower()
        is_ore_folder = any(kw in folder_name for kw in ore_keywords)

        for fname in filenames:
            if is_image(fname):
                full_path = os.path.join(dirpath, fname)
                if is_ore_folder:
                    ore_images.append(full_path)
                else:
                    rock_images.append(full_path)

    # Fallback: if nothing was identified as ore, put all into rock and warn
    if not ore_images:
        print(
            "  WARNING: No ore sub-folders detected by keyword heuristic. "
            "All rock-classification images assigned to 'rock'. "
            "Manually move ore images to data/train/ore and data/val/ore if needed."
        )
        rock_images = collect_images(TMP_ROCK)

    print(f"  Found {len(rock_images)} rock images and {len(ore_images)} ore images.")
    return rock_images, ore_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SmartMine Ops — Dataset Downloader")
    print("=" * 60)

    # Ensure output directories exist
    for cls in CLASSES:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    # ---- Mineral dataset ------------------------------------------------
    print("\n[1/2] Downloading mineral-photos dataset ...")
    download_kaggle_dataset(MINERAL_DATASET, TMP_MINERAL)
    mineral_images = extract_mineral_images()
    if mineral_images:
        n_train, n_val = split_and_copy(mineral_images, "mineral")
        print(f"  Mineral  → train: {n_train}, val: {n_val}")
    else:
        print("  WARNING: No mineral images found after extraction.")

    # ---- Rock / Ore dataset ---------------------------------------------
    print("\n[2/2] Downloading rock-classification dataset ...")
    download_kaggle_dataset(ROCK_DATASET, TMP_ROCK)
    rock_images, ore_images = extract_rock_and_ore_images()

    if rock_images:
        n_train, n_val = split_and_copy(rock_images, "rock")
        print(f"  Rock     → train: {n_train}, val: {n_val}")
    else:
        print("  WARNING: No rock images found after extraction.")

    if ore_images:
        n_train, n_val = split_and_copy(ore_images, "ore")
        print(f"  Ore      → train: {n_train}, val: {n_val}")
    else:
        print("  WARNING: No ore images found. You may need to curate them manually.")

    # ---- Cleanup --------------------------------------------------------
    print("\nCleaning up temporary extraction directories ...")
    for tmp in (TMP_MINERAL, TMP_ROCK):
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
            print(f"  Removed {tmp}")

    # ---- Summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("Resulting structure:")
    for split in ("train", "val"):
        for cls in CLASSES:
            path = os.path.join(DATA_DIR, split, cls)
            count = len([f for f in os.listdir(path) if is_image(f)]) if os.path.isdir(path) else 0
            print(f"  data/{split}/{cls}/  → {count} images")
    print("=" * 60)
    print("\nNext step: run  python train_efficientnet.py  from smartmine/ai-model/")


if __name__ == "__main__":
    main()

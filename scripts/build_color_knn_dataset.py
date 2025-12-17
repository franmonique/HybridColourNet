# build_color_knn_dataset.py
import argparse
from pathlib import Path
import cv2
import numpy as np
import os

# Required first 5 color folders - MUST exist
REQUIRED_COLORS = [
    "black",
    "white",
    "grey",
    "blue",
    "red"
]


def extract_color_feature(img_bgr):
    """Tiny feature: mean Lab of center region (3 floats)."""
    h, w = img_bgr.shape[:2]

    # central 60% crop
    y1, y2 = int(0.2 * h), int(0.8 * h)
    x1, x2 = int(0.2 * w), int(0.8 * w)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab_flat = lab.reshape(-1, 3)
    mean_lab = lab_flat.mean(axis=0)
    return mean_lab.astype(np.float32)


def build_dataset(train_root):
    train_root = Path(train_root)

    features = []
    labels = []
    per_class_count = {}

    print(f"Building KNN dataset from: {train_root.resolve()}")

    # First, verify required folders exist
    missing_folders = []
    for color_name in REQUIRED_COLORS:
        class_dir = train_root / color_name
        if not class_dir.is_dir():
            missing_folders.append(color_name)
    
    if missing_folders:
        raise RuntimeError(
            f"Missing required folders: {missing_folders}\n"
            f"Please ensure these folders exist: {REQUIRED_COLORS}"
        )

    # Get all available color folders (alphabetically sorted)
    all_folders = [d for d in train_root.iterdir() if d.is_dir()]
    color_folders = REQUIRED_COLORS.copy()  # Start with required colors
    
    # Add any additional folders found (excluding duplicates)
    for folder in all_folders:
        folder_name = folder.name
        if folder_name not in color_folders:
            color_folders.append(folder_name)
    
    print(f"\nDetected color folders: {color_folders}")
    
    # Process all color folders
    for class_idx, color_name in enumerate(color_folders):
        class_dir = train_root / color_name
        
        # Check if folder exists (all required ones should, optional ones might not)
        if not class_dir.is_dir():
            if color_name in REQUIRED_COLORS:
                
                print(f"[ERROR] Required folder '{color_name}' not found: {class_dir}")
                continue
            else:
                # Skip optional folders that don't exist
                print(f"[INFO] Optional folder '{color_name}' not found, skipping")
                continue
        
        print(f"\nProcessing '{color_name}' from {class_dir}")

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(class_dir.glob(ext))

        count = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feat = extract_color_feature(img)
            features.append(feat)
            labels.append(class_idx)
            count += 1

        per_class_count[color_name] = count
        print(f"  -> {count} images used")

    if not features:
        raise RuntimeError("No images found! Check dataset path.")

    X = np.stack(features, axis=0)
    y = np.array(labels, dtype=np.int64)
    class_names = np.array(color_folders)

    print("\nDataset built:")
    print("  features:", X.shape)
    print("  labels  :", y.shape)
    print("  classes :", class_names.tolist())
    print("  per-class counts:", per_class_count)

    # -----------------------------
    # Save into results/knn folder
    # -----------------------------
    save_folder = Path("results/knn")
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / "color_knn_data.npz"
    np.savez(save_path, X=X, y=y, classes=class_names)

    print(f"\nSaved KNN dataset to: {save_path.resolve()}")
    return class_names.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to folder containing required subfolders: black, white, grey, blue, red, and optionally other color folders"
    )
    args = parser.parse_args()
    build_dataset(args.train_dir)
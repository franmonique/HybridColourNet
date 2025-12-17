import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
NPZ_PATH = "results/knn/color_knn_data.npz"
OUT_DIR = Path("results/knn")

# 6-class system
FINAL_CLASSES = ["black", "white", "grey", "blue", "red", "other"]


# ---------------------------------------------------------
# LOAD KNN DATA
# ---------------------------------------------------------
def load_knn_data(npz_path: str):
    """Load the trained KNN data from NPZ file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"KNN data not found at: {npz_path}. Please run build_color_knn_dataset.py first."
        )

    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X"].astype(np.float32)   # (N, 3)
    y_train = data["y"].astype(np.int64)     # (N,)
    class_names = data["classes"].tolist()   # list[str]

    print("Loaded KNN data from:", npz_path)
    print("  X_train.shape:", X_train.shape)
    print("  y_train.shape:", y_train.shape)
    print("  classes:", class_names)

    return X_train, y_train, class_names


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_color_feature(img_bgr: np.ndarray) -> np.ndarray:
    """
    Same feature as used to build the KNN dataset:
    mean Lab of the central region.
    """
    h, w = img_bgr.shape[:2]

    y1, y2 = int(0.2 * h), int(0.8 * h)
    x1, x2 = int(0.2 * w), int(0.8 * w)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)  # [L, a, b]
    return mean_lab.astype(np.float32)


def knn_predict(feature: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int = 3) -> int:
    """Simple KNN using squared Euclidean distance in 3D feature space."""
    diff = X_train - feature[None, :]             # (N, 3)
    dists = np.sum(diff * diff, axis=1)           # (N,)
    idx = np.argsort(dists)[:k]
    neighbor_labels = y_train[idx]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


# ---------------------------------------------------------
# MAPPING to 6-class system
# ---------------------------------------------------------
def map_name_to_6class(color_name: str) -> int:
    """Map a folder/class name to FINAL_CLASSES index."""
    c = color_name.lower()
    if c in FINAL_CLASSES:
        return FINAL_CLASSES.index(c)
    return FINAL_CLASSES.index("other")


def map_knn_pred_to_6class(pred_idx: int, knn_class_names) -> int:
    """Map KNN predicted class index -> 6-class index."""
    if pred_idx < 0 or pred_idx >= len(knn_class_names):
        return FINAL_CLASSES.index("other")

    pred_name = str(knn_class_names[pred_idx]).lower()
    if pred_name in FINAL_CLASSES:
        return FINAL_CLASSES.index(pred_name)
    return FINAL_CLASSES.index("other")


# ---------------------------------------------------------
# LOAD TEST DATA (flat folders: test_root/<class>/*.jpg)
# ---------------------------------------------------------
def load_test_images_flat(test_root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):

    test_root = Path(test_root)
    if not test_root.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_root}")

    class_dirs = [p for p in test_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found inside: {test_root}")

    samples = []
    for class_dir in class_dirs:
        true_idx = map_name_to_6class(class_dir.name)

        for img_path in class_dir.iterdir():  # non-recursive (match CNN test style)
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in exts:
                continue
            samples.append((img_path, true_idx))

    return samples


# ---------------------------------------------------------
# EVALUATE KNN
# ---------------------------------------------------------
def evaluate_knn_6class(samples, X_train, y_train, knn_class_names, k=3):
    y_true = []
    y_pred = []

    for img_path, true_idx in samples:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        feat = extract_color_feature(img)
        raw_pred = knn_predict(feat, X_train, y_train, k=k)
        pred_idx = map_knn_pred_to_6class(raw_pred, knn_class_names)

        y_true.append(true_idx)
        y_pred.append(pred_idx)

    return np.array(y_pred, dtype=np.int64), np.array(y_true, dtype=np.int64)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test KNN Vehicle Color Model (6-class)")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test dataset folder")
    parser.add_argument("--knn-npz", type=str, default=NPZ_PATH, help="Path to trained KNN data (.npz)")
    parser.add_argument("--k", type=int, default=3, help="K for KNN")
    args = parser.parse_args()

    # Setup output dir (match CNN style)
    results_dir = OUT_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KNN COLOR CLASSIFICATION - TESTING (6-class)")
    print("=" * 60)
    print(f"Test directory: {args.test_dir}")
    print(f"KNN data:       {args.knn_npz}")
    print(f"k:              {args.k}")
    print("=" * 60)

    # 1. Load KNN data
    print("\n1. LOADING KNN DATA")
    print("-" * 40)
    X_train, y_train, knn_class_names = load_knn_data(args.knn_npz)

    print(f"Final 6 classes: {FINAL_CLASSES}")

    # 2. Load test data
    print("\n2. LOADING TEST DATA")
    print("-" * 40)

    samples = load_test_images_flat(args.test_dir)

    if len(samples) == 0:
        print("ERROR: No test images loaded")
        exit(1)

    print(f"Test images: {len(samples)}")

    # Raw class distribution (6-class indices)
    y_true_tmp = np.array([t for _, t in samples], dtype=np.int64)
    print(f"Raw class distribution: {np.bincount(y_true_tmp, minlength=6)}")

    # 3. 6-class evaluation
    print("\n3. 6-CLASS EVALUATION")
    print("-" * 40)

    final_preds, final_true = evaluate_knn_6class(samples, X_train, y_train, knn_class_names, k=args.k)

    accuracy = accuracy_score(final_true, final_preds)
    print(f"Overall Accuracy: {accuracy:.2%}")

    # 4. Confusion matrix (percentage)
    print("\n4. CONFUSION MATRIX (PERCENTAGE)")
    print("-" * 40)

    cm = confusion_matrix(final_true, final_preds, labels=range(6))
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=FINAL_CLASSES)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(f"KNN 6-Class Confusion Matrix (%)\nAccuracy: {accuracy:.2%}")
    plt.tight_layout()

    cm_percent_path = results_dir / "knn_confusion_matrix_percent.png"
    plt.savefig(cm_percent_path, dpi=150)
    print(f"Saved: {cm_percent_path}")

    # 5. Confusion matrix (counts)
    print("\n5. CONFUSION MATRIX (RAW COUNTS)")
    print("-" * 40)

    plt.figure(figsize=(8, 6))
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FINAL_CLASSES)
    disp2.plot(cmap="Greens", values_format="d")
    plt.title(f"KNN 6-Class Confusion Matrix (Counts)\nAccuracy: {accuracy:.2%}")
    plt.tight_layout()

    cm_counts_path = results_dir / "knn_confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=150)
    print(f"Saved: {cm_counts_path}")

    # 6. Per-class accuracy 
    print("\n6. PER-CLASS ACCURACY")
    print("-" * 40)

    per_class_acc = []
    for i, color in enumerate(FINAL_CLASSES):
        mask = final_true == i
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.sum(final_preds[mask] == i) / count)
            per_class_acc.append((color, acc, count))
        else:
            per_class_acc.append((color, 0.0, 0))

    # Sort by accuracy 
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    colors_sorted = [x[0] for x in per_class_acc]
    accs_sorted = [x[1] for x in per_class_acc]
    counts_sorted = [x[2] for x in per_class_acc]

    plt.figure(figsize=(10, 5))
    colors_bars = []
    for color in colors_sorted:
        if color == "black": colors_bars.append("black")
        elif color == "white": colors_bars.append("lightgray")
        elif color == "grey": colors_bars.append("gray")
        elif color == "blue": colors_bars.append("blue")
        elif color == "red": colors_bars.append("red")
        else: colors_bars.append("orange")

    bars = plt.bar(colors_sorted, accs_sorted, color=colors_bars, edgecolor="black")

    for bar, acc, count in zip(bars, accs_sorted, counts_sorted):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{acc:.1%}\n(n={count})", ha="center", va="bottom", fontsize=9)

    plt.axhline(y=accuracy, color="red", linestyle="--", label=f"Overall: {accuracy:.2%}")
    plt.ylim(0, 1.1)
    plt.xlabel("Color Class")
    plt.ylabel("Accuracy")
    plt.title("KNN Per-Class Accuracy (6-Class System)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = results_dir / "knn_per_class_accuracy.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved: {chart_path}")

    # 7. Classification report
    print("\n7. CLASSIFICATION REPORT")
    print("-" * 40)

    report = classification_report(final_true, final_preds, target_names=FINAL_CLASSES, digits=3, zero_division=0)
    print(report)

    report_path = results_dir / "knn_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("KNN 6-CLASS CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set: {args.test_dir}\n")
        f.write(f"KNN data: {args.knn_npz}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"Total samples: {len(final_true)}\n")
        f.write(f"Overall accuracy: {accuracy:.2%}\n\n")
        f.write("Per-class accuracy:\n")
        f.write("-" * 30 + "\n")
        for color, acc, count in per_class_acc:
            f.write(f"{color:10s}: {acc:6.2%} (n={count})\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write("Detailed report:\n")
        f.write(report)

    print(f"Report saved: {report_path}")

    # 8. Save predictions 
    npz_path = results_dir / "knn_6class_predictions.npz"
    np.savez(
        npz_path,
        final_true=final_true,
        final_preds=final_preds,
        class_names=np.array(FINAL_CLASSES, dtype=object),
        cm_percent=cm_percent,
        cm_counts=cm,
    )
    print(f"Predictions saved: {npz_path}")

    # 9. Show plots
    plt.show()

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"6-class accuracy: {accuracy:.2%}")
    print(f"\nOutput files in: {results_dir}")
    print(f"  • {cm_percent_path.name} (percentage matrix)")
    print(f"  • {cm_counts_path.name} (counts matrix)")
    print(f"  • {chart_path.name} (accuracy chart)")
    print(f"  • {report_path.name} (detailed report)")
    print("=" * 60)


if __name__ == "__main__":
    main()

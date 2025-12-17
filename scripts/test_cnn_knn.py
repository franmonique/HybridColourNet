import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

from vehicle_color_model import VehicleColorNeuralNetwork
from color_utils import (
    get_box_dominant_color_bgr,
    set_knn_data_path,
    load_knn_data,
)

def fuse_color(cnn_color, cnn_conf, knn_color):
    """
    Combine CNN + KNN predictions for a single crop.

    cnn_color: str  (e.g. "red")
    cnn_conf : float in [0,1]
    knn_color: str
    returns: (final_color: str, source: "cnn" or "knn")
    """

    # 1. High-confidence CNN → trust it
    if cnn_conf > 0.60:
        return cnn_color, "cnn"

    # 2. Low-confidence CNN → trust KNN
    if cnn_conf < 0.40:
        return knn_color, "knn"

    # 3. Medium confidence: if they agree, easy
    if cnn_color == knn_color:
        return cnn_color, "cnn"  # default to CNN

    # 4. Special handling for GREY confusion
    if cnn_color == "grey" and knn_color in ["black", "white"]:
        # CNN says grey, KNN says black/white → trust KNN
        return knn_color, "knn"

    if knn_color == "grey" and cnn_color in ["black", "white"]:
        # KNN says grey, CNN says black/white → trust CNN
        return cnn_color, "cnn"

    # 5. Default: fall back to CNN (more accurate overall)
    return cnn_color, "cnn"


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test CNN+KNN fusion on vehicle color test set (6-class)."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to test dataset folder (same format as for CNN).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/cnn/vehicle_color_best.h5",
        help="Path to trained CNN model (.h5).",
    )
    parser.add_argument(
        "--knn-data",
        type=str,
        default=None,
        help="Path to KNN data file (color_knn_data.npz). If not given, uses default loader.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Image size for model input.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/cnn_knn",
        help="Directory to save evaluation outputs.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CNN + KNN FUSION - COLOR CLASSIFICATION TEST (6-class)")
    print("=" * 60)
    print(f"Test directory : {args.test_dir}")
    print(f"CNN model path : {args.model_path}")
    print(f"KNN data path  : {args.knn_data or '(auto)'}")
    print(f"Results dir    : {results_dir}")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load CNN model
    # -------------------------------------------------
    if not os.path.exists(args.model_path):
        print(f"ERROR: CNN model not found at {args.model_path}")
        return

    nn = VehicleColorNeuralNetwork(img_size=args.img_size)
    nn.load_model(args.model_path)

    print("\n[1] CNN MODEL LOADED")
    print("-" * 40)
    print(f"Training classes (original): {nn.num_classes}")
    print(f"Final 6 classes            : {nn.final_classes}")

    # -------------------------------------------------
    # 2. Load KNN data
    # -------------------------------------------------
    print("\n[2] LOADING KNN DATA")
    print("-" * 40)
    if args.knn_data:
        if not os.path.exists(args.knn_data):
            print(f"ERROR: KNN data not found at {args.knn_data}")
            return
        set_knn_data_path(args.knn_data)
        print(f"Using KNN data: {args.knn_data}")
    else:
        print("Auto-loading KNN data via color_utils.load_knn_data()...")
        load_knn_data()
        print("KNN data loaded.")

    # -------------------------------------------------
    # 3. Load test dataset
    # -------------------------------------------------
    print("\n[3] LOADING TEST DATA")
    print("-" * 40)

    X_test, y_test = nn.load_dataset(args.test_dir, max_per_class=1000)

    if len(X_test) == 0:
        print("ERROR: No test images loaded.")
        return

    print(f"Test images: {len(X_test)}")
    print(f"Raw class distribution (10-class): {np.bincount(y_test)}")

    # Ground-truth in 6-class mapping:
    # 0–4: main colours, 5: "other"
    final_true = np.array(
        [yt if yt < 5 else 5 for yt in y_test], dtype=np.int64
    )

    # -------------------------------------------------
    # 4. CNN-only predictions (for comparison)
    # -------------------------------------------------
    print("\n[4] CNN-ONLY PREDICTIONS (6-class mapping)")
    print("-" * 40)

    probs = nn.model.predict(X_test, verbose=0)  # shape: (N, 10)
    preds_10 = np.argmax(probs, axis=1)

    cnn_conf = np.max(probs, axis=1)  # highest softmax per sample

    # Map to 6-class: 0–4 -> same, else -> 5 ("other")
    cnn_final_idx = np.where(preds_10 < 5, preds_10, 5)
    cnn_final_true = final_true  # same GT

    cnn_acc = accuracy_score(cnn_final_true, cnn_final_idx)
    print(f"CNN-only 6-class accuracy: {cnn_acc:.2%}")

    # -------------------------------------------------
    # 5. CNN+KNN Fusion predictions
    # -------------------------------------------------
    print("\n[5] CNN+KNN FUSION PREDICTIONS")
    print("-" * 40)

    fused_final_idx = []
    fused_source = []  # "cnn" vs "knn"

    # Helper: map index (0–5) -> color name
    idx_to_color = {i: c for i, c in enumerate(nn.final_classes)}

    for i in range(len(X_test)):
        # CNN part
        pred_10 = preds_10[i]
        conf = cnn_conf[i]

        cnn_idx_6 = pred_10 if pred_10 < 5 else 5
        cnn_color = idx_to_color[cnn_idx_6]

        # Rebuild BGR image for KNN:
        # X_test assumed in [0, 1] and RGB order (common Keras style).
        img_rgb = (X_test[i] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]

        # KNN color using full crop
        knn_color = get_box_dominant_color_bgr(
            img_bgr, 0, 0, w, h, debug=False, tag="test"
        )

        # Map unexpected color names to "other"
        if knn_color not in nn.final_classes:
            knn_color = "other"

        # Fuse
        final_color, src = fuse_color(cnn_color, conf, knn_color)

        fused_source.append(src)
        fused_final_idx.append(nn.final_classes.index(final_color))

    fused_final_idx = np.array(fused_final_idx, dtype=np.int64)

    fusion_acc = accuracy_score(final_true, fused_final_idx)
    print(f"Fusion 6-class accuracy: {fusion_acc:.2%}")
    print(f"Accuracy improvement over CNN-only: {fusion_acc - cnn_acc:+.2%}")

    # -------------------------------------------------
    # 6. Confusion matrices (fusion)
    # -------------------------------------------------
    print("\n[6] CONFUSION MATRICES (FUSION)")
    print("-" * 40)

    cm = confusion_matrix(final_true, fused_final_idx, labels=range(6))
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)

    # Percentage confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent, display_labels=nn.final_classes
    )
    disp.plot(cmap="Blues", values_format=".1%")
    plt.title(f"CNN+KNN Fusion - 6-Class Confusion Matrix (%)\nAccuracy: {fusion_acc:.2%}")
    plt.tight_layout()
    cm_percent_path = results_dir / "fusion_confusion_matrix_percent.png"
    plt.savefig(cm_percent_path, dpi=150)
    print(f"Saved: {cm_percent_path}")

    # Raw counts confusion matrix
    plt.figure(figsize=(8, 6))
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=nn.final_classes
    )
    disp2.plot(cmap="Greens", values_format="d")
    plt.title(
        f"CNN+KNN Fusion - 6-Class Confusion Matrix (Counts)\nAccuracy: {fusion_acc:.2%}"
    )
    plt.tight_layout()
    cm_counts_path = results_dir / "fusion_confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=150)
    print(f"Saved: {cm_counts_path}")

    # -------------------------------------------------
    # 7. Per-class accuracy (fusion)
    # -------------------------------------------------
    print("\n[7] PER-CLASS ACCURACY (FUSION)")
    print("-" * 40)

    per_class_acc = []
    sample_counts = []

    for i, color in enumerate(nn.final_classes):
        mask = final_true == i
        count = np.sum(mask)
        sample_counts.append(count)

        if count > 0:
            acc = np.sum(fused_final_idx[mask] == i) / count
        else:
            acc = 0.0

        per_class_acc.append((color, acc, count))

    # Sort by accuracy
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    colors_sorted = [x[0] for x in per_class_acc]
    accs_sorted = [x[1] for x in per_class_acc]
    counts_sorted = [x[2] for x in per_class_acc]

    plt.figure(figsize=(10, 5))
    colors_bars = []
    for color in colors_sorted:
        if color == "black":
            colors_bars.append("black")
        elif color == "white":
            colors_bars.append("lightgray")
        elif color == "grey":
            colors_bars.append("gray")
        elif color == "blue":
            colors_bars.append("blue")
        elif color == "red":
            colors_bars.append("red")
        else:
            colors_bars.append("orange")

    bars = plt.bar(colors_sorted, accs_sorted, color=colors_bars, edgecolor="black")

    for bar, acc, count in zip(bars, accs_sorted, counts_sorted):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{acc:.1%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.axhline(
        y=fusion_acc,
        color="red",
        linestyle="--",
        label=f"Overall fusion: {fusion_acc:.2%}",
    )
    plt.axhline(
        y=cnn_acc,
        color="green",
        linestyle="--",
        label=f"CNN-only: {cnn_acc:.2%}",
    )

    plt.ylim(0, 1.1)
    plt.xlabel("Color Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy (6-Class) - CNN+KNN Fusion")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = results_dir / "fusion_per_class_accuracy.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved: {chart_path}")

    # -------------------------------------------------
    # 8. Classification report (fusion)
    # -------------------------------------------------
    print("\n[8] CLASSIFICATION REPORT (FUSION)")
    print("-" * 40)

    report = classification_report(
        final_true,
        fused_final_idx,
        target_names=nn.final_classes,
        digits=3,
    )
    print(report)

    report_path = results_dir / "fusion_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("CNN + KNN FUSION - 6-CLASS CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set         : {args.test_dir}\n")
        f.write(f"CNN model        : {args.model_path}\n")
        f.write(f"KNN data         : {args.knn_data or 'auto'}\n")
        f.write(f"Total samples    : {len(X_test)}\n")
        f.write(f"CNN-only accuracy: {cnn_acc:.2%}\n")
        f.write(f"Fusion accuracy  : {fusion_acc:.2%}\n")
        f.write(f"Improvement      : {fusion_acc - cnn_acc:+.2%}\n\n")
        f.write("Per-class (sorted by fusion accuracy):\n")
        f.write("-" * 40 + "\n")
        for color, acc, count in per_class_acc:
            f.write(f"{color:10s}: {acc:6.2%} (n={count})\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("Detailed classification report (fusion):\n\n")
        f.write(report)

    print(f"Report saved: {report_path}")

    # -------------------------------------------------
    # 9. Save predictions for comparison
    # -------------------------------------------------
    npz_path = results_dir / "fusion_6class_predictions.npz"
    np.savez(
        npz_path,
        final_true=final_true,
        cnn_preds=cnn_final_idx,
        fused_preds=fused_final_idx,
        class_names=np.array(nn.final_classes),
        cm_percent=cm_percent,
        cm_counts=cm,
    )
    print(f"Predictions saved: {npz_path}")

    # Show plots interactively
    plt.show()

    print("\n" + "=" * 60)
    print("FUSION TESTING COMPLETE")
    print("=" * 60)
    print(f"CNN-only accuracy : {cnn_acc:.2%}")
    print(f"Fusion accuracy   : {fusion_acc:.2%}")
    print(f"Improvement       : {fusion_acc - cnn_acc:+.2%}")
    print(f"\nOutput files in: {results_dir}")
    print(f"  • {cm_percent_path.name}")
    print(f"  • {cm_counts_path.name}")
    print(f"  • {chart_path.name}")
    print(f"  • {report_path.name}")
    print(f"  • {npz_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

from vehicle_color_model import VehicleColorNeuralNetwork

def evaluate_with_6class_mapping(nn, X_test, y_test):
    """
    Evaluate predictions mapped to 6-class system.
    """
    # Get raw predictions
    predictions = nn.model.predict(X_test, verbose=0)
    raw_preds = np.argmax(predictions, axis=1)
    
    # Map predictions to 6 classes
    final_preds = []
    for pred_idx in raw_preds:
        final_idx, _ = nn.map_to_final_class(pred_idx)
        final_preds.append(final_idx)
    
    # Map true labels to 6 classes
    final_true = []
    for true_idx in y_test:
        if true_idx < 5:  # Main colors 0-4
            final_true.append(true_idx)
        else:
            final_true.append(5)  # "other"
    
    return np.array(final_preds), np.array(final_true)

def main():
    parser = argparse.ArgumentParser(description='Test Vehicle Color CNN Model')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test dataset folder')
    parser.add_argument('--model-path', type=str, 
                       default='results/cnn/vehicle_color_best.h5',
                       help='Path to trained model (.h5)')
    parser.add_argument('--img-size', type=int, default=128,
                       help='Image size for model input')
    
    args = parser.parse_args()
    
    # Setup
    results_dir = Path("results/cnn")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print("CNN COLOR CLASSIFICATION - TESTING (6-class)")
    print('=' * 60)
    print(f"Test directory: {args.test_dir}")
    print(f"Model: {args.model_path}")
    print('=' * 60)
    
    # 1. Load model
    print("\n1. LOADING MODEL")
    print('-' * 40)
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        exit(1)
    
    nn = VehicleColorNeuralNetwork(img_size=args.img_size)
    nn.load_model(args.model_path)
    
    print(f"Model loaded: {nn.num_classes} training classes")
    print(f"Final 6 classes: {nn.final_classes}")
    
    # 2. Load test data
    print("\n2. LOADING TEST DATA")
    print('-' * 40)
    
    X_test, y_test = nn.load_dataset(args.test_dir, max_per_class=1000)
    
    if len(X_test) == 0:
        print("ERROR: No test images loaded")
        exit(1)
    
    print(f"Test images: {len(X_test)}")
    print(f"Raw class distribution: {np.bincount(y_test)}")
    
    # 3. Evaluate with 6-class mapping
    print("\n3. 6-CLASS EVALUATION")
    print('-' * 40)
    
    final_preds, final_true = evaluate_with_6class_mapping(nn, X_test, y_test)
    
    accuracy = accuracy_score(final_true, final_preds)
    print(f"Overall Accuracy: {accuracy:.2%}")
    
    # 4. CONFUSION MATRIX - PERCENTAGE
    print("\n4. CONFUSION MATRIX (PERCENTAGE)")
    print('-' * 40)
    
    cm = confusion_matrix(final_true, final_preds, labels=range(6))
    
    # Convert to percentages (row-wise normalization)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)  # Handle division by zero
    
    # Plot percentage matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=nn.final_classes)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(f"CNN 6-Class Confusion Matrix (%)\nAccuracy: {accuracy:.2%}")
    plt.tight_layout()
    
    cm_percent_path = results_dir / "cnn_confusion_matrix_percent.png"
    plt.savefig(cm_percent_path, dpi=150)
    print(f"Saved: {cm_percent_path}")
    
    # 5. CONFUSION MATRIX - RAW COUNTS (for reference)
    print("\n5. CONFUSION MATRIX (RAW COUNTS)")
    print('-' * 40)
    
    plt.figure(figsize=(8, 6))
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nn.final_classes)
    disp2.plot(cmap="Greens", values_format="d")
    plt.title(f"CNN 6-Class Confusion Matrix (Counts)\nAccuracy: {accuracy:.2%}")
    plt.tight_layout()
    
    cm_counts_path = results_dir / "cnn_confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=150)
    print(f"Saved: {cm_counts_path}")
    
    # 6. Per-class accuracy bar chart
    print("\n6. PER-CLASS ACCURACY")
    print('-' * 40)
    
    per_class_acc = []
    sample_counts = []
    for i, color in enumerate(nn.final_classes):
        mask = final_true == i
        count = np.sum(mask)
        sample_counts.append(count)
        if count > 0:
            acc = np.sum(final_preds[mask] == i) / count
            per_class_acc.append((color, acc, count))
        else:
            per_class_acc.append((color, 0.0, 0))
    
    # Sort by accuracy
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    colors_sorted = [x[0] for x in per_class_acc]
    accs_sorted = [x[1] for x in per_class_acc]
    counts_sorted = [x[2] for x in per_class_acc]
    
    # Plot
    plt.figure(figsize=(10, 5))
    colors_bars = []
    for color in colors_sorted:
        if color == 'black': colors_bars.append('black')
        elif color == 'white': colors_bars.append('lightgray')
        elif color == 'grey': colors_bars.append('gray')
        elif color == 'blue': colors_bars.append('blue')
        elif color == 'red': colors_bars.append('red')
        else: colors_bars.append('orange')
    
    bars = plt.bar(colors_sorted, accs_sorted, color=colors_bars, edgecolor='black')
    
    # Add labels
    for bar, acc, count in zip(bars, accs_sorted, counts_sorted):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{acc:.1%}\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    plt.axhline(y=accuracy, color='red', linestyle='--', label=f'Overall: {accuracy:.2%}')
    plt.ylim(0, 1.1)
    plt.xlabel('Color Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy (6-Class System)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    chart_path = results_dir / "cnn_per_class_accuracy.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved: {chart_path}")
    
    # 7. Classification report
    print("\n7. CLASSIFICATION REPORT")
    print('-' * 40)
    
    report = classification_report(final_true, final_preds, 
                                  target_names=nn.final_classes, digits=3)
    print(report)
    
    # Save report
    report_path = results_dir / "cnn_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("CNN 6-CLASS CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set: {args.test_dir}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Total samples: {len(X_test)}\n")
        f.write(f"Overall accuracy: {accuracy:.2%}\n\n")
        f.write("Per-class accuracy:\n")
        f.write("-" * 30 + "\n")
        for color, acc, count in per_class_acc:
            f.write(f"{color:10s}: {acc:6.2%} (n={count})\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write("Detailed report:\n")
        f.write(report)
    
    print(f"Report saved: {report_path}")
    
    # 8. Save predictions for comparison
    npz_path = results_dir / "cnn_6class_predictions.npz"
    np.savez(npz_path,
             final_true=final_true,
             final_preds=final_preds,
             class_names=nn.final_classes,
             cm_percent=cm_percent,
             cm_counts=cm)
    print(f"Predictions saved: {npz_path}")
    
    # 9. Show plots
    plt.show()
    
    print('\n' + '=' * 60)
    print("TESTING COMPLETE")
    print('=' * 60)
    print(f"6-class accuracy: {accuracy:.2%}")
    print(f"\nOutput files in: {results_dir}")
    print(f"  • {cm_percent_path.name} (percentage matrix)")
    print(f"  • {cm_counts_path.name} (counts matrix)")
    print(f"  • {chart_path.name} (accuracy chart)")
    print(f"  • {report_path.name} (detailed report)")
    print('=' * 60)

if __name__ == "__main__":
    main()
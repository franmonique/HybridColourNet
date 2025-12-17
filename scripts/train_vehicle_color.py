import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from vehicle_color_model import VehicleColorNeuralNetwork

def plot_training_history(history, results_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = results_dir / "training_history.png"
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to: {history_path}")

def generate_detailed_report(nn, X_test, y_test, results_dir):
    """Generate report with 6-class evaluation."""
    # Get raw predictions
    predictions = nn.model.predict(X_test, verbose=0)
    raw_preds = np.argmax(predictions, axis=1)
    
    # Map to 6-class system
    final_preds = []
    final_true = []
    
    for pred_idx in raw_preds:
        final_idx, _ = nn.map_to_final_class(pred_idx)
        final_preds.append(final_idx)
    
    for true_idx in y_test:
        if true_idx < 5:  # 0-4 are main colors
            final_true.append(true_idx)
        else:
            final_true.append(5)  # Map to "other"
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(final_true, final_preds)
    
    # Save report
    report_path = results_dir / "test_report.txt"
    with open(report_path, 'w') as f:
        f.write("CNN COLOR CLASSIFICATION TEST REPORT (6-class)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total test samples: {len(X_test)}\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        
        f.write("Per-class performance:\n")
        f.write("-" * 40 + "\n")
        
        for color_name in nn.final_classes:
            color_idx = nn.final_mapping[color_name]
            mask = np.array(final_true) == color_idx
            if np.sum(mask) > 0:
                correct = np.sum(np.array(final_preds)[mask] == color_idx)
                total = np.sum(mask)
                acc = correct / total
                f.write(f"{color_name:10s}: {acc:6.2%} ({correct:3d}/{total:3d})\n")
    
    print(f"Detailed report saved to: {report_path}")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train Vehicle Color CNN Model')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Path to training dataset folder')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Path to validation dataset folder (optional)')
    parser.add_argument('--test-dir', type=str, default=None,
                       help='Path to test dataset folder (optional)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=128,
                       help='Image size for model input')
    parser.add_argument('--max-per-class', type=int, default=800,
                       help='Maximum images per class (for balancing)')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path("results/cnn")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print("VEHICLE COLOR NEURAL NETWORK - TRAINING")
    print('=' * 60)
    print(f"Training directory: {args.train_dir}")
    print(f"Max per class: {args.max_per_class}")
    print('=' * 60)
    
    # 1. Initialize model
    nn = VehicleColorNeuralNetwork(img_size=args.img_size)
    
    # 2. Load training dataset (with balancing)
    print("\n1. LOADING TRAINING DATASET")
    print('-' * 60)
    
    try:
        X_train_all, y_train_all = nn.load_dataset(
            args.train_dir,
            max_per_class=args.max_per_class,
            required_colors=['black', 'white', 'grey', 'blue', 'red']
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
    
    print(f"\nTraining dataset loaded: {len(X_train_all)} images")
    print(f"Training with {nn.num_classes} total classes")
    print(f"Class distribution: {np.bincount(y_train_all)}")
    
    # 3. Split data
    if args.val_dir:
        print("\n2. LOADING SEPARATE VALIDATION SET")
        print('-' * 60)
        X_val, y_val = nn.load_dataset(args.val_dir, max_per_class=args.max_per_class)
        X_train, y_train = X_train_all, y_train_all
    else:
        print("\n2. SPLITTING TRAINING DATA (80/20)")
        print('-' * 60)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
        )
    
    print(f"Train: {len(X_train)} images")
    print(f"Val:   {len(X_val)} images")
    
    # 4. Compute class weights
    print("\n3. COMPUTING CLASS WEIGHTS")
    print('-' * 60)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, weights)}
    print("Class weights:", class_weight)
    
    # 5. Build and train model
    print("\n4. BUILDING & TRAINING MODEL")
    print('-' * 60)
    nn.build_model()
    nn.model.summary()
    
    # Callbacks
    best_model_path = results_dir / "vehicle_color_best.h5"
    final_model_path = results_dir / "vehicle_color_final.h5"
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(str(best_model_path), save_best_only=True),
        keras.callbacks.CSVLogger(str(results_dir / "training_log.csv"))
    ]
    
    # Train
    history = nn.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Save final model with metadata
    nn.save_model(str(final_model_path))
    
    # Plot history
    plot_training_history(history, results_dir)
    
    # 6. Test if provided
    if args.test_dir and os.path.exists(args.test_dir):
        print("\n5. FINAL TEST EVALUATION")
        print('-' * 60)
        
        # Load best model
        nn.load_model(str(best_model_path))
        
        # Load test data
        X_test, y_test = nn.load_dataset(args.test_dir, max_per_class=1000)
        
        if len(X_test) > 0:
            # Evaluate
            loss, acc = nn.model.evaluate(X_test, y_test, verbose=0)
            print(f"Raw accuracy: {acc:.2%}")
            print(f"Loss: {loss:.4f}")
            
            # Generate 6-class report
            final_acc = generate_detailed_report(nn, X_test, y_test, results_dir)
            print(f"6-class accuracy: {final_acc:.2%}")
    
    print('\n' + '=' * 60)
    print("TRAINING COMPLETE")
    print(f"Best model:  {best_model_path}")
    print(f"Final model: {final_model_path}")
    print('=' * 60)

if __name__ == "__main__":
    main()

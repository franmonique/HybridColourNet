import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VehicleColorNeuralNetwork:
    def __init__(self, img_size=128):
        self.img_size = img_size
        self.model = None
        self.color_mapping = {}      # Built dynamically from dataset
        self.reverse_mapping = {}    # Reverse lookup
        self.num_classes = 0         # Set after loading data
        
        # Final 6-class system for inference (always fixed)
        self.final_classes = ['black', 'white', 'grey', 'blue', 'red', 'other']
        self.final_mapping = {color: idx for idx, color in enumerate(self.final_classes)}

    # ------------------------------------------------------
    # DATA LOADING (DYNAMIC CLASS DISCOVERY)
    # ------------------------------------------------------
    def load_dataset(self, dataset_path, max_per_class=1000, required_colors=None):
        """
        Load dataset with dynamic class discovery.
        
        Args:
            dataset_path: Path to dataset folder
            max_per_class: Maximum images per class (for balancing)
            required_colors: List of REQUIRED folders (default: 5 main colors)
        
        Returns:
            (images, labels)
        """
        if required_colors is None:
            required_colors = ['black', 'white', 'grey', 'blue', 'red']
        
        images = []
        labels = []
        
        # 1. Discover all color folders
        all_folders = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                all_folders.append(item)
        
        print(f"Found {len(all_folders)} color folders: {all_folders}")
        
        # 2. Verify required colors exist
        missing_required = []
        for color in required_colors:
            if color not in all_folders:
                missing_required.append(color)
        
        if missing_required:
            raise ValueError(f"Missing required color folders: {missing_required}")
        
        # 3. Build dynamic color mapping (required colors first)
        self.color_mapping = {}
        
        # Required colors get fixed indices 0-4
        for idx, color in enumerate(required_colors):
            self.color_mapping[color] = idx
        
        # Additional colors get indices after required ones
        next_idx = len(required_colors)
        for color in all_folders:
            if color not in self.color_mapping:
                self.color_mapping[color] = next_idx
                next_idx += 1
        
        self.num_classes = len(self.color_mapping)
        self.reverse_mapping = {v: k for k, v in self.color_mapping.items()}
        
        print(f"Built dynamic mapping ({self.num_classes} classes):")
        for color, idx in self.color_mapping.items():
            print(f"  {idx:2d}: {color}")
        
        # 4. Load images with balancing
        for color_name, label in self.color_mapping.items():
            color_path = os.path.join(dataset_path, color_name)
            
            image_files = [
                f for f in os.listdir(color_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            print(f"Loading {color_name:12s}: {len(image_files):4d} images available", end="")
            loaded = 0
            
            for img_file in image_files[:max_per_class]:
                img_path = os.path.join(color_path, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(label)
                    loaded += 1
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            print(f" → loaded {loaded}")
        
        if len(images) == 0:
            raise ValueError(f"No images loaded from {dataset_path}")
        
        return np.array(images), np.array(labels)

    # ------------------------------------------------------
    # MODEL ARCHITECTURE (DYNAMIC OUTPUT)
    # ------------------------------------------------------
    def build_model(self):
        """Build CNN with dynamic output based on discovered classes."""
        model = keras.Sequential([
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu',
                          input_shape=(self.img_size, self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')  # Dynamic!
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    # ------------------------------------------------------
    # INFERENCE WITH 6-CLASS MAPPING
    # ------------------------------------------------------
    def map_to_final_class(self, predicted_idx):
        """
        Map any predicted class to the final 6-class system.
        
        Rules:
        - Main 5 colors (black, white, grey, blue, red) → keep as is
        - Any other color → map to "other" (index 5)
        """
        # Get the color name from prediction
        color_name = self.reverse_mapping.get(predicted_idx, "unknown")
        
        # If it's one of the main 5, return it
        if color_name in self.final_mapping and color_name != 'other':
            return self.final_mapping[color_name], color_name
        
        # Everything else is "other"
        return 5, "other"
    
    def predict_frame(self, bgr_frame):
        """
        Predict color from BGR frame with 6-class mapping.
        
        Returns:
            (final_color_name, confidence, probability_dict)
        """
        # Preprocess
        img = cv2.resize(bgr_frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img_batch = np.expand_dims(img, axis=0)
        
        # Get raw predictions
        predictions = self.model.predict(img_batch, verbose=0)[0]
        raw_predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[raw_predicted_idx])
        
        # Map to final 6-class system
        final_idx, final_color = self.map_to_final_class(raw_predicted_idx)
        
        # Build probability dict for 6 classes
        prob_dict = {color: 0.0 for color in self.final_classes}
        
        # Sum probabilities for colors mapping to the same final class
        for idx, prob in enumerate(predictions):
            _, mapped_color = self.map_to_final_class(idx)
            prob_dict[mapped_color] += float(prob)
        
        return final_color, confidence, prob_dict

    def _preprocess_bgr_frame(self, bgr_img):
        """Utility: Preprocess single BGR image."""
        img = cv2.resize(bgr_img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    # ------------------------------------------------------
    # SAVE / LOAD (WITH METADATA)
    # ------------------------------------------------------
    def save_model(self, path='vehicle_color_model.h5'):
        """Save model with color mapping metadata."""
        # Save model weights
        self.model.save(path)
        
        # Save metadata separately
        meta_path = path.replace('.h5', '_metadata.npz')
        np.savez(meta_path,
                 color_mapping=self.color_mapping,
                 reverse_mapping=self.reverse_mapping,
                 num_classes=self.num_classes,
                 img_size=self.img_size)
        
        print(f"Model saved to {path}")
        print(f"Metadata saved to {meta_path}")

    def load_model(self, path='vehicle_color_model.h5'):
        """Load model and its metadata."""
        # Load model
        self.model = keras.models.load_model(path)
        
        # Try to load metadata
        meta_path = path.replace('.h5', '_metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            self.color_mapping = meta['color_mapping'].item()
            self.reverse_mapping = meta['reverse_mapping'].item()
            self.num_classes = int(meta['num_classes'])
            self.img_size = int(meta['img_size'])
            print(f"Loaded metadata: {self.num_classes} classes")
        else:
            print("Warning: No metadata found. Using default 6 classes.")
            self.color_mapping = self.final_mapping.copy()
            self.reverse_mapping = {v: k for k, v in self.color_mapping.items()}
            self.num_classes = 6
        
        print(f"Model loaded from {path}")
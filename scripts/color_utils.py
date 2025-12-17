import cv2
import numpy as np
from pathlib import Path

# YOLO classes we treat as vehicles
VEHICLE_CLASSES = {"car", "van", "truck", "bus"}

# Global variables for KNN data
_knn_loaded = False
_X_train = None
_y_train = None
_class_names = None


def load_knn_data(knn_data_path=None):
    """
    Load KNN color classification data from .npz file.
    
    Args:
        knn_data_path: Path to color_knn_data.npz file. If None, checks common locations.
        
    Returns:
        Tuple of (X_train, y_train, class_names)
    """
    global _knn_loaded, _X_train, _y_train, _class_names
    
    if _knn_loaded:
        return _X_train, _y_train, _class_names
    
    if knn_data_path is None:
        # Check common locations as fallback
        possible_paths = [
            Path("results/knn/color_knn_data.npz"),
            Path("results/knn/improve_knn3/color_knn_data.npz"),
            Path("pretrained/color_knn_data.npz"),
            Path("color_knn_data.npz"),
        ]
        
        for path in possible_paths:
            if path.exists():
                knn_data_path = path
                print(f"[INFO] Auto-selected KNN data: {knn_data_path}")
                break
        
        if knn_data_path is None:
            raise FileNotFoundError(
                "No KNN data file found. Please specify --knn-data argument.\n"
                "Or run build_color_knn_dataset.py to create one."
            )
    else:
        knn_data_path = Path(knn_data_path)
        if not knn_data_path.exists():
            raise FileNotFoundError(f"Specified KNN data file not found: {knn_data_path}")
    
    print(f"[INFO] Loading KNN data from: {knn_data_path}")
    data = np.load(knn_data_path, allow_pickle=True)
    _X_train = data["X"].astype(np.float32)   # shape (N, 3)
    _y_train = data["y"].astype(np.int64)     # shape (N,)
    _class_names = data["classes"].tolist()   # e.g. ["black","white","grey","blue","red","other"]
    _knn_loaded = True
    
    print(f"[INFO] KNN data loaded: {len(_X_train)} samples, {len(_class_names)} classes: {_class_names}")
    return _X_train, _y_train, _class_names


def _ensure_knn_loaded():
    """Ensure KNN data is loaded before prediction."""
    global _X_train, _y_train, _class_names
    if _X_train is None or _y_train is None or _class_names is None:
        # Use default location
        load_knn_data()


def _extract_color_feature_from_crop(crop_bgr):
    """
    mean Lab of centre region (3 floats).
    """
    h, w = crop_bgr.shape[:2]
    y1, y2 = int(0.2 * h), int(0.8 * h)
    x1, x2 = int(0.2 * w), int(0.8 * w)
    center = crop_bgr[y1:y2, x1:x2]
    if center.size == 0:
        center = crop_bgr

    lab = cv2.cvtColor(center, cv2.COLOR_BGR2LAB)
    lab_flat = lab.reshape(-1, 3)
    mean_lab = lab_flat.mean(axis=0)  # [L, a, b]
    return mean_lab.astype(np.float32)


def _knn_predict_color(feature, k=3):
    """
    Simple KNN on the tiny feature space.
    feature: np.array with shape (3,)
    """
    _ensure_knn_loaded()
    
    diff = _X_train - feature[None, :]      # (N, 3)
    dists = np.sum(diff * diff, axis=1)     # (N,)

    idx = np.argsort(dists)[:k]
    neighbor_labels = _y_train[idx]

    unique, counts = np.unique(neighbor_labels, return_counts=True)
    best_label = unique[np.argmax(counts)]
    color_name = _class_names[best_label]
    return color_name


def get_box_dominant_color_bgr(image, x1, y1, x2, y2, debug=False, tag=""):
    """
    Wrapper used by car_finder:
      - crops the YOLO box
      - extracts a tiny Lab feature
      - runs KNN to get a color name
    Returns a string like 'black', 'white', 'grey', 'blue', 'red', or 'other'.
    """
    h_img, w_img = image.shape[:2]

    x1 = max(0, min(int(x1), w_img - 1))
    x2 = max(0, min(int(x2), w_img))
    y1 = max(0, min(int(y1), h_img - 1))
    y2 = max(0, min(int(y2), h_img))

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    feature = _extract_color_feature_from_crop(crop)
    color_name = _knn_predict_color(feature, k=3)

    if debug:
        print(f"[{tag}] predicted color: {color_name}")

    return color_name

def set_knn_data_path(knn_data_path):
    """Explicitly set KNN data file path and reload."""
    global _knn_loaded
    _knn_loaded = False
    load_knn_data(knn_data_path)
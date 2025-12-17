import sys
import os

os.environ['TORCH_WEIGHTS_ONLY'] = '0' 

import torch
import torch.serialization
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("[INFO] PyTorch 2.6+ security patch applied")

import argparse
import time
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn 
from numpy import random 


LEAF_YOLO_PATH = Path(__file__).parent.parent / "LEAF-YOLO"
sys.path.insert(0, str(LEAF_YOLO_PATH))

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# CNN model
from vehicle_color_model import VehicleColorNeuralNetwork

# KNN-based color utils
from color_utils import get_box_dominant_color_bgr, VEHICLE_CLASSES, set_knn_data_path, load_knn_data

VALID_COLORS = ["black", "white", "grey", "blue", "red", "other", "all"]

# ---------------------------------------------------------
# FUSION LOGIC: CNN + KNN
# ---------------------------------------------------------
def fuse_color(cnn_color, cnn_conf, knn_color):
    """
    Combine CNN + KNN predictions for a single vehicle crop.

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
# MAIN PIPELINE
# ---------------------------------------------------------
def run_car_finder(opt):
    source = opt.source
    weights = opt.weights
    imgsz = opt.img_size

    # color filter: one of 6 classes or "all"
    color_target = opt.color.lower()
    if color_target not in VALID_COLORS:
        raise ValueError(f"--color must be one of {VALID_COLORS}")

    # -------------------------------------------------
    # Directories
    # -------------------------------------------------
    # Updated: Save to results/cnn_knn/car_finder
    project_root = Path(opt.project).parent.parent
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Init YOLO (LEAF-YOLO)
    # -------------------------------------------------
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"

    # FIXED: Handle weights as list (from nargs='+')
    if isinstance(weights, list):
        if len(weights) > 0:
            weights = weights[0]  # Take the first weight file
        else:
            raise ValueError("No weight files specified")

    # Check if weights file exists
    weights_path = Path(weights)
    
    if not weights_path.exists():
        # Try 1: Relative to LEAF-YOLO folder
        leaf_weights = LEAF_YOLO_PATH / weights
        if leaf_weights.exists():
            weights_path = leaf_weights
            print(f"[INFO] Using weights from LEAF-YOLO folder: {weights_path}")
        else:
            # Try 2: Absolute path
            abs_weights = Path(weights).absolute()
            if abs_weights.exists():
                weights_path = abs_weights
                print(f"[INFO] Using absolute path: {weights_path}")
            else:
                # Try 3: Check common locations
                possible_paths = [
                    Path("LEAF-YOLO/cfg/LEAF-YOLO/leaf-sizes/weights/best.pt"),
                    Path("weights/best.pt"),
                    Path("best.pt"),
                ]
                
                for path in possible_paths:
                    if path.exists():
                        weights_path = path
                        print(f"[INFO] Found weights at: {weights_path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"Could not find weights file: {weights}\n"
                        f"Checked:\n"
                        f"  1. {weights}\n"
                        f"  2. {leaf_weights}\n"
                        f"  3. Common locations"
                    )
    
    print(f"[INFO] Loading YOLO model from: {weights_path}")
    model = attempt_load(str(weights_path), map_location=device)
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half()

    names = model.module.names if hasattr(model, "module") else model.names

    default_box_color = (255, 0, 255)

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz)
            .to(device)
            .type_as(next(model.parameters()))
        )

    # -------------------------------------------------
    # Init COLOR CNN
    # -------------------------------------------------
    color_nn = VehicleColorNeuralNetwork(img_size=128)
    
    if not Path(opt.color_model).exists():
        cnn_path = Path("results/cnn/vehicle_color_best.h5")
        if cnn_path.exists():
            opt.color_model = str(cnn_path)
        else:
            raise FileNotFoundError(f"Could not find CNN model: {opt.color_model}")
    
    color_nn.load_model(opt.color_model)

    # -------------------------------------------------
    # Init COLOR KNN
    # -------------------------------------------------
    if opt.knn_data:
        print(f"[INFO] Using specified KNN data: {opt.knn_data}")
        set_knn_data_path(opt.knn_data)
    else:
        print("[INFO] Auto-loading KNN data...")
        load_knn_data()

    # -------------------------------------------------
    # Data loader (images / folder / video)
    # -------------------------------------------------
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    t0 = time.time()

    # --- VIDEO WRITER STATE ---
    vid_writer = None
    vid_path = None

    for path, img, im0s, vid_cap in dataset:
        # img is letterboxed, im0s is original BGR
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img_tensor, augment=opt.augment)[0]
        t2 = time_synchronized()

        # NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        p, im0 = path, im0s.copy()
        p = Path(p)

        # Decide output mode: image vs video
        is_video = vid_cap is not None

        if is_video:
            # Create VideoWriter once per video source
            if vid_writer is None:
                vid_path = str(save_dir / f"{p.stem}_annotated.mp4")
                fps = vid_cap.get(cv2.CAP_PROP_FPS) or 25
                h, w = im0.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))
        else:
            # For images, keep per-file output
            save_path = str(save_dir / p.name)

        # Per-class counters for IDs (CAR01, TRUCK02, ...)
        id_counters = {}

        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes to original image size
                det[:, :4] = scale_coords(
                    img_tensor.shape[2:], det[:, :4], im0.shape
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    cls_id = int(cls)
                    cls_name = names[cls_id]

                    # Filter: only run on vehicle classes
                    if cls_name not in VEHICLE_CLASSES:
                        continue

                    # Crop vehicle region from original frame
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im0.shape[1], x2)
                    y2 = min(im0.shape[0], y2)
                    crop = im0[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    # ----- CNN prediction -----
                    cnn_color, cnn_conf, _ = color_nn.predict_frame(crop)

                    # ----- KNN prediction -----
                    knn_color = get_box_dominant_color_bgr(
                        im0, x1, y1, x2, y2, debug=False, tag=cls_name
                    )

                    # Map unknown KNN colors to "other"
                    if knn_color not in VALID_COLORS[:-1]:  # exclude "all"
                        knn_color = "other"

                    # ----- Fusion -----
                    final_color, source = fuse_color(cnn_color, cnn_conf, knn_color)

                    # Apply color filter (if not "all")
                    if color_target != "all" and final_color != color_target:
                        continue

                    # Build ID: e.g. CAR01, TRUCK02
                    prefix = cls_name.upper()
                    id_counters.setdefault(prefix, 0)
                    id_counters[prefix] += 1
                    vehicle_id = f"{prefix}{id_counters[prefix]:02d}"

                    print(
                        f"[{vehicle_id}] class={cls_name}, "
                        f"CNN={cnn_color} ({cnn_conf*100:.1f}%), "
                        f"KNN={knn_color}, "
                        f"FINAL={final_color} (from {source}), "
                        f"YOLO_conf={conf:.2f}"
                    )

                    # Box color based on source
                    if source == "cnn":
                        box_color = (255, 0, 255)   # CNN → magenta
                    else:
                        box_color = (128, 0, 0)     # KNN → darker red

                    label = f"{vehicle_id}: {final_color}"
                    plot_one_box(
                        xyxy, im0, label=label, color=box_color, line_thickness=2
                    )

        # --- SAVE OUTPUT ---
        if is_video:
            # Write current frame to video
            vid_writer.write(im0)
            print(
                f"Video frame processed | "
                f"Inference: {(1E3 * (t2 - t1)):.1f}ms, "
                f"NMS: {(1E3 * (t3 - t2)):.1f}ms"
            )
        else:
            # Save annotated image (even if no boxes drawn)
            cv2.imwrite(save_path, im0)
            print(
                f"Saved: {save_path} | "
                f"Inference: {(1E3 * (t2 - t1)):.1f}ms, "
                f"NMS: {(1E3 * (t3 - t2)):.1f}ms"
            )

    # Release video writer if used
    if vid_writer is not None:
        vid_writer.release()
        print(f"Saved video to: {vid_path}")

    print(f"Done. ({time.time() - t0:.3f}s total)")


# ---------------------------------------------------------
# ARG PARSER
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Find vehicles and classify their colors using LEAF-YOLO + CNN + KNN fusion."
    )
    parser.add_argument(
        "--weights", nargs="+", type=str,
        default="cfg/LEAF-YOLO/leaf-sizes/weights/best.pt",
        help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="inference/images",
        help="source image/folder or video file"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25,
        help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45,
        help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--classes", nargs="+", type=int,
        help="filter by class index (YOLO class IDs)"
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true",
        help="class-agnostic NMS"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="augmented inference"
    )
    parser.add_argument(
        "--project", default="results/cnn_knn/car_finder",
        help="save results to project/name"
    )
    parser.add_argument(
        "--name", default="exp",
        help="save results to project/name"
    )
    parser.add_argument(
        "--exist-ok", action="store_true",
        help="existing project/name ok, do not increment"
    )
    parser.add_argument(
        "--color-model", type=str,
        default="results/cnn/vehicle_color_best.h5",
        help="path to trained CNN color model (.h5)"
    )
    parser.add_argument(
        "--knn-data", type=str, default=None,
        help="path to KNN data file (color_knn_data.npz). If not specified, checks common locations."
    )
    parser.add_argument(
        "--color", type=str, default="all",
        help="target color: black/white/grey/blue/red/other/all"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    print(opt)
    with torch.no_grad():
        run_car_finder(opt)
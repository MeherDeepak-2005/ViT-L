import json
import numpy as np
from pathlib import Path


def load_megadetector_results(json_path: str, base_dir: str = '') -> dict:
    """
    Parses a MegaDetector output JSON and returns a dict mapping
    each image filepath → best animal bbox, or None if no animal detected.

    bbox format: [xmin, ymin, width, height] normalised to 0-1

    Usage:
        bbox_map = load_megadetector_results("./data/megadetector_train.json")
        bbox = bbox_map.get("data/train_features/Z01234.jpg")  # None if blank
    """
    with open(json_path) as f:
        data = json.load(f)

    bbox_map = {}
    for img in data['images']:
        filepath = img['file']

        # Prepend base_dir so keys match your full file paths
        if base_dir:
            filepath = f"{base_dir}/{filepath}"

        detections = img.get('detections', [])

        # Category '1' = animal in MegaDetector's output
        animal_dets = [
            d for d in detections
            if d['category'] == '1' and d['conf'] > 0.1
        ]

        if animal_dets:
            # Take the highest-confidence animal detection
            best = max(animal_dets, key=lambda d: d['conf'])
            bbox_map[Path(filepath).name] = best['bbox']  # [xmin, ymin, w, h] normalised
        else:
            bbox_map[Path(filepath).name] = None  # blank or vehicle/human only

    n_with_bbox = sum(1 for v in bbox_map.values() if v is not None)
    n_without_bbox = len(bbox_map) - n_with_bbox
    print(f"  bbox_map: {n_with_bbox} with animal detection, "
          f"{n_without_bbox} blank/no detection")

    return bbox_map


def crop_to_bbox(img_np: np.ndarray, bbox, margin: float = 0.15) -> np.ndarray:
    """
    Crops image to the bounding box with a margin for context.
    If bbox is None (blank image), returns the original image unchanged.

    img_np : HWC uint8 numpy array
    bbox   : [xmin, ymin, width, height] normalised 0-1, or None
    margin : fraction of bbox size to add as padding on each side
    """
    if bbox is None:
        return img_np

    H, W = img_np.shape[:2]
    xmin, ymin, bw, bh = bbox

    # Convert normalised coords to pixels
    x1 = int(xmin * W)
    y1 = int(ymin * H)
    x2 = int((xmin + bw) * W)
    y2 = int((ymin + bh) * H)

    # Add margin so animal has some background context
    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)

    return img_np[y1:y2, x1:x2]

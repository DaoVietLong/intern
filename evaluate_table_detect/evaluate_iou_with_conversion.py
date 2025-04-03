from pathlib import Path
import json
from typing import List, Tuple
import argparse
import numpy as np
import json

def convert_bbox_percentage_to_absolute(bbox: dict) -> List[float]:
    x1 = bbox["x"] * bbox["original_width"] / 100
    y1 = bbox["y"] * bbox["original_height"] / 100
    x2 = x1 + (bbox["width"] * bbox["original_width"] / 100)
    y2 = y1 + (bbox["height"] * bbox["original_height"] / 100)
    return [x1, y1, x2, y2]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

def evaluate_iou_to_json(ground_truth_file: str, prediction_dir: str, iou_threshold: float = 0.5, output_path: str = "iou_results.json"):
    with open(ground_truth_file, 'r') as f:
        ground_truth_list = json.load(f)

    iou_values = []
    results = {}

    for gt_entry in ground_truth_list:
        image_path = Path(gt_entry["image"])
        image_name = image_path.name.split("-", 1)[-1]
        labels = gt_entry.get("label", [])
        gt_bboxes = [
            convert_bbox_percentage_to_absolute(label)
            for label in labels if "rectanglelabels" in label and "Table" in label["rectanglelabels"]
        ]

        pred_file = Path(prediction_dir) / image_name.replace(".jpg", "_objects.json")
        if not pred_file.exists():
            print(f"[!] Prediction file not found: {pred_file}")
            results[image_name] = []
            continue

        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        pred_bboxes = [p["bbox"] for p in pred_data if p["label"] == "table"]

        used_preds = set()
        image_iou_scores = []

        for gt_box in gt_bboxes:
            best_iou = 0
            best_idx = -1
            for i, pred_box in enumerate(pred_bboxes):
                if i in used_preds:
                    continue
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold:
                image_iou_scores.append(round(best_iou, 4))
                iou_values.append(best_iou)
                used_preds.add(best_idx)

        results[image_name] = image_iou_scores

    mAP = round(np.mean(iou_values), 4) if iou_values else 0.0
    results["mAP"] = mAP

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[✓] IoU results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_iou_file", default="iou_per_image.txt")
    parser.add_argument("--output_file", default="iou_results.txt", help="Path to save per-image IoU and mAP.")
    args = parser.parse_args()

    evaluate_iou_to_json(args.ground_truth, args.prediction_dir, args.iou_threshold, args.output_file)


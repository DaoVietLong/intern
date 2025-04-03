import json
from pathlib import Path
from typing import List
import argparse
from PIL import Image
import shutil

def convert_relative_bbox_to_absolute(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    x1 = bbox[0] * image_width
    y1 = bbox[1] * image_height
    x2 = bbox[2] * image_width
    y2 = bbox[3] * image_height
    return [x1, y1, x2, y2]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    print("inter area: ", inter_area)
    union_area = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
                  (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area)
    print("union_area: ", union_area)
    return inter_area / union_area

def evaluate_single_image(gt_file: Path, pred_file: Path, image_file: Path, iou_threshold: float = 0.5) -> dict:
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with Image.open(image_file) as img:
        width, height = img.size

    gt_bboxes = [convert_relative_bbox_to_absolute(b, width, height) for b in gt_data.get("TD", [])]
    pred_bboxes = [p["bbox"] for p in pred_data if (p["label"].lower() == "table" or p["label"].lower() == "table rotated")]

    print("gt_bboxes: ", gt_bboxes)
    print("pred_bboxes: ", pred_bboxes)
    used_preds = set()
    ious = []

    for gt in gt_bboxes:
        best_iou = 0
        best_idx = -1
        for i, pred in enumerate(pred_bboxes):
            if i in used_preds:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold:
            used_preds.add(best_idx)
            ious.append(best_iou)

    score = sum(ious) / len(gt_bboxes) if gt_bboxes else 0.0
    return {"iou": score}

def main(gt_dir: str, pred_dir: str, image_dir: str, iou_threshold: float, output_dir: str):
    gt_files = sorted(Path(gt_dir).glob("*_tables_gt.json"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    all_ious = []

    for gt_file in gt_files:
        image_id = gt_file.stem.replace("_tables_gt", "")
        pred_file = Path(pred_dir) / f"{image_id}_objects.json"
        image_file = Path(image_dir) / f"{image_id}.jpg"

        print(image_file)
        if not pred_file.exists() or not image_file.exists():
            print(f"[!] Missing file for {image_id}")
            continue

        result = evaluate_single_image(gt_file, pred_file, image_file, iou_threshold)
        iou_score = result["iou"]
        all_ious.append(iou_score)
        results[image_id] = {"iou": iou_score}

        # Categorize into bins
        bin_index = int(iou_score * 100) // 10
        bin_index = min(bin_index, 9)
        bin_folder = output_path / f"{bin_index*10:02d}-{(bin_index+1)*10:02d}%"
        bin_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy(gt_file, bin_folder / gt_file.name)
        shutil.copy(pred_file, bin_folder / pred_file.name)
        shutil.copy(image_file, bin_folder / image_file.name)

    results["mAP"] = sum(all_ious) / len(all_ious) if all_ious else 0.0

    output_json = output_path / "iou_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_json}")
    print(f"Overall mAP: {results['mAP']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Directory with *_tables_gt.json files.")
    parser.add_argument("--pred_dir", required=True, help="Directory with *_objects.json prediction files.")
    parser.add_argument("--image_dir", required=True, help="Directory containing original .jpg images.")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True, help="Directory to store output JSON and 10 folder bins.")
    args = parser.parse_args()

    main(args.gt_dir, args.pred_dir, args.image_dir, args.iou_threshold, args.output_dir)

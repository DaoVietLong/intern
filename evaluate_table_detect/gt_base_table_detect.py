import json
import cv2
import os
from pathlib import Path

def draw_bboxes_on_image(image_path, gt_path, output_path):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Failed to load image: {image_path}")
        return

    height, width = img.shape[:2]

    # Load ground truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    if "TD" not in gt_data:
        print(f"[!] 'TD' field not found in: {gt_path}")
        return

    for rel_box in gt_data["TD"]:
        x1 = int(rel_box[0] * width)
        y1 = int(rel_box[1] * height)
        x2 = int(rel_box[2] * width)
        y2 = int(rel_box[3] * height)

        # Draw rectangle with red border
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    # Save output
    cv2.imwrite(str(output_path), img)
    print(f"[✓] Saved visualization to {output_path}")

def draw_all_tables(gt_dir, image_dir, output_dir):
    gt_dir = Path(gt_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for gt_file in gt_dir.glob("*_tables_gt.json"):
        image_id = gt_file.stem.replace("_tables_gt", "")
        image_path = image_dir / f"{image_id}.jpg"
        output_path = output_dir / f"{image_id}_with_gt.jpg"
        draw_bboxes_on_image(image_path, gt_file, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Directory containing *_tables_gt.json files")
    parser.add_argument("--image_dir", required=True, help="Directory containing the original .jpg images")
    parser.add_argument("--output_dir", required=True, help="Where to save the visualized output")
    args = parser.parse_args()

    draw_all_tables(args.gt_dir, args.image_dir, args.output_dir)

import argparse
from pathlib import Path
import cv2
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load pretrained OCR model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def draw_boxes_and_export_json(image_path, image_out_path, json_out_path):
  doc = DocumentFile.from_images(str(image_path))
  result = model(doc)

  image = cv2.imread(str(image_path))
  h, w = image.shape[:2]
  annotations = []

  for block in result.pages[0].blocks:
    for line in block.lines:
      for word in line.words:
        ((x_min, y_min), (x_max, y_max)) = word.geometry
        abs_xmin = int(x_min * w)
        abs_ymin = int(y_min * h)
        abs_xmax = int(x_max * w)
        abs_ymax = int(y_max * h)

        # Draw boxes and labels
        cv2.rectangle(image, (abs_xmin, abs_ymin), (abs_xmax, abs_ymax), (0, 255, 0), 2)
        cv2.putText(image, word.value, (abs_xmin, abs_ymin - 5),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        annotations.append({
          "value": word.value,
          "bbox_relative": [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)],
          "bbox_absolute": [abs_xmin, abs_ymin, abs_xmax, abs_ymax]
        })

  # Save image and JSON
  cv2.imwrite(str(image_out_path), image)
  with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump({"tokens": annotations}, f, ensure_ascii=False, indent=2)

  print(f" {image_path.name} → image: {image_out_path.name}, json: {json_out_path.name}")

def process_images(input_dir, output_dir):
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)

  images_dir = output_dir / "images"
  annotations_dir = output_dir / "annotations"
  images_dir.mkdir(parents=True, exist_ok=True)
  annotations_dir.mkdir(parents=True, exist_ok=True)

  for image_path in input_dir.glob("*.*"):
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
      continue

    out_image_path = images_dir / f"{image_path.stem}_docTR.jpg"
    out_json_path = annotations_dir / f"{image_path.stem}_docTR.json"

    draw_boxes_and_export_json(image_path, out_image_path, out_json_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DocTR OCR with bounding boxes + JSON output")
  parser.add_argument('--input_dir', required=True, help="Folder with input images")
  parser.add_argument('--out_dir', required=True, help="Output root directory (will contain images/ and annotations/)")
  args = parser.parse_args()

  process_images(args.input_dir, args.out_dir)

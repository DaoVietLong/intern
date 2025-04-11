import os
import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image
import argparse

def create_pascal_voc_annotation(image_path: Path, bboxes: list, out_dir: Path):
    image = Image.open(image_path)
    width, height = image.size
    filename = image_path.name
    folder = image_path.parent.name

    annotation = Element('annotation')
    SubElement(annotation, 'folder').text = folder
    SubElement(annotation, 'filename').text = filename
    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'  # Assuming RGB images

    for bbox in bboxes:
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox
        x1 = int(rel_x1 * width)
        y1 = int(rel_y1 * height)
        x2 = int(rel_x2 * width)
        y2 = int(rel_y2 * height)

        obj = SubElement(annotation, 'object')
        SubElement(obj, 'name').text = 'table'
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(x1)
        SubElement(bndbox, 'ymin').text = str(y1)
        SubElement(bndbox, 'xmax').text = str(x2)
        SubElement(bndbox, 'ymax').text = str(y2)

    xml_str = parseString(tostring(annotation)).toprettyxml(indent="  ")
    out_path = out_dir / f"{image_path.stem}.xml"
    with open(out_path, 'w') as f:
        f.write(xml_str)
    print(f"Saved {out_path} successfully.")

def main(gt_dir: str, image_dir: str, out_dir: str):
    gt_dir = Path(gt_dir)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gt_file in sorted(gt_dir.glob("*_tables_gt.json")):
        image_id = gt_file.stem.replace("_tables_gt", "")
        image_path = image_dir / f"{image_id}.jpg"
        
        if not image_path.exists():
            print(f"Image file {image_path} not found for {gt_file}.")
            continue

        with open(gt_file, 'r') as f:
            gt_data = json.load(f)

        bboxes = gt_data.get("TD", [])
        if not bboxes:
            print(f"No bounding boxes found in {gt_file}.")
            continue
        create_pascal_voc_annotation(image_path, bboxes, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON annotations to Pascal VOC XML format.")
    parser.add_argument("--gt_json", required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--image_dir", required=True, help="Directory containing the images.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output XML files.")
    args = parser.parse_args()

    main(args.gt_json, args.image_dir, args.out_dir)
import os
import json
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from pathlib import Path
from PIL import Image
import argparse

def convert_to_voc(image_path: Path, json_path: Path, out_path: Path):
    image = Image.open(image_path)
    width, height = image.size

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, "filename").text = image_path.name

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3" # Assuming RGB images

    for obj in annotations:
        if obj["label"].lower() != "table" or obj["label"].lower() != "table rotated":
            continue
        x1, y1, x2, y2 = map(int, obj["bbox"])

        obj_tag = ET.SubElement(annotation, "object")
        ET.SubElement(obj_tag, "name").text = obj["label"]
        bndbox = ET.SubElement(obj_tag, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)

    xml_str = parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    with open(out_path, 'w') as f:
        f.write(xml_str)

    print(f"Saved {out_path} successfully.")

def batch_convert(input_dir: str, out_dir: str):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for json_file in input_dir.glob("*_objects.json"):
        base_name = json_file.stem.replace("_objects", "")
        image_file = input_dir / f"{base_name}_fig_tables.jpg"

        if not image_file.exists():
            print(f"Image file {image_file} not found for {json_file}.")
            continue

        output_xml = out_dir / f"{base_name}.xml"
        convert_to_voc(image_file, json_file, output_xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON annotations to VOC XML format.")
    parser.add_argument("--input_dir", required=True, help="Directory containing JSON files.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output XML files.")
    args = parser.parse_args()

    batch_convert(args.input_dir, args.out_dir)
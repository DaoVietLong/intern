import cv2
import pytesseract
import argparse
import os
from pathlib import Path

def draw_token_boxes(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to RGB (pytesseract uses RGB format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pytesseract to get bounding boxes
    boxes = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(boxes['level'])

    # Draw boxes around each character
    for i in range(n_boxes):
        x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
        text = boxes['text'][i].strip()
        if text:
            # Draw a rectangle around the character
            cv2.rectangle(image, (x, h - y), (x + w, h - (y + h)), (0, 255, 0), 2)
            # Put the character text above the rectangle
            cv2.putText(image, text, (x, h - y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save the output image
    cv2.imwrite(output_path, image)

def process_directory(input_dir, output_dir):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            output_path = output_dir / f"{image_path.stem}_boxes{image_path.suffix}"
            # Process the image and draw token boxes
            draw_token_boxes(image_path, output_path)
            print(f"Processed {image_path.name} -> {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw token boxes on images using Tesseract OCR.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--out_dir", required=True, help="Directory to save output images with token boxes.")
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.out_dir)
# Example usage:
# python ocr_test.py /path/to/input/images /path/to/output/images
# Note: Ensure that Tesseract OCR is installed and pytesseract is configured correctly.
# Ensure you have the required libraries installed:

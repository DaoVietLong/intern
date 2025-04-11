import os
import json
import argparse
import pytesseract
from PIL import Image
import subprocess

def extract_ocr_tokens(image_path):
    image = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    tokens = []
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        
        if text:
            tokens.append({
                'text': text,
                'bbox': [
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i]
                ]
            })

    return tokens

def detect_table_llama(tokens):
    prompt = f"""You are a document structure analyst. Here are OCR tokens from a scanned page
Each token has text and a bounding box [x1, y1, x2, y2].

Your task is to identify the table region.
Return the table bounding box as [x1, y1, x2, y2].

Tokens:
{json.dumps(tokens[:100], indent=2)}

Return result in JSON format:

{{
    "table_bbox": [x1, y1, x2, y2]
}}"""
    
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3'],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output_text = result.stdout.decode()
        json_start = output_text.find('{')
        json_data = json.loads(output_text[json_start:])
        return json_data
    except Exception as e:
        return {"error": str(e)}
    
def main(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in images:
        image_path = os.path.join(input_dir, image_name)
        print(f"Processing {image_name}...")

        tokens = extract_ocr_tokens(image_path)
        result = detect_table_llama(tokens)

        output_path = os.path.join(out_dir, os.path.splitext(image_name[0] + ".json"))
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table detection using OCR + LLaMa")
    parser.add_argument("--input_dir", required=True, help="Path to directory contain images")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    main(args.input_dir, args.out_dir)
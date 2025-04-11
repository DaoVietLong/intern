import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict

def load_tokens(json_path: Path) -> List[Dict]:
    """Load tokens from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data['tokens']

def build_prompt(tokens: List[Dict]) -> str:
    prompt = (
        "You are an AI model that analyzes document layouts.\n"
        "Here is a list of OCR tokens, each with a bounding box in pixel coordinates.\n"
        "Your task is to classify each token as either being a part of a table or not.\n"
        "Return a JSON list with this format for each token:\n"
        "{\"value\": ..., \"bbox_absolute\": [...], \"in_table\": true/false}\n\n"
        "Tokens:\n"
        "Return only the JSON list, without any additional text.\n"
    )

    for t in tokens:
        prompt += f"- {{ \"value\": \"{t['value']}\", \"bbox_absolute\": {t['bbox_absolute']} }}\n"

        return prompt
    
def call_llama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return result.stdout.strip()

def process_json(json_path: Path, output_path: Path):
    tokens = load_tokens(json_path)
    prompt = build_prompt(tokens)
    print(f" Sending {json_path.name} to Llama3...")

    llama_response = call_llama(prompt)

    try:
        annotated_tokens = json.loads(llama_response)
    except json.JSONDecodeError:
        print("Llama3 response is not valid JSON. Dumping raw output.")
        output_path.write_text(llama_response, encoding="utf-8")
        return
    
    # Save the annotated tokens to a new JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated_tokens, f, ensure_ascii=False, indent=2)

    print(f" Annotated file saved to {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files with Llama3.")
    parser.add_argument("--json_dir", required=True, help="Directory containing input JSON files.")
    parser.add_argument("--out_dir", required=True, help="Directory to save output JSON files.")
    
    args = parser.parse_args()
    
    json_dir = Path(args.json_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in json_dir.glob("*.json"):
        output_path = output_dir / f"{json_path.stem}_annotated.json"
        process_json(json_path, output_path)
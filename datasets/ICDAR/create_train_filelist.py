import argparse
from pathlib import Path

def create_filelist(input_dir: str, out_file: str):
    input_path = Path(input_dir)
    out_path = Path(out_file)
    out_path.mkdir(parents=True, exist_ok=True)

    filelist_path = out_path / 'train_filelist.txt'
    image_stems = sorted([f.stem for f in input_path.glob('*.jpg')])

    with open(filelist_path, 'w') as f:
        for name in image_stems:
            f.write(name + "\n")

    print(f"Saved {len(image_stems)} entries to {filelist_path}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create a filelist of image names.")
    argparser.add_argument("--input_dir", required=True, help="Directory containing the images.")
    argparser.add_argument("--out_dir", required=True, help="Directory to save the filelist.")
    args = argparser.parse_args()

    create_filelist(args.input_dir, args.out_dir)

# Example usage:
# python create_train_filelist.py --input_dir /path/to/images --out_dir /path/to/output
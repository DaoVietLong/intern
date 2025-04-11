import shutil
import random
from pathlib import Path
import argparse

def create_val_split(train_dir: str, val_dir: str, split_ratio: float = 0.2, seed: int = 42):
  train_path = Path(train_dir)
  val_path = Path(val_dir)
  val_path.mkdir(parents=True, exist_ok=True)

  # Get all .jpg images with corresponding .xml files
  image_files = sorted(train_path.glob("*.jpg"))
  paired_files = [img for img in image_files if (train_path / f"{img.stem}.xml").exists()]

  # Shuffle and split
  random.seed(seed)
  val_samples = random.sample(paired_files, int(len(paired_files) * split_ratio))

  # Write val_filelist.txt
  val_filelist_path = val_path / "val_filelist.txt"
  with open(val_filelist_path, "w") as f:
    for img in val_samples:
      base = img.stem
      shutil.copy(train_path / f"{base}.jpg", val_path / f"{base}.jpg")
      shutil.copy(train_path / f"{base}.xml", val_path / f"{base}.xml")
      f.write(base + "\n")

  print(f"[✓] {len(val_samples)} validation images copied to {val_path}")
  print(f"[ ] val_filelist.txt created with {val_filelist_path.name}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_dir", required=True, help="Path to the train/ folder")
  parser.add_argument("--val_dir", required=True, help="Where to create the val/ folder")
  parser.add_argument("--split_ratio", type=float, default=0.2, help="Fraction of training data to use for validation")
  args = parser.parse_args()

  create_val_split(args.train_dir, args.val_dir, args.split_ratio)

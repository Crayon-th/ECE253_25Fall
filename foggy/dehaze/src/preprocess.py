"""Preprocess hazy images to a unified format for downstream dehazing experiments.

Example (sanity check only):
    python src/preprocess.py --in data/sots_o/hazy --out outputs/original
The main evaluation should still rely on images placed under data/real.
"""

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Supported extensions for image discovery.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _iter_images(input_dir: Path) -> Iterable[Path]:
    for item in sorted(input_dir.iterdir()):
        if item.is_file() and _is_image_file(item):
            yield item


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_current = max(h, w)
    if long_current == 0:
        raise ValueError("Encountered empty image during resizing.")
    if long_current == long_side:
        return image
    scale = long_side / float(long_current)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    return resized


def _save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr8):
        raise IOError(f"Failed to write image: {path}")


def preprocess_folder(input_dir: Path, output_dir: Path, long_side: int) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for img_path in _iter_images(input_dir):
        try:
            rgb = _load_rgb(img_path)
            rgb = _resize_long_side(rgb, long_side)
            out_name = img_path.stem + ".png"
            _save_png(rgb, output_dir / out_name)
            total += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping {img_path}: {exc}")
    print(f"Preprocessed {total} images from {input_dir} to {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize and standardize hazy images.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input image directory")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for PNGs")
    parser.add_argument("--long_side", type=int, default=1024, help="Target long side in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_folder(Path(args.input_dir), Path(args.output_dir), args.long_side)


if __name__ == "__main__":
    main()

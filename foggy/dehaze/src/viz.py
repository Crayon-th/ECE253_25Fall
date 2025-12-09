"""Create side-by-side visualization grids for Original/DCP/AOD-Net."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _gather_examples(orig_dir: Path, dcp_dir: Path, aod_dir: Path, max_examples: int) -> List[Tuple[str, Path, Path, Path]]:
    examples: List[Tuple[str, Path, Path, Path]] = []
    for orig_path in sorted(orig_dir.iterdir()):
        if not orig_path.is_file() or not _is_image_file(orig_path):
            continue
        name = orig_path.name
        dcp_path = dcp_dir / name
        aod_path = aod_dir / name
        if not (dcp_path.is_file() and aod_path.is_file()):
            continue
        examples.append((name, orig_path, dcp_path, aod_path))
        if len(examples) >= max_examples:
            break
    return examples


def _load_scores(csv_path: Path) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    if not csv_path.is_file():
        return scores
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = row.get("filename")
            method = row.get("method")
            niqe_str = row.get("niqe")
            if not filename or not method or not niqe_str:
                continue
            try:
                scores[(filename, method)] = float(niqe_str)
            except ValueError:
                continue
    return scores


def create_grid(orig_dir: Path, dcp_dir: Path, aod_dir: Path, out_dir: Path, max_examples: int, niqe_csv: Path) -> Path:
    examples = _gather_examples(orig_dir, dcp_dir, aod_dir, max_examples)
    if not examples:
        raise RuntimeError("No valid triplets (Original/DCP/AOD-Net) found for visualization.")
    scores = _load_scores(niqe_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["Original", "DCP", "AOD-Net"]
    rows = len(examples)
    figsize = (4 * len(columns), 3 * rows)
    fig, axes = plt.subplots(rows, len(columns), figsize=figsize)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, (name, orig_path, dcp_path, aod_path) in enumerate(examples):
        images = [_load_rgb(orig_path), _load_rgb(dcp_path), _load_rgb(aod_path)]
        name_base = Path(name).stem
        for col_idx, (image, title) in enumerate(zip(images, columns)):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(image, 0.0, 1.0))
            ax.set_title(title)
            ax.axis("off")
            score = scores.get((name, title.lower().replace("-", "")))
            text = name_base
            if score is not None:
                text = f"{text}\nNIQE {score:.2f}"
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=3),
            )
    fig.tight_layout()
    output_path = out_dir / "dehaze_grid.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization grid to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dehazing results side by side")
    parser.add_argument("--orig", required=True, help="Directory with preprocessed originals")
    parser.add_argument("--dcp", required=True, help="Directory with DCP results")
    parser.add_argument("--aod", required=True, help="Directory with AOD-Net results")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    parser.add_argument("--max_examples", type=int, default=8, help="Maximum number of examples to plot")
    parser.add_argument("--niqe_csv", default="niqe_results.csv", help="Optional NIQE CSV for overlays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_grid(
        Path(args.orig),
        Path(args.dcp),
        Path(args.aod),
        Path(args.out),
        args.max_examples,
        Path(args.niqe_csv),
    )


if __name__ == "__main__":
    main()

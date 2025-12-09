"""Compute NIQE scores for Original, DCP, and AOD-Net results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

_TORCH_NIQE_CALLABLE: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None
_LOCAL_NIQE_FN: Optional[Callable[[np.ndarray], float]] = None

try:  # Preferred fast path (piq exposes niqe function)
    from piq import niqe as _piq_niqe  # type: ignore

    def _NIQE_TORCH_WRAPPER(tensor: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        return _piq_niqe(tensor, data_range=data_range)  # type: ignore

    _TORCH_NIQE_CALLABLE = _NIQE_TORCH_WRAPPER

except ImportError:
    try:  # piq versions exposing NIQE class
        from piq import NIQE as _PiqNiqeClass  # type: ignore

        _PIQ_NIQE_INSTANCE = _PiqNiqeClass(data_range=1.0)

        def _NIQE_TORCH_WRAPPER(tensor: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
            return _PIQ_NIQE_INSTANCE(tensor)

        _TORCH_NIQE_CALLABLE = _NIQE_TORCH_WRAPPER

    except ImportError:
        try:  # piqa stores NIQE in metrics.noise
            from piqa.metrics.noise import NIQE as _PiqaNiqeClass  # type: ignore

            _PIQA_NIQE_INSTANCE = _PiqaNiqeClass()

            def _NIQE_TORCH_WRAPPER(tensor: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
                return _PIQA_NIQE_INSTANCE(tensor)

            _TORCH_NIQE_CALLABLE = _NIQE_TORCH_WRAPPER

        except ImportError:
            _TORCH_NIQE_CALLABLE = None

if _TORCH_NIQE_CALLABLE is None:
    from niqe_fallback import compute_niqe_score as _compute_niqe_locally

    def _LOCAL_NIQE_FN(gray: np.ndarray) -> float:
        return float(_compute_niqe_locally(gray))

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def _compute_niqe(image: np.ndarray) -> float:
    gray = _to_gray(image)
    if _TORCH_NIQE_CALLABLE is not None:
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        score = _TORCH_NIQE_CALLABLE(tensor, data_range=1.0)
        return float(score.item())
    if _LOCAL_NIQE_FN is None:
        raise RuntimeError("NIQE fallback implementation is unavailable.")
    return _LOCAL_NIQE_FN(gray)


def _gather_filenames(folders: Iterable[Path]) -> List[str]:
    names = set()
    for folder in folders:
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and _is_image_file(path):
                    names.add(path.name)
    return sorted(names)


def evaluate(orig_dir: Path, dcp_dir: Path, aod_dir: Path, csv_path: Path, summary_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    methods: Dict[str, Path] = {"original": orig_dir, "dcp": dcp_dir, "aodnet": aod_dir}
    filenames = _gather_filenames(methods.values())
    rows: List[Tuple[str, str, float]] = []
    stats: Dict[str, List[float]] = defaultdict(list)

    for name in filenames:
        for method, folder in methods.items():
            img_path = folder / name
            if not img_path.is_file() or not _is_image_file(img_path):
                continue
            try:
                score = _compute_niqe(_load_rgb(img_path))
                rows.append((name, method, score))
                stats[method].append(score)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] NIQE failed for {img_path}: {exc}")

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "method", "niqe"])
        for filename, method, score in rows:
            writer.writerow([filename, method, f"{score:.6f}"])

    summary_lines = []
    for method in ("original", "dcp", "aodnet"):
        values = stats.get(method, [])
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values))
            line = f"{method}: mean={mean:.4f}, std={std:.4f}, n={len(values)}"
        else:
            line = f"{method}: no images evaluated"
        summary_lines.append(line)

    with summary_path.open("w") as summary_file:
        summary_file.write("\n".join(summary_lines))

    print("NIQE summary:")
    for line in summary_lines:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NIQE for multiple dehazing methods")
    parser.add_argument("--orig", required=True, help="Directory with preprocessed originals")
    parser.add_argument("--dcp", required=True, help="Directory with DCP outputs")
    parser.add_argument("--aod", required=True, help="Directory with AOD-Net outputs")
    parser.add_argument("--csv", default="niqe_results.csv", help="Path to NIQE CSV output")
    parser.add_argument("--summary", default="niqe_summary.txt", help="Path to NIQE summary text file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(Path(args.orig), Path(args.dcp), Path(args.aod), Path(args.csv), Path(args.summary))


if __name__ == "__main__":
    main()

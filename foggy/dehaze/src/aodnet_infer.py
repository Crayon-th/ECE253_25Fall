"""AOD-Net inference-only script for dehazing."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def _save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr8):
        raise IOError(f"Failed to write image: {path}")


class AODNet(nn.Module):
    """Lightweight implementation of AOD-Net for inference."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(torch.cat([x1, x2], dim=1)))
        x4 = F.relu(self.conv4(torch.cat([x1, x3], dim=1)))
        k = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        clean = torch.clamp(x * k - k + 1.0, 0.0, 1.0)
        return clean


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device=device)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def load_model(weights: Path, device: torch.device) -> AODNet:
    if not weights.is_file():
        print(
            f"[ERROR] Missing weights at {weights}. Place the trained AOD-Net weights at this path "
            "or pass --weights pointing to the file."
        )
        sys.exit(1)
    model = AODNet().to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model: AODNet, device: torch.device, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with torch.no_grad():
        for img_path in _iter_images(input_dir):
            try:
                rgb = _load_rgb(img_path)
                tensor = _to_tensor(rgb, device)
                pred = model(tensor)
                result = _to_numpy(pred)
                _save_png(result, output_dir / (img_path.stem + ".png"))
                total += 1
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] AOD-Net failed for {img_path}: {exc}")
    print(f"Finished AOD-Net inference: {total} images written to {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AOD-Net inference")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input directory (preprocessed RGB PNGs)")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for results")
    parser.add_argument(
        "--weights",
        default="ckpt/aodnet.pth",
        help="Path to the trained AOD-Net weights file",
    )
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.weights), device)
    run_inference(model, device, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

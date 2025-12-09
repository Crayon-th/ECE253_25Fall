"""Dark Channel Prior dehazing with guided filter refinement."""

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

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


def get_dark_channel(image: np.ndarray, window_size: int) -> np.ndarray:
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    flat_dark = dark_channel.reshape(-1)
    num_pixels = flat_dark.size
    count = max(int(num_pixels * top_percent), 1)
    indices = np.argpartition(flat_dark, -count)[-count:]
    image_flat = image.reshape(-1, 3)
    top_pixels = image_flat[indices]
    brightest = top_pixels[np.argmax(np.sum(top_pixels, axis=1))]
    return brightest


def estimate_transmission(image: np.ndarray, atmospheric_light: np.ndarray, window_size: int, omega: float) -> np.ndarray:
    normed = image / np.maximum(atmospheric_light, 1e-6)
    transmission = 1.0 - omega * get_dark_channel(normed, window_size)
    return np.clip(transmission, 0.0, 1.0)


def guided_filter(guidance: np.ndarray, to_refine: np.ndarray, radius: int, eps: float) -> np.ndarray:
    r = max(1, int(radius))
    kernel_size = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(guidance, -1, kernel_size)
    mean_p = cv2.boxFilter(to_refine, -1, kernel_size)
    mean_Ip = cv2.boxFilter(guidance * to_refine, -1, kernel_size)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guidance * guidance, -1, kernel_size)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, kernel_size)
    mean_b = cv2.boxFilter(b, -1, kernel_size)
    refined = mean_a * guidance + mean_b
    return np.clip(refined, 0.0, 1.0)


def recover_radiance(image: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray, t0: float) -> np.ndarray:
    t = np.clip(transmission, t0, 1.0)
    t = t[..., None]
    recovered = (image - atmospheric_light) / t + atmospheric_light
    return np.clip(recovered, 0.0, 1.0)


def dehaze_dcp(image: np.ndarray, window_size: int, omega: float, radius: int, eps: float, t0: float) -> np.ndarray:
    dark = get_dark_channel(image, window_size)
    atmosphere = estimate_atmospheric_light(image, dark)
    transmission = estimate_transmission(image, atmosphere, window_size, omega)
    guidance = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    refined_t = guided_filter(guidance, transmission.astype(np.float32), radius, eps)
    result = recover_radiance(image, refined_t, atmosphere, t0)
    return result


def process_directory(input_dir: Path, output_dir: Path, window: int, omega: float, radius: int, eps: float, t0: float) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for img_path in _iter_images(input_dir):
        try:
            rgb = _load_rgb(img_path)
            dehazed = dehaze_dcp(rgb, window, omega, radius, eps, t0)
            _save_png(dehazed, output_dir / (img_path.stem + ".png"))
            total += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] DCP failed for {img_path}: {exc}")
    print(f"Finished DCP processing: {total} images written to {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dark Channel Prior dehazing")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input directory containing RGB PNGs")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for DCP results")
    parser.add_argument("--window", type=int, default=15, help="Dark channel window size")
    parser.add_argument("--omega", type=float, default=0.95, help="Transmission strength parameter")
    parser.add_argument("--radius", type=int, default=60, help="Guided filter radius")
    parser.add_argument("--eps", type=float, default=1e-3, help="Guided filter epsilon")
    parser.add_argument("--t0", type=float, default=0.1, help="Minimum transmission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        args.window,
        args.omega,
        args.radius,
        args.eps,
        args.t0,
    )


if __name__ == "__main__":
    main()

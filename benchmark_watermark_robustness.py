#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from watermarking.fwht import embed_logo_watermark, extract_logo_watermark

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RNG = np.random.default_rng(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FWHT/logo watermark robustness on a dataset split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/test/images"), help="Directory containing input images.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Watermark embedding strength.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images for fast runs.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness_report.json"), help="JSON report output path.")
    parser.add_argument("--logo-name", default="Blason_univ_Yaoundé_1.png", help="Logo filename stored next to watermarking/fwht.py.")
    parser.add_argument("--block-size", type=int, default=8, help="FWHT block size.")
    parser.add_argument("--redundancy", type=int, default=3, help="Redundant copies per watermark bit.")
    parser.add_argument("--min-bit-accuracy", type=float, default=0.90, help="Success threshold for extracted logo bits.")
    parser.add_argument("--min-correlation", type=float, default=0.50, help="Success threshold for logo bit correlation.")
    parser.add_argument("--attacks", default="", help="Comma-separated attack names to run. Empty means all attacks.")
    return parser.parse_args()


def list_images(data_dir: Path, max_images: int = 0) -> list[Path]:
    image_paths = sorted(
        path for path in data_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]
    return image_paths


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.float64)


def array_to_png_bytes(image_array: np.ndarray) -> bytes:
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def jpeg_compress(image_bytes: bytes, quality: int) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def add_gaussian_noise(image_bytes: bytes, sigma: float) -> bytes:
    arr = image_bytes_to_array(image_bytes)
    noisy = arr + RNG.normal(0, sigma, size=arr.shape)
    return array_to_png_bytes(noisy)


def center_crop_and_restore(image_bytes: bytes, crop_ratio: float) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    new_width = max(8, int(width * crop_ratio))
    new_height = max(8, int(height * crop_ratio))
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    cropped = image.crop((left, top, left + new_width, top + new_height))
    restored = cropped.resize((width, height), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    restored.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def rotate_and_restore(image_bytes: bytes, angle: float) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    rotated.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def resize_roundtrip(image_bytes: bytes, scale: float) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    downscaled = image.resize(
        (max(8, int(width * scale)), max(8, int(height * scale))),
        Image.Resampling.BICUBIC,
    )
    restored = downscaled.resize((width, height), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    restored.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def compute_psnr(reference_bytes: bytes, candidate_bytes: bytes) -> float:
    ref = image_bytes_to_array(reference_bytes)
    cand = image_bytes_to_array(candidate_bytes)
    if ref.shape != cand.shape:
        image = Image.fromarray(np.clip(cand, 0, 255).astype(np.uint8)).resize(
            (ref.shape[1], ref.shape[0]),
            Image.Resampling.BICUBIC,
        )
        cand = np.array(image, dtype=np.float64)
    mse = np.mean((ref - cand) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(10 * np.log10((255.0 ** 2) / mse))


def translate(image_bytes: bytes, dx_ratio: float, dy_ratio: float) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    dx = int(width * dx_ratio)
    dy = int(height * dy_ratio)
    arr = np.array(image)
    shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[:dy, :, :] = 255
    elif dy < 0:
        shifted[dy:, :, :] = 255
    if dx > 0:
        shifted[:, :dx, :] = 255
    elif dx < 0:
        shifted[:, dx:, :] = 255
    return array_to_png_bytes(shifted)


def gaussian_blur(image_bytes: bytes, radius: float) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    buf = io.BytesIO()
    blurred.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sharpen(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    sharpened = image.filter(ImageFilter.SHARPEN)
    buf = io.BytesIO()
    sharpened.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def perspective_warp(image_bytes: bytes, strength: float) -> bytes:
    import cv2

    arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    height, width = arr.shape[:2]
    margin_x = width * strength
    margin_y = height * strength
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = np.float32(
        [
            [margin_x, margin_y * 0.5],
            [width - 1 - margin_x * 0.5, margin_y],
            [width - 1 - margin_x, height - 1 - margin_y * 0.5],
            [margin_x * 0.5, height - 1 - margin_y],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return array_to_png_bytes(warped)


def build_attacks() -> dict[str, Callable[[bytes], bytes]]:
    return {
        "clean": lambda image_bytes: image_bytes,
        "jpeg_q90": lambda image_bytes: jpeg_compress(image_bytes, quality=90),
        "jpeg_q70": lambda image_bytes: jpeg_compress(image_bytes, quality=70),
        "jpeg_q50": lambda image_bytes: jpeg_compress(image_bytes, quality=50),
        "noise_sigma_3": lambda image_bytes: add_gaussian_noise(image_bytes, sigma=3.0),
        "noise_sigma_8": lambda image_bytes: add_gaussian_noise(image_bytes, sigma=8.0),
        "noise_sigma_15": lambda image_bytes: add_gaussian_noise(image_bytes, sigma=15.0),
        "crop_90": lambda image_bytes: center_crop_and_restore(image_bytes, crop_ratio=0.90),
        "crop_75": lambda image_bytes: center_crop_and_restore(image_bytes, crop_ratio=0.75),
        "crop_60": lambda image_bytes: center_crop_and_restore(image_bytes, crop_ratio=0.60),
        "rotate_2": lambda image_bytes: rotate_and_restore(image_bytes, angle=2.0),
        "rotate_5": lambda image_bytes: rotate_and_restore(image_bytes, angle=5.0),
        "rotate_15": lambda image_bytes: rotate_and_restore(image_bytes, angle=15.0),
        "rotate_45": lambda image_bytes: rotate_and_restore(image_bytes, angle=45.0),
        "rotate_90": lambda image_bytes: rotate_and_restore(image_bytes, angle=90.0),
        "translate_10_05": lambda image_bytes: translate(image_bytes, dx_ratio=0.10, dy_ratio=0.05),
        "resize_75": lambda image_bytes: resize_roundtrip(image_bytes, scale=0.75),
        "resize_50": lambda image_bytes: resize_roundtrip(image_bytes, scale=0.50),
        "blur_1_0": lambda image_bytes: gaussian_blur(image_bytes, radius=1.0),
        "blur_2_0": lambda image_bytes: gaussian_blur(image_bytes, radius=2.0),
        "sharpen": sharpen,
        "perspective_03": lambda image_bytes: perspective_warp(image_bytes, strength=0.03),
        "perspective_08": lambda image_bytes: perspective_warp(image_bytes, strength=0.08),
    }


def summarize(records: list[dict], min_bit_accuracy: float, min_correlation: float) -> dict[str, float | int]:
    ok_records = [record for record in records if not record.get("error")]
    if not ok_records:
        return {
            "samples": len(records),
            "valid_samples": 0,
            "errors": len(records),
            "success_rate_bit_accuracy": 0.0,
            "success_rate_correlation": 0.0,
        }

    return {
        "samples": len(records),
        "valid_samples": len(ok_records),
        "errors": len(records) - len(ok_records),
        "mean_correlation": round(float(np.mean([record["correlation"] for record in ok_records])), 6),
        "min_correlation": round(float(np.min([record["correlation"] for record in ok_records])), 6),
        "mean_bit_accuracy": round(float(np.mean([record["bit_accuracy"] for record in ok_records])), 6),
        "min_bit_accuracy": round(float(np.min([record["bit_accuracy"] for record in ok_records])), 6),
        "mean_attack_psnr": round(float(np.mean([record["attack_psnr_db"] for record in ok_records])), 6),
        "success_rate_bit_accuracy": round(
            float(np.mean([1.0 if record["bit_accuracy"] >= min_bit_accuracy else 0.0 for record in ok_records])),
            6,
        ),
        "success_rate_correlation": round(
            float(np.mean([1.0 if record["correlation"] >= min_correlation else 0.0 for record in ok_records])),
            6,
        ),
    }


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.data_dir, max_images=args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    attacks = build_attacks()
    if args.attacks:
        selected_names = [name.strip() for name in args.attacks.split(",") if name.strip()]
        unknown = sorted(set(selected_names) - set(attacks))
        if unknown:
            raise ValueError(f"Unknown attacks: {', '.join(unknown)}")
        attacks = {name: attacks[name] for name in selected_names}

    results: dict[str, list[dict]] = {attack_name: [] for attack_name in attacks}

    for image_path in image_paths:
        original_bytes = image_path.read_bytes()

        try:
            watermarked_bytes, embed_metrics = embed_logo_watermark(
                original_bytes,
                logo_filename=args.logo_name,
                alpha=args.alpha,
                block_size=args.block_size,
                redundancy=args.redundancy,
            )
        except Exception as exc:
            for attack_name in attacks:
                results[attack_name].append({"image": image_path.name, "error": f"embed_failed: {exc}"})
            continue

        for attack_name, attack_fn in attacks.items():
            try:
                attacked_bytes = attack_fn(watermarked_bytes)
                bits, extract_metrics = extract_logo_watermark(
                    attacked_bytes,
                    original_bytes,
                    logo_filename=args.logo_name,
                    alpha=args.alpha,
                    block_size=args.block_size,
                    redundancy=args.redundancy,
                )
                results[attack_name].append(
                    {
                        "image": image_path.name,
                        "embed_psnr_db": embed_metrics["psnr_db"],
                        "bits_embedded": embed_metrics["bits_embedded"],
                        "attack_psnr_db": round(compute_psnr(watermarked_bytes, attacked_bytes), 6),
                        "correlation": extract_metrics["correlation"],
                        "bit_accuracy": extract_metrics["bit_accuracy"],
                        "logo_bits": extract_metrics["logo_bits"],
                        "logo_filename": extract_metrics["logo_filename"],
                        "rotation_k": extract_metrics.get("rotation_k", 0),
                        "homography_found": extract_metrics.get("homography_found", False),
                        "sift_matches": extract_metrics.get("sift_matches", 0),
                        "sift_inliers": extract_metrics.get("sift_inliers", 0),
                    }
                )
            except Exception as exc:
                results[attack_name].append({"image": image_path.name, "error": str(exc)})

    summary = {
        attack_name: summarize(records, args.min_bit_accuracy, args.min_correlation)
        for attack_name, records in results.items()
    }
    report = {
        "config": {
            "data_dir": str(args.data_dir),
            "alpha": args.alpha,
            "images_evaluated": len(image_paths),
            "block_size": args.block_size,
            "redundancy": args.redundancy,
            "min_bit_accuracy": args.min_bit_accuracy,
            "min_correlation": args.min_correlation,
            "attacks": list(attacks.keys()),
            "logo_name": args.logo_name,
        },
        "summary": summary,
        "records": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for attack_name, metrics in summary.items():
        print(json.dumps({"attack": attack_name, **metrics}, ensure_ascii=True))
    print(json.dumps({"report": str(args.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

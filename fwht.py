from __future__ import annotations

import io
import os
import hashlib

import numpy as np
from PIL import Image

from core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_logo_path(logo_filename: str) -> str:
    return os.path.join(_module_dir(), logo_filename)


def _pad_to_power_of_2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape[:2]
    new_h = 1 << (h - 1).bit_length()
    new_w = 1 << (w - 1).bit_length()
    if arr.ndim == 3:
        padded = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
    else:
        padded = np.zeros((new_h, new_w), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded, (h, w)


def fwht_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    result = x.astype(np.float64).copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    return result / n


def ifwht_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    result = x.astype(np.float64).copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    return result


def fwht_2d(block: np.ndarray) -> np.ndarray:
    rows = np.array([fwht_1d(row) for row in block])
    cols = np.array([fwht_1d(col) for col in rows.T]).T
    return cols


def ifwht_2d(block: np.ndarray) -> np.ndarray:
    rows = np.array([ifwht_1d(row) for row in block])
    cols = np.array([ifwht_1d(col) for col in rows.T]).T
    return cols


def _prepare_logo_bits(
    logo_filename: str,
    target_shape: tuple[int, int],
    threshold: int = 127,
    max_ratio: float = 0.25,
) -> tuple[np.ndarray, dict]:
    logo_path = _resolve_logo_path(logo_filename)
    logo = Image.open(logo_path).convert("L")

    target_h, target_w = target_shape
    max_h = max(1, int((target_h // 8) * max_ratio))
    max_w = max(1, int((target_w // 8) * max_ratio))
    max_h = max(max_h, 8)
    max_w = max(max_w, 8)

    if logo.size != (max_w, max_h):
        logo = logo.resize((max_w, max_h), Image.Resampling.LANCZOS)

    logo_arr = np.array(logo, dtype=np.uint8)
    bits = (logo_arr > threshold).astype(np.float64)

    meta = {
        "logo_filename": logo_filename,
        "logo_path": logo_path,
        "logo_shape": [int(bits.shape[0]), int(bits.shape[1])],
        "logo_bits": int(bits.size),
        "logo_sha256": hashlib.sha256(logo_arr.tobytes()).hexdigest(),
    }
    return bits, meta


def _rotate_90_candidates(arr: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(arr, k=k, axes=(0, 1)).copy() for k in range(4)]


def _sift_homography_warp(reference_bgr: np.ndarray, target_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    if cv2 is None:
        return target_bgr, {"homography_found": False, "sift_matches": 0, "sift_inliers": 0}

    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(tgt_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return target_bgr, {"homography_found": False, "sift_matches": 0, "sift_inliers": 0}

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return target_bgr, {"homography_found": False, "sift_matches": len(good), "sift_inliers": 0}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return target_bgr, {"homography_found": False, "sift_matches": len(good), "sift_inliers": 0}

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    warped = cv2.warpPerspective(target_bgr, H, (reference_bgr.shape[1], reference_bgr.shape[0]))

    return warped, {
        "homography_found": True,
        "sift_matches": len(good),
        "sift_inliers": inliers,
    }


def _select_logo_blocks(
    image_shape: tuple[int, int],
    logo_shape: tuple[int, int],
    block_size: int,
    redundancy: int,
) -> list[list[tuple[int, int]]]:
    h, w = image_shape
    lh, lw = logo_shape
    bh = h // block_size
    bw = w // block_size

    if lh > bh or lw > bw:
        raise ValueError("Le logo redimensionné est encore trop grand pour l'image donnée avec ce block_size.")

    total_bits = lh * lw
    needed = total_bits * redundancy
    if needed > bh * bw:
        raise ValueError("Image trop petite pour contenir le logo et sa redondance.")

    start_i = max(0, (bh - lh) // 2)
    start_j = max(0, (bw - lw) // 2)

    base_positions = [(start_i + i, start_j + j) for i in range(lh) for j in range(lw)]
    all_blocks = [(i, j) for i in range(bh) for j in range(bw)]

    placements: list[list[tuple[int, int]]] = []
    cursor = 0
    for _ in range(total_bits):
        red_blocks = []
        for _ in range(redundancy):
            while cursor < len(all_blocks) and all_blocks[cursor] in base_positions:
                cursor += 1
            if cursor >= len(all_blocks):
                cursor = 0
            red_blocks.append(all_blocks[cursor])
            cursor += 1
        placements.append(red_blocks)

    return placements


def _embed_bit_in_block(
    watermarked: np.ndarray,
    block_i: int,
    block_j: int,
    bit: float,
    alpha: float,
    block_size: int,
) -> None:
    r_start, r_end = block_i * block_size, (block_i + 1) * block_size
    c_start, c_end = block_j * block_size, (block_j + 1) * block_size
    wm_val = (2 * bit - 1) * alpha

    for ch in range(watermarked.shape[2]):
        block = watermarked[r_start:r_end, c_start:c_end, ch]
        padded, orig_shape = _pad_to_power_of_2(block)
        coeffs = fwht_2d(padded)
        mid = block_size // 2
        dc = max(abs(coeffs[0, 0]), 1.0)
               
        coeffs[mid, mid] += wm_val * dc
        coeffs[min(mid + 1, block_size - 1), mid] += wm_val * dc * 0.5
        reconstructed = ifwht_2d(coeffs)
        watermarked[r_start:r_end, c_start:c_end, ch] = reconstructed[:orig_shape[0], :orig_shape[1]]


def _read_bit_score_from_block(
    candidate: np.ndarray,
    reference: np.ndarray,
    block_i: int,
    block_j: int,
    block_size: int,
) -> float:
    r_start, r_end = block_i * block_size, (block_i + 1) * block_size
    c_start, c_end = block_j * block_size, (block_j + 1) * block_size
    if r_end > candidate.shape[0] or c_end > candidate.shape[1]:
        return 0.0

    score = 0.0
    mid = block_size // 2
    second_row = min(mid + 1, block_size - 1)

    for ch in range(candidate.shape[2]):
        ref_block = reference[r_start:r_end, c_start:c_end, ch]
        cand_block = candidate[r_start:r_end, c_start:c_end, ch]
        coeffs_ref = fwht_2d(_pad_to_power_of_2(ref_block)[0])
        coeffs_cand = fwht_2d(_pad_to_power_of_2(cand_block)[0])
        score += coeffs_cand[mid, mid] - coeffs_ref[mid, mid]
        score += 0.5 * (coeffs_cand[second_row, mid] - coeffs_ref[second_row, mid])

    return float(score)


def embed_logo_watermark(
    image_bytes: bytes,
    logo_filename: str = "Blason_univ_Yaoundé_1.png",
    alpha: float = 0.005,
    block_size: int = 8,
    redundancy: int = 3,
) -> tuple[bytes, dict]:
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("block_size doit être une puissance de 2.")
    if redundancy < 1:
        raise ValueError("redundancy doit être >= 1.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = np.array(img, dtype=np.float64)
    watermarked = original.copy()

    logo_bits, logo_meta = _prepare_logo_bits(logo_filename, original.shape[:2])
    placements = _select_logo_blocks(original.shape[:2], logo_bits.shape, block_size, redundancy)

    bits = logo_bits.flatten()
    if len(placements) < len(bits):
        raise ValueError("Erreur de préparation des placements.")

    for bit_idx, bit in enumerate(bits):
        for block_i, block_j in placements[bit_idx]:
            _embed_bit_in_block(watermarked, block_i, block_j, bit, alpha, block_size)

    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    mse = np.mean((original - watermarked.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-10))
    orig_flat = original.flatten()
    wm_flat = watermarked.astype(np.float64).flatten()
    nc = np.dot(orig_flat, wm_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(wm_flat))

    buf = io.BytesIO()
    Image.fromarray(watermarked).save(buf, format="PNG", optimize=True)

    metrics = {
        "mode": "fwht_logo_sift_ransac_rot90",
        "psnr_db": round(float(psnr), 2),
        "nc": round(float(nc), 6),
        "mse": round(float(mse), 6),
        "alpha": alpha,
        "block_size": block_size,
        "redundancy": redundancy,
        "bits_embedded": int(len(bits)),
        **logo_meta,
    }
    logger.info("fwht_logo_watermark_embedded", **metrics)
    return buf.getvalue(), metrics


def _extract_once(
    attacked: np.ndarray,
    original: np.ndarray,
    logo_bits: np.ndarray,
    logo_meta: dict,
    block_size: int,
    redundancy: int,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    placements = _select_logo_blocks(original.shape[:2], logo_bits.shape, block_size, redundancy)
    expected_bits = logo_bits.flatten()
    extracted_bits = np.zeros(len(expected_bits), dtype=np.float64)

    for idx in range(len(expected_bits)):
        scores = []
        for block_i, block_j in placements[idx]:
            score = _read_bit_score_from_block(attacked, original, block_i, block_j, block_size)
            scores.append(score)
        extracted_bits[idx] = 1.0 if np.mean(scores) > 0 else 0.0

    corr = np.corrcoef(expected_bits, extracted_bits)[0, 1] if len(expected_bits) > 1 else 0.0
    corr = 0.0 if np.isnan(corr) else float(corr)
    bit_accuracy = float(np.mean(extracted_bits == expected_bits))

    metrics = {
        "correlation": round(corr, 6),
        "bit_accuracy": round(bit_accuracy, 6),
        "alpha": alpha,
        "block_size": block_size,
        "redundancy": redundancy,
        "logo_bits": int(len(expected_bits)),
        **logo_meta,
    }
    return extracted_bits, metrics


def extract_logo_watermark(
    attacked_bytes: bytes,
    original_bytes: bytes,
    logo_filename: str = "Blason_univ_Yaoundé_1.png",
    alpha: float = 0.005,
    block_size: int = 8,
    redundancy: int = 3,
) -> tuple[np.ndarray, dict]:
    attacked_rgb = np.array(Image.open(io.BytesIO(attacked_bytes)).convert("RGB"), dtype=np.uint8)
    original_rgb = np.array(Image.open(io.BytesIO(original_bytes)).convert("RGB"), dtype=np.uint8)

    logo_bits, logo_meta = _prepare_logo_bits(logo_filename, original_rgb.shape[:2])

    best_bits = None
    best_metrics = None
    best_score = -1.0
    best_rot = 0
    best_geom = {"homography_found": False, "sift_matches": 0, "sift_inliers": 0}

    for k in range(4):
        candidate = np.rot90(attacked_rgb, k=k, axes=(0, 1)).copy()
        aligned, geom_metrics = _sift_homography_warp(original_rgb, candidate)
        bits, metrics = _extract_once(
            aligned.astype(np.float64),
            original_rgb.astype(np.float64),
            logo_bits,
            logo_meta,
            block_size,
            redundancy,
            alpha,
        )
        score = metrics["bit_accuracy"] + max(metrics["correlation"], 0.0)
        if score > best_score:
            best_score = score
            best_bits = bits
            best_metrics = metrics
            best_rot = k
            best_geom = geom_metrics

    assert best_bits is not None and best_metrics is not None

    best_metrics.update(best_geom)
    best_metrics["rotation_k"] = best_rot
    best_metrics["mode"] = "fwht_logo_sift_ransac_rot90_extract"
    logger.info("fwht_logo_watermark_extracted", **best_metrics)
    return best_bits, best_metrics

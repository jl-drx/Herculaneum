#!/usr/bin/env python3
"""Prepare bounded 2D image views for scientific review; never infer text or ink."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_guard(shape: tuple[int, ...], itemsize: int, max_pixels: int) -> None:
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3 and shape[2] in (3, 4):
        height, width, channels = shape
    else:
        raise ValueError("Only 2D grayscale or RGB/RGBA images are supported; extract a documented 2D asset first.")
    pixels = int(height) * int(width)
    if pixels <= 0 or pixels > max_pixels:
        raise ValueError(f"Input has {pixels:,} pixels; maximum is {max_pixels:,}. Use a smaller source asset.")
    # Include decoded channels and several float working arrays; cap total work.
    estimated_bytes = pixels * (channels * itemsize + 28)
    if estimated_bytes > 512 * 1024 * 1024:
        raise ValueError("Estimated image working memory exceeds 512 MiB; use a smaller source asset.")


def read_image(path: Path, max_pixels: int):
    import numpy as np
    from PIL import Image, __version__ as pillow_version

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError:
            tifffile = None
        if tifffile is not None:
            with tifffile.TiffFile(path) as source:
                if len(source.pages) != 1:
                    raise ValueError("Multi-page TIFF input is unsupported; supply a single documented 2D image.")
                page = source.pages[0]
                pixel_guard(tuple(page.shape), page.dtype.itemsize, max_pixels)
                array = page.asarray()
                return array, {"decoder": "tifffile", "decoder_version": tifffile.__version__, "source_format": "TIFF", "source_mode": str(page.photometric)}

    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Animated or multi-page images are unsupported; supply one documented 2D image.")
        original_mode = source.mode
        width, height = source.size
        # Guard before load/convert; estimate all potential channel bytes.
        channels = 4 if original_mode in {"RGBA", "CMYK", "P", "PA", "LA"} else 3 if original_mode == "RGB" else 1
        itemsize = 4 if original_mode in {"I", "F"} else 2 if original_mode.startswith("I;16") else 1
        shape = (height, width, channels) if channels in (3, 4) else (height, width)
        pixel_guard(shape, itemsize, max_pixels)
        if original_mode == "1":
            decoded = source.convert("L")
        elif original_mode in {"P", "PA", "LA"}:
            decoded = source.convert("RGBA")
        elif original_mode not in {"L", "RGB", "RGBA", "I", "F"} and not original_mode.startswith("I;16"):
            decoded = source.convert("RGB")
        else:
            decoded = source
        array = np.array(decoded)
        return array, {"decoder": "Pillow", "decoder_version": pillow_version, "source_format": source.format, "source_mode": original_mode}


def gray_array(array):
    import numpy as np

    if array.ndim == 2:
        return array.astype(np.float32), "grayscale values retained"
    if array.ndim == 3 and array.shape[2] in (3, 4):
        if array.shape[2] == 4:
            if not np.issubdtype(array.dtype, np.integer) or not np.all(array[:, :, 3] == np.iinfo(array.dtype).max):
                raise ValueError("Transparency needs an explicit scientific compositing choice; supply an opaque image.")
        rgb = array[:, :, :3].astype(np.float32)
        gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        note = "RGB display luminance: 0.299 R + 0.587 G + 0.114 B"
        if array.shape[2] == 4:
            note += "; fully opaque alpha discarded"
        return gray, note
    raise ValueError("Unsupported image shape.")


def display_array(gray, low: float, high: float):
    import numpy as np

    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = (gray - low) * (255.0 / (high - low))
    return np.rint(np.clip(np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0), 0, 255)).astype(np.uint8)


def write_preview(array, output: Path, max_side: int) -> dict:
    from PIL import Image

    view = Image.fromarray(array)
    input_size = view.size
    if max(input_size) > max_side:
        view.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    view.save(output, format="PNG")
    return {"path": output.name, "size_wh": list(view.size), "resampling": "Lanczos" if view.size != input_size else "none", "sha256": file_sha256(output)}


def inspect(args: argparse.Namespace) -> Path:
    import numpy as np

    source = args.image.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a regular image file.")
    digest = file_sha256(source)
    if args.sha256 and digest != args.sha256.lower():
        raise ValueError("Input SHA-256 does not match the expected digest; no review outputs were created.")
    array, decoder_info = read_image(source, args.max_pixels)
    source_shape = list(array.shape)
    source_dtype = str(array.dtype)
    height, width = array.shape[:2]
    crop = args.crop or [0, 0, width, height]
    x, y, crop_width, crop_height = crop
    if x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0 or x + crop_width > width or y + crop_height > height:
        raise ValueError("Crop must have positive width/height and lie within source image bounds.")
    cropped = array[y:y + crop_height, x:x + crop_width]
    gray, grayscale_note = gray_array(cropped)
    finite = np.isfinite(gray)
    if not finite.any():
        raise ValueError("The selected region contains no finite pixel values.")
    finite_values = gray[finite]
    observed_min, observed_max = float(finite_values.min()), float(finite_values.max())
    if np.issubdtype(array.dtype, np.integer):
        dtype_limits = np.iinfo(array.dtype)
        original_low, original_high = float(dtype_limits.min), float(dtype_limits.max)
        original_mapping = "integer dtype range mapped linearly to 8-bit display"
    elif np.issubdtype(array.dtype, np.floating):
        original_low, original_high = observed_min, observed_max
        original_mapping = "floating-point finite crop minimum/maximum mapped linearly to 8-bit display"
    else:
        raise ValueError(f"Unsupported pixel dtype: {array.dtype}.")
    contrast_low, contrast_high = (float(value) for value in np.percentile(finite_values, args.percentiles))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = ["source-preview.png", "percentile-contrast.png", "metadata.json"]
    if any((output_dir / name).exists() for name in output_names):
        raise ValueError("Output directory already contains review outputs; choose a fresh output directory.")
    source_view = write_preview(display_array(gray, original_low, original_high), output_dir / output_names[0], args.max_side)
    contrast_view = write_preview(display_array(gray, contrast_low, contrast_high), output_dir / output_names[1], args.max_side)
    source_view.update({"display_mapping": original_mapping, "display_range": [original_low, original_high]})
    contrast_view.update({"display_mapping": "percentile clipping followed by linear mapping to 8-bit display", "percentiles": args.percentiles, "display_range": [contrast_low, contrast_high]})
    preview_width, preview_height = source_view["size_wh"]
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "2D image processing for review; no ink detection, OCR, transcription, or translation",
        "source": {
            "path": str(source),
            "source_url": args.source_url,
            "sha256": digest,
            "expected_sha256": args.sha256.lower() if args.sha256 else None,
            "size_bytes": source.stat().st_size,
            "shape": source_shape,
            "dtype": source_dtype,
            "provenance_note": args.provenance_note,
            "preserved": "source file opened read-only; previews are generated derivatives",
            **decoder_info,
        },
        "region": {
            "bbox_xywh": crop,
            "coordinate_convention": "zero-based source pixels; x is column, y is row; half-open crop extent",
            "grayscale_processing": grayscale_note,
            "nonfinite_pixel_count": int((~finite).sum()),
            "finite_value_range": [observed_min, observed_max],
            "source_to_preview_edge_coordinates": {
                "x": {"offset_before_scale": -x, "scale": preview_width / crop_width},
                "y": {"offset_before_scale": -y, "scale": preview_height / crop_height},
            },
        },
        "outputs": {"source_preview": source_view, "processed_percentile_contrast": contrast_view},
        "limitations": [
            "Preview contrast and texture cannot establish ink or writing.",
            "Display PNGs use 8-bit grayscale; retain the source for full intensity and color information.",
            "Image orientation is retained as decoded; EXIF rotation is not automatically applied.",
            "Crop selection follows full-image bounds checks and may still decode the full 2D source.",
            "No dataset license, source authenticity, annotation alignment, or physical scale is inferred.",
        ],
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-url", default=None, help="Recorded source URL; no network requests are made.")
    parser.add_argument("--sha256", "--sha", dest="sha256", default=None, help="Expected SHA-256, if independently available.")
    parser.add_argument("--provenance-note", default=None)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--max-side", type=int, default=2048, help="Maximum preview edge length (default: 2048).")
    parser.add_argument("--max-pixels", type=int, default=16_000_000, help="Full 2D input pixel bound; a 512 MiB memory estimate also applies.")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[1.0, 99.0], metavar=("LOW", "HIGH"))
    args = parser.parse_args()
    if args.max_side <= 0 or args.max_pixels <= 0:
        parser.error("--max-side and --max-pixels must be positive.")
    if not (0 <= args.percentiles[0] < args.percentiles[1] <= 100):
        parser.error("Percentiles must satisfy 0 <= LOW < HIGH <= 100.")
    if args.sha256 and (len(args.sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in args.sha256)):
        parser.error("--sha256 must be 64 hexadecimal characters.")
    try:
        path = inspect(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"Image review failed: {error}\n")
    print(path)


if __name__ == "__main__":
    main()

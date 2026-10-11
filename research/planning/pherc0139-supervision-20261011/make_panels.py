#!/usr/bin/env python3
"""Fixed 0..255 primary-plane panels; no fitted contrast or target selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def display(values, valid):
    # Float scores are rounded ONLY for a fixed 8-bit display, not for metrics.
    gray = np.rint(np.clip(values, 0, 255)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = [255, 0, 255]
    return Image.fromarray(rgb).resize((192, 192), Image.Resampling.NEAREST)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.study
    out = args.output or root / "panels"
    out.mkdir(parents=True, exist_ok=False)
    candidates = json.loads((root / "CT_CANDIDATES.json").read_text())
    value_path = root / "results/transferred_teacher_float64_uint8_scale.npz"
    mask_path = root / "results/transferred_teacher_pseudo_supervision_support.npz"
    values, masks = np.load(value_path), np.load(mask_path)
    entries = []
    for support in candidates["supports"]:
        sid = support["support"]
        image = Image.new("RGB", (1030, 670), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 12), f"Site 1 / support {sid} / native centre {support['centre_native_xy']} / j=14", fill="black", font=font(21))
        draw.text((20, 43), f"All {len(support['candidates'])} CT alternatives retained; fixed grayscale 0-255 on every panel", fill="black", font=font(16))
        columns = ("Common CT fit", "Finite-set minimum", "Finite-set maximum", "Finite-set range")
        for col, title in enumerate(columns):
            draw.text((190 + 202*col, 92), title, fill="black", font=font(16))
        for row, (op, caption) in enumerate((("mean16_bilinear", "Mean of 4x4\nthen bilinear"),
                                           ("point_bilinear", "Point bilinear\n(no pre-pool)"))):
            keys = [f"{op}_s{sid}_fit_j14"] + [f"{op}_s{sid}_candidate_{i:02d}_j14" for i in range(len(support["candidates"]))]
            v = np.stack([values[k] for k in keys])
            common = np.stack([masks[k] for k in keys]).all(axis=0)
            panels = (v[0], v.min(axis=0), v.max(axis=0), np.ptp(v, axis=0))
            top = 120 + row*245
            draw.multiline_text((20, top+55), caption, fill="black", font=font(18), spacing=6)
            draw.text((20, top+125), f"Common: {common.sum()}/1024", fill="black", font=font(14))
            for col, pixels in enumerate(panels):
                image.paste(display(pixels, common), (190+202*col, top))
        draw.text((20, 585), "Magenta = excluded from common teacher>0 AND native-eligible support.", fill="black", font=font(16))
        draw.text((20, 612), "Two-dimensional teacher projections only. Range is variability, not error or ink accuracy.", fill="black", font=font(16))
        draw.text((20, 639), "Prior structural screen remains FAIL; none of these panels certifies an ink face or depth.", fill="black", font=font(16))
        path = out / f"support{sid}-j14-fixed.png"
        image.save(path)
        entries.append(dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size))
    manifest = dict(display="Linear fixed 0..255 for score AND range, no fitted contrast; nearest-neighbour 6x expansion",
                    native_plane=14, all_four_checking_supports=True,
                    image_quantization="round-to-nearest for display only; float metric arrays unchanged",
                    value_archive_sha256=sha(value_path), mask_archive_sha256=sha(mask_path),
                    code_sha256=sha(__file__), panels=entries)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(status="COMPLETE", panels=len(entries), output=str(out))))


if __name__ == "__main__":
    main()

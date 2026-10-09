"""Synthetic PNG schema and fixed crop/filter/mean fixtures; no published image reads."""
import argparse
import io
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from regional_common import write_json
from run_ir_diagnostic import validate_image, mean3


def body(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checks = []
    for mode in ("L", "RGB", "RGBA", "P"):
        image = Image.new(mode, (3, 2), (10, 20, 30, 255) if mode == "RGBA" else ((10, 20, 30) if mode == "RGB" else 1))
        if mode == "P":
            image.putpalette([0, 0, 0, 10, 20, 30] + [0]*762)
        decoded, metadata = validate_image(body(image), [2, 3])
        assert metadata["original_mode"] == mode and metadata["fully_opaque"]
        if mode in ("RGB", "RGBA", "P"):
            assert np.array(decoded.convert("L")).tolist() == [[18, 18, 18], [18, 18, 18]]
        decoded.close()
        checks.append("permitted-" + mode)
    for name, image, expected in (
        ("alpha", Image.new("RGBA", (3, 2), (10, 20, 30, 0)), [2, 3]),
        ("16bit", Image.fromarray(np.ones((2, 3), np.uint16)), [2, 3]),
        ("canvas", Image.new("L", (3, 2)), [3, 2]),
    ):
        try:
            validate_image(body(image), expected)
        except ValueError:
            checks.append("reject-" + name)
        else:
            raise AssertionError(name)
    indexed = Image.new("P", (3, 2), 1)
    indexed.putpalette([0]*768)
    indexed.info["transparency"] = 1
    try:
        validate_image(body(indexed), [2, 3])
    except ValueError:
        checks.append("reject-palette-transparency")
    else:
        raise AssertionError("transparent palette accepted")
    constant = np.full((1152, 1152), 27.0, np.float64)
    assert np.array_equal(mean3(constant), np.full((384, 384), 27.0))
    filtered = gaussian_filter(constant, sigma=16, truncate=4, mode="reflect") - constant
    assert np.max(np.abs(mean3(filtered))) < 1e-12
    checks.extend(["constant-block-mean", "constant-local-darkness-zero"])
    write_json(args.output_dir / "ir-fixtures.json", dict(status="PASS", synthetic_only=True, count=len(checks), checks=checks,
               actual_published_images_read=0))
    print(f"PASS: {len(checks)} synthetic infrared checks")


if __name__ == "__main__":
    main()

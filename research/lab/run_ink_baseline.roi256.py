#!/usr/bin/env python3
"""Run verified ink_9um checkpoints on one bounded surface ROI using CPU FP32.

This standalone adapter uses unmodified, pinned upstream model and normalization
modules. It does not install the complete vesuvius package, read remote data,
train a model, identify letters, or generate a transcription or translation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


LAB = Path(__file__).resolve().parent
VENDOR = LAB / "vendor" / "villa-ink-baseline"
VILLA_COMMIT = "e0bbb8b40a2db58b1d71864f286eb85717e59e64"
HF_COMMIT = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
CHECKPOINTS = {
    "seed42": {
        "size_bytes": 138360039,
        "sha256": "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab",
        "repository_path": "hybrid_3d2d-seed42/step-075000.pth",
    },
    "seed43": {
        "size_bytes": 138360231,
        "sha256": "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f",
        "repository_path": "hybrid_3d2d-seed43/step-075000.pth",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_focused_sources():
    """Verify the source snapshot before importing its namespace modules."""
    manifest_path = VENDOR / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_commit"] != VILLA_COMMIT:
        raise ValueError("Focused source manifest has an unexpected upstream revision.")
    for entry in manifest["files"]:
        source_path = (VENDOR / entry["path"]).resolve()
        if not source_path.is_relative_to(VENDOR.resolve()):
            raise ValueError("Source manifest path leaves the vendor directory.")
        if sha256_file(source_path) != entry["sha256"]:
            raise ValueError(f"Focused source checksum mismatch: {entry['path']}")
    if sha256_file(VENDOR / "UPSTREAM-LICENSE.txt") != manifest["license_sha256"]:
        raise ValueError("Retained upstream license checksum mismatch.")
    if "vesuvius" in sys.modules:
        raise ValueError("Run this adapter in a fresh process; a vesuvius module is already loaded.")
    sys.path.insert(0, str(VENDOR))
    package = importlib.import_module("vesuvius")
    expected_namespace = (VENDOR / "vesuvius").resolve()
    namespace_paths = {Path(path).resolve() for path in package.__path__}
    if getattr(package, "__file__", None) is not None or namespace_paths != {expected_namespace}:
        raise ValueError("An installed vesuvius package conflicts with the isolated focused namespace.")
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.image_proc.intensity.normalization import normalize_robust

    return InkConfig, make_model, normalize_robust, sha256_file(manifest_path)


def selected_state(payload: Mapping):
    for name in ("ema_model", "state_dict", "model_state_dict", "model"):
        state = payload.get(name)
        if isinstance(state, Mapping):
            if not state:
                raise ValueError(f"Checkpoint weight container {name!r} is empty.")
            if all(str(key).startswith("module.") for key in state):
                state = {str(key)[len("module."):]: value for key, value in state.items()}
            return name, state
    raise ValueError("Checkpoint has no supported nonempty model-state container.")


def build_checkpoint_model(path: Path, label: str, InkConfig, make_model):
    import torch

    expected = CHECKPOINTS[label]
    if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"{label} checkpoint does not match the pinned final checkpoint size and SHA-256.")
    # Never fall back to unrestricted pickle loading.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise ValueError("Checkpoint must contain an embedded configuration mapping.")
    authored_config = dict(payload["config"])
    config = InkConfig.from_mapping(authored_config)
    if config.data.mode != "flat" or tuple(config.model.crop_size) != (17, 128, 128):
        raise ValueError("Checkpoint does not have the required flat [17,128,128] input contract.")
    if config.model.model_type not in {"vesuvius_unet_3d_stem_2d", "unet_3d_stem_2d"}:
        raise ValueError("Checkpoint is not the expected hybrid 3D-stem/2D U-Net.")
    normalizer = config.data.normalization
    if normalizer.mode != "robust_mad" or (normalizer.percentile_lower, normalizer.percentile_upper) != (1.0, 99.0):
        raise ValueError("Checkpoint normalization differs from the audited robust-MAD contract.")
    if config.model.pretrained_backbone is not None or config.model.input_pad_depth_to is not None:
        raise ValueError("Unexpected backbone or depth-padding option in the checkpoint.")
    state_name, state = selected_state(payload)
    model = make_model(config)
    # strict=True rejects any missing, unexpected, or mismatched parameters.
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError("Checkpoint state is not an exact match to the pinned model factory.")
    model = model.float().to(torch.device("cpu")).eval()
    metadata = {
        "label": label,
        "path": str(path.resolve()),
        "size_bytes": expected["size_bytes"],
        "sha256": expected["sha256"],
        "model_repository": "scrollprize/ink_9um",
        "model_revision": HF_COMMIT,
        "repository_path": expected["repository_path"],
        "deserialization": "torch.load(map_location='cpu', weights_only=True)",
        "weight_container": state_name,
        "strict_state_match": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "model_type": config.model.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "embedded_config": authored_config,
    }
    del payload, state
    gc.collect()
    return model, metadata


def hann_blend_map():
    """Reproduce pinned upstream compute_importance_map_2d(mode='hann')."""
    import torch

    weight = torch.outer(
        torch.hann_window(128, periodic=False, dtype=torch.float32),
        torch.hann_window(128, periodic=False, dtype=torch.float32),
    )
    weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
    return torch.clamp(weight, min=0.001).numpy()


def infer_surface(model, surface, layer_start: int, direction: str, normalize_robust):
    import numpy as np
    import torch

    layer_stop = layer_start + 17
    layers = surface[layer_start:layer_stop]
    if direction == "reverse":
        layers = layers[::-1]
    probability_sum = np.zeros((256, 256), dtype=np.float32)
    weight_sum = np.zeros((256, 256), dtype=np.float32)
    weights = hann_blend_map()
    tile_records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for y in (0, 64, 128):
            for x in (0, 64, 128):
                raw_patch = layers[:, y:y + 128, x:x + 128]
                if not np.any(raw_patch):
                    raise ValueError("Empty source tile needs an explicit mask; this initial experiment requires all nine tiles to be nonempty.")
                normalized = np.ascontiguousarray(normalize_robust(raw_patch), dtype=np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError("Normalization produced non-finite values.")
                tensor = torch.from_numpy(normalized)[None, None]
                tile_started = time.perf_counter()
                outputs = model(tensor)
                if not isinstance(outputs, Mapping) or "ink" not in outputs:
                    raise ValueError("Pinned model did not produce an ink logits mapping.")
                logits = outputs["ink"]
                if isinstance(logits, (list, tuple)):
                    if not logits:
                        raise ValueError("Empty model output.")
                    logits = logits[0]
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, 128, 128):
                    raise ValueError("Unexpected output shape; required [1,1,128,128] logits.")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model produced non-finite logits.")
                probabilities = logits.float().sigmoid().cpu().numpy()[0, 0]
                probability_sum[y:y + 128, x:x + 128] += probabilities * weights
                weight_sum[y:y + 128, x:x + 128] += weights
                tile_records.append({"bbox_roi_xywh": [x, y, 128, 128], "forward_seconds": time.perf_counter() - tile_started})
    if not np.all(weight_sum > 0):
        raise ValueError("Hann blending left uncovered output pixels.")
    probabilities = probability_sum / weight_sum
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Blended model scores are not finite values in [0,1].")
    return probabilities.astype(np.float32, copy=False), {"tiles": tile_records, "tile_count": len(tile_records), "inference_seconds": time.perf_counter() - started}


def score_summary(array) -> dict:
    import numpy as np

    return {
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "std": float(array.std()),
        "percentiles_1_50_99": [float(value) for value in np.percentile(array, [1, 50, 99])],
    }


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    surface_path = args.surface.resolve(strict=True)
    provenance_path = args.provenance.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Choose a fresh output directory; existing files are preserved.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_sha = sha256_file(surface_path)
    if source_sha != provenance.get("array_sha256"):
        raise ValueError("Surface file SHA-256 does not match retained input provenance.")
    surface = np.load(surface_path, mmap_mode="r", allow_pickle=False)
    if surface.shape != (31, 256, 256) or surface.dtype != np.uint8:
        raise ValueError("This bounded experiment requires an exact uint8 [31,256,256] surface ROI.")
    if list(surface.shape) != provenance.get("shape_zyx"):
        raise ValueError("Input dimensions do not match retained provenance.")
    if not 0 <= args.layer_start <= 14:
        raise ValueError("A 17-layer window must lie within the 31 retained source layers.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    InkConfig, make_model, normalize_robust, source_manifest_sha = load_focused_sources()
    predictions = {}
    model_records = []
    for label in ("seed42", "seed43"):
        path = getattr(args, f"{label}_checkpoint")
        if path is None:
            continue
        model, model_metadata = build_checkpoint_model(path.resolve(strict=True), label, InkConfig, make_model)
        print(f"Loaded {label} with strict state match; processing nine CPU tiles.", flush=True)
        prediction, inference_metadata = infer_surface(model, surface, args.layer_start, args.direction, normalize_robust)
        predictions[label] = prediction
        model_metadata.update(inference_metadata)
        model_metadata["score_summary"] = score_summary(prediction)
        model_metadata["interior_score_summary"] = score_summary(prediction[64:192, 64:192])
        model_records.append(model_metadata)
        del model
        gc.collect()
    if not predictions:
        raise ValueError("Supply at least one pinned seed checkpoint.")
    if sha256_file(surface_path) != source_sha:
        raise ValueError("Source file changed during the run.")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for label, values in predictions.items():
        path = output_dir / f"{label}-probabilities.npy"
        np.save(path, values, allow_pickle=False)
        outputs[label] = {"path": path.name, "sha256": sha256_file(path), "dtype": str(values.dtype), "shape_yx": list(values.shape)}
    stability = None
    if len(predictions) == 2:
        mean = (predictions["seed42"] + predictions["seed43"]) * np.float32(0.5)
        difference = np.abs(predictions["seed42"] - predictions["seed43"])
        for name, values in (("ensemble-mean", mean), ("seed-absolute-difference", difference)):
            path = output_dir / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            outputs[name] = {"path": path.name, "sha256": sha256_file(path), "dtype": str(values.dtype), "shape_yx": list(values.shape)}
        stability = {"absolute_difference": score_summary(difference), "interior_absolute_difference": score_summary(difference[64:192, 64:192])}
        if all(float(values.std()) > 0 for values in predictions.values()):
            stability["pearson_correlation"] = float(np.corrcoef(predictions["seed42"].ravel(), predictions["seed43"].ravel())[0, 1])
    package_names = ["torch", "torchvision", "timm", "einops", "huggingface-hub", "numpy"]
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Verified-checkpoint model scoring completed; no letter reading or translation",
        "source": {"path": str(surface_path), "sha256": source_sha, "provenance_path": str(provenance_path), "provenance_sha256": sha256_file(provenance_path), "provenance": provenance},
        "focused_upstream": {"repository": "ScrollPrize/villa", "commit": VILLA_COMMIT, "source_manifest": str(VENDOR / "source-manifest.json"), "source_manifest_sha256": source_manifest_sha, "source_modules_unmodified": True, "complete_package_installed": False},
        "adapter": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "differences_from_stock_cli": ["Explicit CPU FP32 with no compilation/autocast/TTA", "Prebounded .npy surface input instead of remote/full Zarr scheduling", "weights_only=True checkpoint deserialization", "Strict state matching", "Unquantized float32 model scores retained instead of uint8 TIFF"]},
        "processing": {"device": "cpu", "dtype": "float32", "threads": args.threads, "interop_threads": 1, "deterministic_algorithms": True, "batch_size": 1, "patch_size_zyx": [17, 128, 128], "stride_xy": [64, 64], "layer_window_start_stop": [args.layer_start, args.layer_start + 17], "source_layer_indices": list(range(args.layer_start, args.layer_start + 17))[::(-1 if args.direction == "reverse" else 1)], "depth_direction": args.direction, "normalization": "Unmodified upstream normalize_robust over each complete ZYX patch: percentile clip 1/99, median/MAD*1.4826 with upstream fallbacks", "blend": "Nonperiodic torch Hann outer product, normalized maximum, floored at 0.001", "review_interior_bbox_roi_xywh": [64, 64, 128, 128], "display_rescaling_applied": False},
        "models": model_records,
        "outputs": outputs,
        "seed_stability": stability,
        "runtime": {"python": sys.version.split()[0], "packages": {name: importlib.metadata.version(name) for name in package_names}},
        "limitations": ["Scores are model outputs, not independently established ink or calibrated authenticity probabilities.", "Seed agreement is a stability diagnostic; both seeds share a training corpus.", "No target ink labels or vetted reading were used; no precision/recall or reading accuracy is established.", "Native 8.640um target sampling differs from stated 9.362um/~9.596um training sources.", "Interpretation requires source geometry, depth/orientation controls and independent stroke review.", "BCE smoothing maps training targets to 0.25/0.75; any (p-0.25)/0.5 display transform must be recorded separately."]
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path, help="Retained uint8 .npy surface ROI, shape [31,256,256].")
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed42-checkpoint", type=Path)
    parser.add_argument("--seed43-checkpoint", type=Path)
    parser.add_argument("--layer-start", type=int, default=7, help="First of 17 retained source layers (default: 7).")
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.seed42_checkpoint is None and args.seed43_checkpoint is None:
        parser.error("Provide --seed42-checkpoint and/or --seed43-checkpoint.")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16.")
    try:
        metadata_path = run(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"Baseline run failed: {error}\n")
    print(metadata_path)


if __name__ == "__main__":
    main()

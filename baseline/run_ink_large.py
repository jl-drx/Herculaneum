#!/usr/bin/env python3
"""Run verified ink_9um checkpoints on a bounded large surface ROI using CPU FP32.

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
ADAPTER_CODE_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
MAX_ROI_DIMENSION = 2048
DEFAULT_TOTAL_TILE_BUDGET = 1200
HARD_TOTAL_TILE_BUDGET = 1922
BASELINE_SOURCE_SHA256 = '2cbdb280777e19794fc78857969f9c1d43965da17772f6a1659264e6f5a48a59'
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


def sliding_positions(length: int) -> list[int]:
    """Schedule full 128-pixel windows, including a final edge-aligned tile."""
    if not 128 <= length <= MAX_ROI_DIMENSION:
        raise ValueError(f"Each ROI dimension must be between 128 and {MAX_ROI_DIMENSION} pixels.")
    positions = list(range(0, length - 128 + 1, 64))
    if positions[-1] != length - 128:
        positions.append(length - 128)
    return positions


def review_interior(height: int, width: int) -> tuple[slice, slice]:
    """Return a central 128-pixel review region without assigning validity."""
    y0, x0 = height // 2 - 64, width // 2 - 64
    return slice(y0, y0 + 128), slice(x0, x0 + 128)


def infer_surface(model, surface, layer_start: int, direction: str, normalize_robust, valid_mask=None):
    import numpy as np
    import torch

    layer_stop = layer_start + 17
    layers = surface[layer_start:layer_stop]
    if direction == "reverse":
        layers = layers[::-1]
    height, width = surface.shape[1:]
    if valid_mask is None:
        valid_mask = np.ones((height, width), dtype=bool)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    weights = hann_blend_map()
    tile_records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for y in sliding_positions(height):
            for x in sliding_positions(width):
                bbox = [x, y, 128, 128]
                if not np.any(valid_mask[y:y + 128, x:x + 128]):
                    tile_records.append({"bbox_roi_xywh": bbox, "status": "skipped", "reason": "no valid geometry-mask pixels"})
                    continue
                raw_patch = layers[:, y:y + 128, x:x + 128]
                if not np.any(raw_patch):
                    tile_records.append({"bbox_roi_xywh": bbox, "status": "skipped", "reason": "raw selected-layer patch is entirely zero"})
                    continue
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
                tile_records.append({"bbox_roi_xywh": bbox, "status": "scored", "forward_seconds": time.perf_counter() - tile_started})
    evaluated_mask = (weight_sum > 0) & valid_mask
    # Upstream empty-tile behavior leaves zero probability where no patch was
    # evaluated. Distinguish those pixels from a model's prediction in metadata
    # and evaluated-mask outputs, and exclude geometrically invalid pixels.
    probabilities = np.zeros((height, width), dtype=np.float32)
    np.divide(probability_sum, weight_sum, out=probabilities, where=weight_sum > 0)
    if not np.isfinite(probabilities[valid_mask]).all() or np.any(probabilities[valid_mask] < 0) or np.any(probabilities[valid_mask] > 1):
        raise ValueError("Blended model scores are not finite values in [0,1].")
    probabilities[~valid_mask] = np.nan
    metadata = {
        "tiles": tile_records,
        "tile_count": len(tile_records),
        "scored_tile_count": sum(tile["status"] == "scored" for tile in tile_records),
        "skipped_tile_count": sum(tile["status"] == "skipped" for tile in tile_records),
        "valid_pixel_count": int(valid_mask.sum()),
        "evaluated_valid_pixel_count": int(evaluated_mask.sum()),
        "valid_but_unscored_pixel_count": int((valid_mask & ~evaluated_mask).sum()),
        "inference_seconds": time.perf_counter() - started,
    }
    return probabilities, metadata, evaluated_mask


def score_summary(array) -> dict:
    import numpy as np

    values = array[np.isfinite(array)]
    return {
        "finite_pixel_count": int(values.size),
        "excluded_nonfinite_pixel_count": int(array.size - values.size),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "percentiles_1_50_99": [float(value) for value in np.percentile(values, [1, 50, 99])] if values.size else None,
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
    if surface.ndim != 3 or surface.shape[0] != 31 or surface.dtype != np.uint8:
        raise ValueError("This bounded experiment requires a uint8 [31,H,W] surface ROI.")
    height, width = surface.shape[1:]
    sliding_positions(height)
    sliding_positions(width)
    checkpoint_count = sum(path is not None for path in (args.seed42_checkpoint, args.seed43_checkpoint))
    scheduled_tiles_per_checkpoint = len(sliding_positions(height)) * len(sliding_positions(width))
    total_scheduled_tiles = scheduled_tiles_per_checkpoint * checkpoint_count
    if total_scheduled_tiles > args.max_total_tiles:
        raise ValueError(f"Scheduled {total_scheduled_tiles} checkpoint tiles exceed the declared --max-total-tiles budget {args.max_total_tiles}.")
    if list(surface.shape) != provenance.get("shape_zyx"):
        raise ValueError("Input dimensions do not match retained provenance.")
    if not 0 <= args.layer_start <= 14:
        raise ValueError("A 17-layer window must lie within the 31 retained source layers.")
    valid_mask = np.ones((height, width), dtype=bool)
    mask_metadata = None
    if args.mask is not None:
        mask_path = args.mask.resolve(strict=True)
        mask_sha = sha256_file(mask_path)
        valid_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if valid_mask.shape != (height, width) or valid_mask.dtype != np.bool_:
            raise ValueError("--mask must be a boolean .npy array aligned to the ROI's [H,W] pixels.")
        if not np.any(valid_mask):
            raise ValueError("The supplied validity mask contains no valid pixels.")
        mask_metadata = {"path": str(mask_path), "sha256": mask_sha, "shape_yx": [height, width], "valid_pixel_count": int(valid_mask.sum()), "assumption": "Caller-provided geometry validity; mask/source alignment must be established independently."}
    interior = review_interior(height, width)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    InkConfig, make_model, normalize_robust, source_manifest_sha = load_focused_sources()
    predictions = {}
    evaluated_masks = {}
    model_records = []
    for label in ("seed42", "seed43"):
        path = getattr(args, f"{label}_checkpoint")
        if path is None:
            continue
        model, model_metadata = build_checkpoint_model(path.resolve(strict=True), label, InkConfig, make_model)
        tile_count = len(sliding_positions(height)) * len(sliding_positions(width))
        print(f"Loaded {label} with strict state match; scheduling {tile_count} CPU tiles.", flush=True)
        prediction, inference_metadata, evaluated = infer_surface(model, surface, args.layer_start, args.direction, normalize_robust, valid_mask)
        predictions[label] = prediction
        evaluated_masks[label] = evaluated
        model_metadata.update(inference_metadata)
        model_metadata["score_summary"] = score_summary(prediction)
        model_metadata["evaluated_score_summary"] = score_summary(prediction[evaluated])
        model_metadata["interior_score_summary"] = score_summary(prediction[interior])
        model_records.append(model_metadata)
        del model
        gc.collect()
    if not predictions:
        raise ValueError("Supply at least one pinned seed checkpoint.")
    if sha256_file(surface_path) != source_sha:
        raise ValueError("Source file changed during the run.")
    if mask_metadata is not None and sha256_file(mask_path) != mask_sha:
        raise ValueError("Validity-mask file changed during the run.")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for label, values in predictions.items():
        path = output_dir / f"{label}-probabilities.npy"
        np.save(path, values, allow_pickle=False)
        outputs[label] = {"path": path.name, "sha256": sha256_file(path), "dtype": str(values.dtype), "shape_yx": list(values.shape)}
        evaluated_path = output_dir / f"{label}-evaluated-mask.npy"
        np.save(evaluated_path, evaluated_masks[label], allow_pickle=False)
        outputs[f"{label}-evaluated-mask"] = {"path": evaluated_path.name, "sha256": sha256_file(evaluated_path), "dtype": "bool", "shape_yx": [height, width]}
    validity_path = output_dir / "validity-mask.npy"
    np.save(validity_path, valid_mask, allow_pickle=False)
    outputs["validity-mask"] = {"path": validity_path.name, "sha256": sha256_file(validity_path), "dtype": "bool", "shape_yx": [height, width]}
    stability = None
    if len(predictions) == 2:
        mean = (predictions["seed42"] + predictions["seed43"]) * np.float32(0.5)
        difference = np.abs(predictions["seed42"] - predictions["seed43"])
        for name, values in (("ensemble-mean", mean), ("seed-absolute-difference", difference)):
            path = output_dir / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            outputs[name] = {"path": path.name, "sha256": sha256_file(path), "dtype": str(values.dtype), "shape_yx": list(values.shape)}
        stability = {"absolute_difference": score_summary(difference), "interior_absolute_difference": score_summary(difference[interior])}
        common = valid_mask & evaluated_masks["seed42"] & evaluated_masks["seed43"]
        seed42_values, seed43_values = predictions["seed42"][common], predictions["seed43"][common]
        if seed42_values.size >= 2 and seed42_values.std() > 0 and seed43_values.std() > 0:
            stability["pearson_correlation_on_evaluated_valid_pixels"] = float(np.corrcoef(seed42_values, seed43_values)[0, 1])
    package_names = ["torch", "torchvision", "timm", "einops", "huggingface-hub", "numpy"]
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Verified-checkpoint model scoring completed; no letter reading or translation",
        "source": {"path": str(surface_path), "sha256": source_sha, "provenance_path": str(provenance_path), "provenance_sha256": sha256_file(provenance_path), "provenance": provenance, "validity_mask": mask_metadata},
        "focused_upstream": {"repository": "ScrollPrize/villa", "commit": VILLA_COMMIT, "source_manifest": str(VENDOR / "source-manifest.json"), "source_manifest_sha256": source_manifest_sha, "source_modules_unmodified": True, "complete_package_installed": False},
        "adapter": {"path": str(Path(__file__).resolve()), "sha256": ADAPTER_CODE_SHA256, "derived_from_baseline_source_sha256": BASELINE_SOURCE_SHA256, "declared_changes": "Maximum dimension2048; explicit scheduled-tile budget; metadata/help only. Model, normalization, mask, blending and inference methods are unchanged.", "hash_capture": "Captured at module startup before any checkpoint or target inference", "differences_from_stock_cli": ["Explicit CPU FP32 with no compilation/autocast/TTA", "Prebounded .npy surface input instead of remote/full Zarr scheduling", "weights_only=True checkpoint deserialization", "Strict state matching", "Unquantized float32 model scores retained instead of uint8 TIFF", "Caller-provided boolean geometry mask excludes invalid pixels as NaN and is retained separately"]},
        "processing": {"device": "cpu", "dtype": "float32", "threads": args.threads, "interop_threads": 1, "deterministic_algorithms": True, "batch_size": 1, "patch_size_zyx": [17, 128, 128], "stride_xy": [64, 64], "large_surface_budget": {"max_roi_dimension": MAX_ROI_DIMENSION, "declared_max_total_tiles": args.max_total_tiles, "scheduled_tiles_per_checkpoint": scheduled_tiles_per_checkpoint, "total_scheduled_checkpoint_tiles": total_scheduled_tiles, "guard_counts_all_tiles_before_mask_or_empty_skips": True}, "tile_starts_y": sliding_positions(height), "tile_starts_x": sliding_positions(width), "edge_policy": "Final full tile starts at dimension-128; no cropping or padding of source pixels", "empty_tile_policy": "Skip entirely zero raw selected-layer patches; valid pixels without model contribution retain stock zero-output semantics and evaluated-mask=false", "mask_policy": "Mask affects scheduling and final validity, never patch normalization; invalid scores are NaN", "layer_window_start_stop": [args.layer_start, args.layer_start + 17], "source_layer_indices": list(range(args.layer_start, args.layer_start + 17))[::(-1 if args.direction == "reverse" else 1)], "depth_direction": args.direction, "normalization": "Unmodified upstream normalize_robust over each complete ZYX patch: percentile clip 1/99, median/MAD*1.4826 with upstream fallbacks", "blend": "Nonperiodic torch Hann outer product, normalized maximum, floored at 0.001", "review_interior_bbox_roi_xywh": [interior[1].start, interior[0].start, 128, 128], "display_rescaling_applied": False},
        "models": model_records,
        "outputs": outputs,
        "seed_stability": stability,
        "runtime": {"python": sys.version.split()[0], "packages": {name: importlib.metadata.version(name) for name in package_names}},
        "limitations": ["Scores are model outputs, not independently established ink or calibrated authenticity probabilities.", "Seed agreement is a stability diagnostic; both seeds share a training corpus.", "No target ink labels or vetted reading were used; no precision/recall or reading accuracy is established.", "Native 8.640um target sampling differs from stated 9.362um/~9.596um training sources.", "Interpretation requires source geometry, depth/orientation controls and independent stroke review.", "Mask alignment and geometric validity are caller-supplied; no independent mask correctness is inferred.", "Finite zero output where evaluated-mask=false is the empty-source policy, not a learned background confidence.", "BCE smoothing maps training targets to 0.25/0.75; any (p-0.25)/0.5 display transform must be recorded separately."]
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path, help="Retained uint8 .npy surface ROI, shape [31,H,W], 128<=H,W<=2048; explicit scheduled-tile budget applies.")
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask", type=Path, help="Optional boolean .npy geometry-validity mask aligned to [H,W]; never used in normalization.")
    parser.add_argument("--seed42-checkpoint", type=Path)
    parser.add_argument("--seed43-checkpoint", type=Path)
    parser.add_argument("--layer-start", type=int, default=7, help="First of 17 retained source layers (default: 7).")
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-total-tiles", type=int, default=DEFAULT_TOTAL_TILE_BUDGET, help="Maximum scheduled tiles across all supplied checkpoints before mask skips (default:1200; hard cap:1922). Two2048-square checkpoints require explicit1922.")
    args = parser.parse_args()
    if args.seed42_checkpoint is None and args.seed43_checkpoint is None:
        parser.error("Provide --seed42-checkpoint and/or --seed43-checkpoint.")
    if not 1 <= args.max_total_tiles <= HARD_TOTAL_TILE_BUDGET:
        parser.error(f"--max-total-tiles must be between 1 and {HARD_TOTAL_TILE_BUDGET}.")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16.")
    try:
        metadata_path = run(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"Large-surface run failed: {error}\n")
    print(metadata_path)


if __name__ == "__main__":
    main()

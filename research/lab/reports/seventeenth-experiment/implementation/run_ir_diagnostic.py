"""Frozen crop-first infrared associations, using only retained bounded PNG objects."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seventeenth-matplotlib")
import sys
sys.dont_write_bytecode = True
import argparse
import gc
import hashlib
import io
from pathlib import Path
import resource
import signal
import socket
import time
import json


def offline(*args, **kwargs):
    raise RuntimeError("Network disabled for retained infrared diagnostic")


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from PIL import Image, __version__ as pillow_version
from regional_common import checked_bytes, checked_json, record, require, verify, write_json
from regional_stats import rank_metrics
from regional_plot import make_ir_panel


PROTOCOLS = dict(protocol="08bc48f203d6722216c70d54b961d0d29ce59204359da49d2b6110661a9a9cc1",
                clarifications="22ed3e0749d56bd61113b6a2cd74f6ae7373082a34a1c090352ccba99eb2c643",
                ppm_amendment="ed3096856274505990fb9b4069ce6fdbb657eaeb93ae52b6bfae2144de6d9f9e",
                ppm_zero_clarification="114e3791f2a7549b2e1d82192ef454865224f3ea9506550a0513f4eaec591d1a")


def mean3(array):
    return array.reshape(384, 3, 384, 3).mean(axis=(1, 3), dtype=np.float64)


def load_mask(pin, dtype):
    array = np.load(io.BytesIO(checked_bytes(pin, 1024 * 1024)), allow_pickle=False)
    require(array.shape == (384, 384) and array.dtype == np.dtype(dtype), "Literal support/label shape or dtype differs")
    return array


def validate_image(body, expected_yx):
    Image.MAX_IMAGE_PIXELS = 150000000
    image = Image.open(io.BytesIO(body))
    require(image.format == "PNG", "Published infrared must be PNG")
    require([image.height, image.width] == expected_yx and image.width * image.height <= 150000000,
            "Published infrared canvas differs or exceeds pixel cap")
    require(image.mode in ("L", "RGB", "RGBA", "P"), "Unexpected infrared encoding; no silent rescale")
    image.load()
    palette = image.getpalette() if image.mode == "P" else None
    transparency = image.info.get("transparency")
    if isinstance(transparency, bytes):
        transparency = dict(kind="bytes", hex=transparency.hex())
    if image.mode == "RGBA":
        require(image.getchannel("A").getextrema() == (255, 255), "Nonopaque infrared pixels")
    elif image.mode == "P" and transparency is not None:
        rgba = image.convert("RGBA")
        require(rgba.getchannel("A").getextrema() == (255, 255), "Nonopaque palette pixels")
        del rgba
    elif image.mode in ("L", "RGB") and transparency is not None:
        rgba = image.convert("RGBA")
        require(rgba.getchannel("A").getextrema() == (255, 255), "Nonopaque keyed infrared pixels")
        del rgba
    metadata = dict(format="PNG", original_mode=image.mode, shape_yx=[image.height, image.width],
                    fully_opaque=True, palette=palette, transparency=transparency,
                    grayscale_conversion="Pillow crop.convert('L'); RGB grayscale conversion follows Pillow documented behavior",
                    no_dynamic_rescaling=True)
    return image, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    plan_pin = record(args.plan)
    plan = checked_json(plan_pin)
    require(plan["kind"] == "phase17-frozen-ir-diagnostic", "Unexpected infrared plan")
    pins = {plan_pin["path"]: plan_pin}
    for key, expected in PROTOCOLS.items():
        require(plan[key]["sha256"] == expected, "Protocol/amendment SHA differs")
    for pin in [plan[k] for k in PROTOCOLS] + plan["code"] + [plan["descriptor"], plan["raw_provenance"], plan["acquisition_result"], plan["acquisition_plan"]]:
        verify(pin)
        pins[pin["path"]] = pin
    descriptor = checked_json(plan["descriptor"])
    raw = checked_json(plan["raw_provenance"])
    acquisition = checked_json(plan["acquisition_result"])
    require(acquisition["status"] == "PASS", "Infrared acquisition is not complete")
    objects = {obj["object_id"]: obj for obj in acquisition["objects"]}
    contracts = {item["fragment"]: item for item in raw["tiff_contracts"] if item["plane_index"] == 7}
    require([c["id"] for c in descriptor["cases"]] == [f"frag{f}-context{c}" for f in (1, 2, 3) for c in (1, 2)], "Frozen six cases differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs, image_receipts = [], []
    for fragment in ("frag1", "frag2", "frag3"):
        obj = objects[fragment + "-ir"]
        image_pin = obj["file"]
        pins[image_pin["path"]] = image_pin
        body = checked_bytes(image_pin, 48 * 1024**2)
        expected_yx = [contracts[fragment]["height"], contracts[fragment]["width"]]
        image, image_metadata = validate_image(body, expected_yx)
        image_receipts.append(dict(object_id=obj["object_id"], source=obj, encoding=image_metadata))
        del body
        for case in (case for case in descriptor["cases"] if case["id"].split("-", 1)[0] == fragment):
            arrays = case["arrays"]
            for key in ("labels", "primary_mask", "surface_mask", "known_mask"):
                pins[arrays[key]["path"]] = arrays[key]
                verify(arrays[key])
            labels = load_mask(arrays["labels"], "uint8")
            primary = load_mask(arrays["primary_mask"], "bool")
            surface = load_mask(arrays["surface_mask"], "bool")
            known = load_mask(arrays["known_mask"], "bool")
            require(np.isin(labels, [0, 1]).all() and np.all(primary <= surface) and np.all(primary <= known), "Literal support relationships differ")
            x, y, width, height = case["bbox_source_xywh"]
            require(case["native_bbox_yxhw"] == [y, x, height, width] and (width, height) == (1152, 1152), "Frozen native crop differs")
            require(0 <= x < x + width <= image.width and 0 <= y < y + height <= image.height, "Infrared crop escapes canvas")
            gray_native = np.asarray(image.crop((x, y, x + width, y + height)).convert("L"), dtype=np.uint8)
            gray64 = gray_native.astype(np.float64)
            gray_coarse = mean3(gray64)
            darkness = 255.0 - gray_coarse
            local = mean3(gaussian_filter(gray64, sigma=16, truncate=4, mode="reflect") - gray64)
            case_dir = args.output_dir / case["id"]
            case_dir.mkdir()
            maps = {}
            for key, array in (("gray", gray_coarse), ("darkness", darkness), ("local_darkness", local)):
                path = case_dir / (key + ".npy")
                with path.open("xb") as handle:
                    np.save(handle, array, allow_pickle=False)
                maps[key] = record(path)
            panel = case_dir / "ir-panel.png"
            make_ir_panel(panel, case, gray_native, gray_coarse, local,
                          dict(labels=labels, primary_mask=primary, surface_mask=surface))
            output = dict(id=case["id"], fragment=case["fragment"], role=case["role"],
                          bbox_source_xywh=case["bbox_source_xywh"], source_object_id=obj["object_id"],
                          arrays=maps, panel=record(panel),
                          darkness=rank_metrics(darkness[primary], labels[primary]),
                          local_darkness=rank_metrics(local[primary], labels[primary]),
                          primary_support_count=int(primary.sum()),
                          definitions=dict(darkness="255 - 3x3 mean gray", local_darkness="3x3 mean of gaussian_filter(gray64,sigma=16,truncate=4,mode='reflect') - gray64; crop first"))
            outputs.append(output)
            write_json(case_dir / "association.json", output)
            print(json.dumps(dict(case_id=case["id"], status="PASS", wall_seconds=time.perf_counter()-started)), flush=True)
        image.close()
        del image
        gc.collect()
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 3 * 1024**3, "Peak RSS exceeds limit")
    for pin in pins.values():
        verify(pin)
    result = dict(status="PASS", plan=plan_pin, images=image_receipts, cases=outputs,
                  all_inputs_unchanged_before_after=True, network_calls=0, model_calls=0, checkpoint_reads=0, gpu_seconds=0,
                  command=[sys.executable, *sys.argv], wall_seconds=time.perf_counter()-started,
                  cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pillow=pillow_version, threads=4),
                  resource_limits=dict(wall_seconds=600,cpu_seconds=1200,threads=4,peak_rss_bytes=3*1024**3,address_space_bytes=12*1024**3,output_bytes=256*1024**2),
                  interpretation="Agreement with supplied IR-derived labels; not independent registration accuracy or CT ink detection")
    write_json(args.output_dir / "input-integrity.json", dict(status="PASS", inputs=list(pins.values())))
    write_json(args.output_dir / "ir-diagnostics.json", result)
    require(sum(p.stat().st_size for p in args.output_dir.rglob("*") if p.is_file()) <= 256*1024**2, "New retained output cap exceeded")
    require(time.perf_counter()-started <= 600, "Wall cap exceeded")
    signal.alarm(0)
    print(json.dumps(dict(status="PASS", output=record(args.output_dir / "ir-diagnostics.json"))), flush=True)


if __name__ == "__main__":
    main()

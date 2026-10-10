"""Independent saved-output NumPy audit; source/synthetic preparation first.

No author module, Torch, model or checkpoint is imported or read. Explicit root
authorization plus entry hashes are required before reading completed outputs
or scientific arrays. CT byte reconstruction belongs to the separate CT audit.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import resource
import signal
import socket
import sys

import numpy as np


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
ROLES = ["frozen_seed42", "A_preserve", "B_preserve"]
CALIBRATION = ["frag3-context1", "frag3-context2"]
EVALUATION = ["frag1-context1", "frag1-context2", "frag2-context1", "frag2-context2"]
CASES = CALIBRATION + EVALUATION
PHASES = ["calibration", "evaluation"]
RULES = ["retained_fourteenth", "frag3_only"]
RANK_TOLERANCE = 1e-12
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
SOURCE_PIN_SHA = "280445f79ee921263657bc7ffb85a4f558df9e0a351ce607dee535e207bf067d"
DESIGN_SHA = "34fd5a0312240a1e27fc5fafa2727c1205e9b2283148fbe27d9a4d70b63f6778"
ACCOUNTING_SHA = "7d263c32041bca7edb99462f8205e73b641c0ba4620fb68ff02b58154271f14b"
BENCHMARK_SHA = "a4026cce7ca55d5aaeeaa0736061ab6b7c8aa13875155682d24e74c139fe5735"
RUNTIME_SHA = "530aba06cf9f2fde2c16a18731457631416f1f3cdd5c7ec66181ec1c0efef7db"
BLEND = dict(path=str(HERE.parent / "blending/blend_verifier.py"), size_bytes=9710,
             sha256="4c1f4c8dcafcd8928b358477fc3239e2a9448911d8f1c6c1d1ad89bef30fb5de")
MATH = dict(path=str(HERE.parent / "benchmark_math.py"), size_bytes=9789,
            sha256="0178660db9885727c0c66feaf8b431024584e8bb0429edf7a08afa4f3dc6e594")
PREPROCESSING = dict(source_planes_inclusive=[7, 57], depth_group_size=3, xy_group_size=3,
    sum_dtype="uint32", division_dtype="float64", output_dtype="float32", scale_divisor=257,
    formula="float32(uint32_sum_of_27_source_uint16_values / float64(27*257))",
    source_group_centers=list(range(8, 57, 3)), center_source_plane=32,
    pixel_spacing_um=9.72, integer_quantization=False)
INFERENCE = dict(layer_start=0, direction="forward", patch_shape=[17, 128, 128], stride=64,
    normalization="unchanged robust_mad1/99 over full17x128x128 patch",
    mask_passed_to_adapter=False, tiles_per_context=25)
LIMITS = dict(threads=4, worker_wall_seconds=120, worker_cpu_seconds=360,
    total_wall_seconds=600, rss_bytes=3 * 1024**3, address_space_bytes=12 * 1024**3,
    maximum_tile_forwards=450, output_bytes=96 * 1024**2, retries=0)
LOADS = dict(final_role_worker_constructions=6, initializer_strict_loads=6,
             candidate_raw_state_strict_loads=4, total_load_state_dict_calls=10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def offline(*args, **kwargs):
    raise RuntimeError("Network disabled in independent saved-output review")


socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline


def identity(item):
    require(isinstance(item, dict) and isinstance(item.get("path"), str), "File record path is missing")
    sizes = [item[k] for k in ("size_bytes", "bytes", "byte_length") if k in item]
    require(sizes and all(type(n) is int and n >= 0 and n == sizes[0] for n in sizes),
            "File record size aliases differ")
    require(isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "File record SHA256 differs")
    return dict(path=str(Path(item["path"]).resolve()), size_bytes=sizes[0], sha256=item["sha256"])


def record(path):
    path = Path(path).resolve()
    require(path.suffix.lower() not in CHECKPOINT_SUFFIXES, "Checkpoint/raw binary reads are prohibited")
    require(path.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".npy"}, "Unapproved file type")
    require(path.is_file(), "Expected a regular file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def unique_object(pairs):
    value = {}
    for key, child in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = child
    return value


def reject_constant(value):
    raise ValueError("Nonfinite JSON numeric constant: " + value)


def decode_json(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


class PinnedIO:
    def __init__(self):
        self.pins = {}
        self.array_reads = 0

    def verify(self, item):
        expected = identity(item)
        require(expected["size_bytes"] <= 64 * 1024**2, "Single review file exceeds size cap")
        actual = record(expected["path"])
        require(actual == expected, "File identity differs: " + expected["path"])
        require(expected["path"] not in self.pins or self.pins[expected["path"]] == expected,
                "Conflicting input pins")
        self.pins[expected["path"]] = expected
        return Path(expected["path"])

    def json(self, item):
        path = self.verify(item)
        require(path.suffix == ".json", "Expected JSON receipt")
        return decode_json(path.read_text())

    def jsonl(self, item):
        path = self.verify(item)
        require(path.suffix == ".jsonl", "Expected JSONL phase log")
        lines = path.read_text().splitlines()
        require(len(lines) == 12 and all(lines), "Exactly twelve phase events required")
        return [decode_json(line) for line in lines]

    def array(self, item):
        path = self.verify(item)
        require(path.suffix == ".npy", "Expected saved NumPy array")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        self.array_reads += 1
        return value

    def discovered_json(self, path):
        pin = record(path)
        return self.json(pin), pin

    def finish(self):
        for item in self.pins.values():
            require(record(item["path"]) == item, "Input changed during independent review")
        return list(self.pins.values())


def pinned_module(item, name):
    expected = identity(item)
    require(expected["path"] in {BLEND["path"], MATH["path"], str(HERE / "receipt_checks.py")},
            "Only independent frozen review modules may be imported")
    require(record(expected["path"]) == expected, "Independent arithmetic source changed")
    spec = importlib.util.spec_from_file_location(name, expected["path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require("torch" not in sys.modules, "Independent arithmetic loaded Torch")
    return module


def compare(expected, actual, where="value", counts=None):
    """Exact counts/types/None/structure; fixed float64 arithmetic tolerance."""
    if counts is None:
        counts = {"scalar_comparisons": 0, "maximum_absolute_float_error": 0.}
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Dictionary fields differ at " + where)
        for key, value in expected.items():
            compare(value, actual[key], where + "." + str(key), counts)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List structure differs at " + where)
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, where + "[" + str(i) + "]", counts)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual), "Finite numeric value required at " + where)
        error = abs(expected - float(actual))
        require(error <= RANK_TOLERANCE, "Float arithmetic differs at " + where)
        counts["maximum_absolute_float_error"] = max(counts["maximum_absolute_float_error"], error)
        counts["scalar_comparisons"] += 1
    else:
        require(type(actual) is type(expected) and actual == expected, "Exact value/type differs at " + where)
        counts["scalar_comparisons"] += 1
    return counts


def same_record(left, right):
    require(identity(left) == identity(right), "File record lineage differs")


def validate_arrays(case, arrays):
    require(set(arrays) == {"surface", "labels", "primary_mask", "known_mask", "surface_mask"},
            "Exactly five original input array roles required")
    surface, labels = arrays["surface"], arrays["labels"]
    require(surface.dtype == np.float32 and surface.shape == (17, 384, 384) and
            np.isfinite(surface).all() and ((surface >= 0.) & (surface <= 255.)).all(),
            "Original pooled FP32 surface contract differs")
    require(labels.shape == (384, 384) and labels.dtype in (np.dtype("uint8"), np.dtype("bool")) and
            np.isin(labels, [0, 1]).all(), "Literal supplied label encoding differs")
    for key in ("primary_mask", "known_mask", "surface_mask"):
        require(arrays[key].dtype == np.bool_ and arrays[key].shape == (384, 384), "Boolean support contract differs")
    known, primary = arrays["known_mask"], arrays["primary_mask"]
    require(np.all(~known | arrays["surface_mask"]), "Known labels escape declared surface")
    yy, xx = np.indices((384, 384))
    interior = (yy >= 32) & (yy < 352) & (xx >= 32) & (xx < 352)
    require(np.array_equal(primary, known & interior), "Literal fixed 32-pixel primary support differs")
    positive = int((primary & (labels == 1)).sum())
    negative = int((primary & (labels == 0)).sum())
    require(positive == case["primary_positive"] and negative == case["primary_negative"] and
            positive >= 512 and negative >= 8192, "Frozen primary class counts/gates differ")
    return dict(pixels=positive + negative, positive=positive, negative=negative,
                known_pixels=int(known.sum()), surface_pixels=int(arrays["surface_mask"].sum()),
                primary_mask_exact=True)


def validate_plan(plan, plan_record, io):
    require(plan["kind"] == "Paris-fragments-fixed-state-calibration-transfer-v1" and
            plan["ready_for_execution"] is True and plan["roles"] == ROLES and
            [case["id"] for case in plan["cases"]] == CASES, "Ready plan scope/order differs")
    compare(PREPROCESSING, plan["preprocessing"], "plan.preprocessing")
    compare(INFERENCE, plan["inference"], "plan.inference")
    compare(LIMITS, plan["limits"], "plan.limits")
    compare(LOADS, plan["planned_model_load_operations"], "plan.loads")
    require(plan["source_pins"]["sha256"] == SOURCE_PIN_SHA and
            plan["research_design"]["sha256"] == DESIGN_SHA and
            plan["runtime"]["sha256"] == RUNTIME_SHA and
            plan["load_accounting_clarification"]["sha256"] == ACCOUNTING_SHA,
            "Frozen entry source/design/runtime/accounting pin differs")
    source = io.json(plan["source_pins"])
    io.verify(plan["research_design"])
    io.verify(plan["load_accounting_clarification"])
    require(plan["models"] == source["models"], "Model/state/retained-cutoff metadata differs")
    for key in ("initializer", "helper", "shared", "legacy_prepare", "legacy_worker", "cutoff", "metrics",
                "transitive_runtime_inputs", "load_accounting_clarification"):
        require(plan[key] == source[key], "Source metadata lineage differs: " + key)
    environment = io.json(plan["runtime"])
    require(list(sys.version_info[:3]) == environment["python"] and np.__version__ == environment["packages"]["numpy"],
            "NumPy/Python runtime differs")
    for name, version in environment["packages"].items():
        require(importlib.metadata.version(name) == version, "Installed package metadata differs: " + name)
    require(environment["cpu_only"] is True and environment["threads"] == 4, "CPU/thread runtime pin differs")
    freeze = io.json(plan["code_freeze"])
    require(freeze["no_further_edits"] is True, "Author source package is not sealed")
    same_record(freeze["research_design"], plan["research_design"])
    same_record(freeze["load_accounting_clarification"], plan["load_accounting_clarification"])
    executable = {}
    for item in freeze["files"]:
        path = io.verify(item)
        if path.suffix == ".py":
            executable[path.name] = identity(item)
    require(plan["code"] == executable, "Plan executable sources differ from sealed source")
    go, ct = io.json(plan["source_go"]), io.json(plan["ct_verification"])
    require(go["status"] == "GO" and ct["status"] == "PASS" and ct["all_CT_arrays_exact"] is True,
            "Source GO or exact independent CT PASS is missing")
    same_record(go["code_freeze"], plan["code_freeze"])
    same_record(go["research_design"], plan["research_design"])
    same_record(ct["descriptor"], plan["descriptor"])
    descriptor = io.json(plan["descriptor"])
    require(descriptor["status"] == "pooled_reference_complete_no_inference" and
            all(descriptor["preprocessing"].get(k) == v for k, v in PREPROCESSING.items()), "CT descriptor contract differs")
    described = {case["id"]: case for case in descriptor["cases"]}
    require(len(described) == 6 and set(described) == set(CASES), "CT descriptor case scope differs")
    require(source["benchmark_contract"]["sha256"] == BENCHMARK_SHA, "Original support contract pin differs")
    benchmark = io.json(source["benchmark_contract"])
    original = {row["case_id"]: row for row in benchmark["selected_contexts"]}
    require(set(original) == set(CASES), "Literal benchmark case scope differs")
    names = dict(labels="labels.npy", primary_mask="primary-mask.npy", known_mask="known-mask.npy", surface_mask="surface-mask.npy")
    for case in plan["cases"]:
        row, desc = original[case["id"]], described[case["id"]]
        for key in ("role", "fragment", "parent_papyrus", "native_bbox_yxhw", "bbox_source_xywh", "primary_positive", "primary_negative"):
            compare(row[key], case[key], "case." + key)
        require(case["role"] == ("calibration" if case["id"] in CALIBRATION else "evaluation"), "Calibration/evaluation role differs")
        require(set(case["arrays"]) == set(desc["arrays"]) == {"surface", *names}, "CT/support array roles differ")
        compare(case["native_bbox_yxhw"], desc["native_bbox_yxhw"], "descriptor.origin")
        for key, item in case["arrays"].items():
            same_record(item, desc["arrays"][key])
            if key in names:
                same_record(item, row["files"][names[key]])
    return source, descriptor


def reconstruct_metrics(plan, plan_record, results_record, barrier_record, thresholds, per_context, arithmetic):
    fragments = {}
    for fragment in dict.fromkeys(case["fragment"] for case in plan["cases"]):
        ids = [case["id"] for case in plan["cases"] if case["fragment"] == fragment]
        require(len(ids) == 2, "Exactly two contexts per fragment required")
        fragments[fragment] = dict(context_ids=ids, models={role: arithmetic.macro([per_context[k]["models"][role] for k in ids])
                                                         for role in ROLES})
    paris2 = {role: arithmetic.macro([fragments[f]["models"][role] for f in ("PHercParis2Fr47", "PHercParis2Fr143")])
              for role in ROLES}
    screens = {role: {rule: arithmetic.transfer_screen({case: per_context[case]["models"][role]["operating"][rule]
                                                     for case in EVALUATION}) for rule in RULES} for role in ROLES}
    differences = {scope: {role: arithmetic.delta(rows[role], rows["frozen_seed42"]) for role in ROLES[1:]}
                   for scope, rows in [("Paris2_equal_fragment", paris2), *[(f, row["models"]) for f, row in fragments.items()]]}
    return dict(status="PASS", plan=plan_record, results=results_record, locks_ready=barrier_record,
        roles=ROLES, thresholds=thresholds, per_context=per_context, per_fragment=fragments,
        paris2_equal_fragment=paris2, candidate_minus_baseline=differences,
        specificity_transfer_checks=screens, primary_specificity_transfer_rule="frag3_only",
        primary_specificity_transfer_all_four={role: screens[role]["frag3_only"]["all_four"] for role in ROLES},
        automatic_promotion=False, model_calls=0, threshold_fitting_calls=0,
        interpretation="Conditional supplied-reference agreement; Paris2 is one parent, labels gate-selected, upstream exposure unknown; no significance/CI/promotion.",
        all_bound_inputs_unchanged_before_after=True)


def completed_review(plan_pin, result_pin, metric_pin, freeze):
    io = PinnedIO()
    blend = pinned_module(BLEND, "sixteenth_independent_blend")
    arithmetic = pinned_module(MATH, "sixteenth_independent_benchmark")
    receipts = pinned_module(freeze["receipt_checker"], "sixteenth_independent_receipts")
    io.verify(BLEND); io.verify(MATH); io.verify(freeze["receipt_checker"])
    plan = io.json(plan_pin)
    validate_plan(plan, plan_pin, io)
    results, observed_metrics = io.json(result_pin), io.json(metric_pin)
    require(results["status"] == observed_metrics["status"] == "PASS" and
            results["plan"] == observed_metrics["plan"] == plan_pin, "Completed root run/analysis lineage differs")
    workers = {phase: {role: io.json(results["models"][phase][role]) for role in ROLES} for phase in PHASES}
    barrier_pin = identity(results["locks_ready"])
    barrier = io.json(barrier_pin)
    events = io.jsonl(results["phase_order"])
    execution = receipts.verify_execution_receipts(plan, plan_pin, results, result_pin, barrier, barrier_pin, workers, events)
    locks = {role: io.json(barrier["threshold_locks"][role]) for role in ROLES}
    thresholds = {role: {"retained_fourteenth": plan["models"][role]["original_cutoff"], "frag3_only": locks[role]["primary"]}
                  for role in ROLES}
    per_context, map_checks, input_checks, regional = {}, [], [], {role: {} for role in ROLES}
    comparisons = {"scalar_comparisons": 0, "maximum_absolute_float_error": 0.}
    for case in plan["cases"]:
        arrays = {name: io.array(item) for name, item in case["arrays"].items()}
        support = validate_arrays(case, arrays)
        labels = arrays["labels"][arrays["primary_mask"]].astype(np.bool_)
        phase, models, reference_inputs = case["role"], {}, None
        for role in ROLES:
            item = workers[phase][role]["runs"][case["id"]]
            same_record(item["fixed_primary_mask"], case["arrays"]["primary_mask"])
            require(item["normalization"] == INFERENCE["normalization"] and
                    item["input_fingerprint_before_model_call"] is True and item["model_input_tensor_unchanged"] is True and
                    item["raw_logits_semantics"] == "FP32 pre-sigmoid scored-tile logits in row-major schedule; no blended logit mosaic",
                    "Capture/preprocessing semantics differ")
            metadata_path = Path(item["scores"]["path"]).resolve().parent / "metadata.json"
            metadata, metadata_pin = io.discovered_json(metadata_path)
            compare(item, metadata, "metadata", comparisons)
            scores, evaluated, logits = [io.array(item[name]) for name in ("scores", "evaluated", "logits")]
            check = blend.verify_map(logits, scores, evaluated, (384, 384), item)
            check.update(role=role, context=case["id"], metadata=metadata_pin,
                         source_records={name: item[name] for name in ("scores", "evaluated", "logits")})
            map_checks.append(check)
            require(evaluated[arrays["primary_mask"]].all(), "Fixed primary support is not completely evaluated")
            if reference_inputs is None:
                reference_inputs = blend.verify_normalized_inputs(arrays["surface"], (384, 384), item)
            compare(reference_inputs["records"], item["captured_tiles"], "input_hashes", comparisons)
            input_checks.append(dict(role=role, context=case["id"], tiles=25, normalized_input_hashes_exact=True))
            selected_scores = scores[arrays["primary_mask"]]
            ranking = arithmetic.ranking(labels, selected_scores)
            models[role] = dict(auc=ranking["auc"], ap=ranking["ap"], supplied_positive_prevalence=float(labels.mean()),
                operating={rule: arithmetic.derived_operating(labels, selected_scores, value) for rule, value in thresholds[role].items()},
                scores=item["scores"], evaluated=item["evaluated"], logits=item["logits"])
            if phase == "calibration":
                regional[role][case["id"]] = arithmetic.select_negatives(selected_scores[~labels])
                actual = locks[role]["regional"][case["id"]]
                compare(regional[role][case["id"]], actual, "regional_cutoff", comparisons)
                for field in ("threshold", "excluded_negative_boundary"):
                    require(arithmetic.float32_bits(regional[role][case["id"]][field]) == arithmetic.float32_bits(actual[field]),
                            "Exact float32 cutoff/boundary bits differ")
            del scores, evaluated, logits, selected_scores
        per_context[case["id"]] = dict(fragment=case["fragment"], parent_papyrus=case["parent_papyrus"], role=case["role"],
            fixed_primary_mask=case["arrays"]["primary_mask"], pixel_count=support["pixels"], positive=support["positive"],
            negative=support["negative"], models=models,
            candidate_minus_baseline={role: arithmetic.delta(models[role], models["frozen_seed42"]) for role in ROLES[1:]})
        del arrays, labels
    for role in ROLES:
        lock = locks[role]
        require(lock["role"] == role and lock["plan"] == plan_pin and
                lock["state_sha256"] == plan["models"][role]["state_sha256"] and
                lock["evaluation_predictions"] == 0 and lock["selector"] == plan["cutoff"] and
                lock["original_cutoff"] == plan["models"][role]["original_cutoff"] and
                lock["scope"] == "Only two Frag3 primary negative vectors; no Frag1/2 inference", "Cutoff lock lineage differs")
        expected = arithmetic.lock_values(regional[role])["primary"]
        require(arithmetic.float32_bits(expected) == arithmetic.float32_bits(lock["primary"]), "Exact max regional cutoff bits differ")
        for rule in RULES:
            require(arithmetic.float32_bits(thresholds[role][rule]) == arithmetic.float32_bits(observed_metrics["thresholds"][role][rule]),
                    "Reported cutoff bits differ")
    rebuilt = reconstruct_metrics(plan, plan_pin, result_pin, barrier_pin, thresholds, per_context, arithmetic)
    compare(rebuilt, observed_metrics, "all_reported_metrics", comparisons)
    require(len(map_checks) == len(input_checks) == 18 and sum(row["tiles_verified"] for row in map_checks) == 450,
            "Complete 18-map/450-capture scope differs")
    for path, cap in ((Path(result_pin["path"]).parent, 96 * 1024**2), (Path(metric_pin["path"]).parent, 16 * 1024**2)):
        require(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= cap, "Retained output byte cap exceeded")
    pins = io.finish()
    require("torch" not in sys.modules, "Independent review loaded Torch")
    return dict(status="PASS", plan=plan_pin, results=result_pin, metrics=metric_pin, execution_receipts=execution,
        maps_verified=18, captured_logits_verified=450, normalized_input_hashes_verified=450,
        all_evaluated_masks_exact=True, all_normalized_input_hashes_exact=True,
        maximum_absolute_score_error=max(row["maximum_absolute_score_error"] for row in map_checks),
        maximum_absolute_score_error_limit=2e-6, map_checks=map_checks, normalized_input_checks=input_checks,
        all_cutoff_bits_exact=True, all_reported_metrics_reconstructed=True, arithmetic_comparisons=comparisons,
        verified_inputs=pins, array_reads=io.array_reads, all_input_pins_unchanged_before_after=True,
        model_calls=0, Torch_imports=0, checkpoint_reads=0, network_requests=0,
        limitation="Saved-array arithmetic and pinned operation/state receipts do not independently execute/authenticate the model weights, CT acquisition, annotations or unseen exposure.")


def root_entry(path, sha):
    pin = record(path)
    require(pin["sha256"] == sha, "Root-authorized entry SHA256 differs")
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-saved-output-review", action="store_true")
    parser.add_argument("--source-freeze-sha256", required=True)
    for name in ("plan", "results", "metrics"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.authorized_saved_output_review, "Explicit root completed-output review authorization is required")
    require(not args.output.exists(), "Independent completed-review output must be fresh")
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3,) * 2)
    freeze_pin = root_entry(HERE / "source-freeze.json", args.source_freeze_sha256)
    freeze = decode_json(Path(freeze_pin["path"]).read_text())
    require(freeze["status"] == "SEALED_SOURCE_SYNTHETIC_PREPARATION", "Independent preparation is not sealed")
    for item in freeze["executable_sources"]:
        require(record(item["path"]) == identity(item), "Independent checker source differs after freeze")
    require(record(freeze["preferred_synthetic_receipt"]["path"]) == identity(freeze["preferred_synthetic_receipt"]),
            "Frozen synthetic PASS receipt changed")
    plan = root_entry(args.plan, args.plan_sha256)
    results = root_entry(args.results, args.results_sha256)
    metrics = root_entry(args.metrics, args.metrics_sha256)
    value = completed_review(plan, results, metrics, freeze)
    require(record(freeze_pin["path"]) == freeze_pin, "Independent source freeze changed during review")
    value["independent_source_freeze"] = freeze_pin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    signal.alarm(0)
    print(json.dumps(dict(status="PASS", maps=18, captures=450, verification=record(args.output)), sort_keys=True))


if __name__ == "__main__":
    main()

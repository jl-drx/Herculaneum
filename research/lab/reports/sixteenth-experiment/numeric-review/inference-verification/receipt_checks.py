"""Pure in-memory checks of retained sixteenth execution receipts.

No file or array I/O, author imports, checkpoint reads, model execution or
network. The caller authenticates every input file pin before passing JSON
objects, rejects duplicate JSON keys, and separately verifies arrays and cutoff contents. Receipt agreement
does not authenticate replay of model weights or prove absence of hidden calls.
"""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import posixpath
import re


ROLES = ("frozen_seed42", "A_preserve", "B_preserve")
PHASES = ("calibration", "evaluation")
CASE_IDS = {
    "calibration": ("frag3-context1", "frag3-context2"),
    "evaluation": ("frag1-context1", "frag1-context2", "frag2-context1", "frag2-context2"),
}
TOTAL_LOADS = {
    "final_role_worker_constructions": 6, "initializer_strict_loads": 6,
    "candidate_raw_state_strict_loads": 4, "total_load_state_dict_calls": 10,
}
LIMITS = {
    "threads": 4, "worker_wall_seconds": 120, "worker_cpu_seconds": 360,
    "total_wall_seconds": 600, "rss_bytes": 3 * 1024**3,
    "address_space_bytes": 12 * 1024**3, "maximum_tile_forwards": 450,
    "output_bytes": 96 * 1024**2, "retries": 0,
}
STORAGE = {
    "named_tensor_bytes": 272701704, "unique_storage_bytes": 138185992,
    "duplicate_named_bytes": 134515712, "tensor_count": 508,
    "unique_storage_count": 264, "alias_group_count": 244,
    "unique_storage_cap_bytes": 160 * 1024**2,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(value, label):
    require(type(value) is dict, label + " must be a JSON dictionary")
    return value


def _integer(value, label, expected=None, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be a nonnegative literal integer")
    if expected is not None:
        require(value == expected, label + " differs")
    return value


def _number(value, label, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            label + " must be a finite nonnegative JSON number")
    if maximum is not None:
        require(value <= maximum, label + " exceeds its frozen limit")
    return float(value)


def _sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            label + " must be a lowercase SHA256")
    return value


def _record(value, label):
    value = _mapping(value, label)
    path = value.get("path")
    require(type(path) is str and path.startswith("/") and posixpath.normpath(path) == path
            and "\x00" not in path and "\n" not in path, label + " has a noncanonical absolute path")
    _integer(value.get("size_bytes"), label + " size_bytes", minimum=1)
    _sha(value.get("sha256"), label + " sha256")
    return {key: value[key] for key in ("path", "size_bytes", "sha256")}


def _same_record(value, expected, label):
    require(_record(value, label) == _record(expected, label + " expected"), label + " pin differs")


def _utc(value, label):
    require(type(value) is str, label + " must be an ISO UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(label + " has malformed UTC") from error
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), label + " must identify UTC")
    return parsed


def _loads(value, expected, label):
    value = _mapping(value, label)
    require(set(value) == set(expected), label + " keys differ")
    for key, count in expected.items():
        _integer(value[key], label + " " + key, count)


def _storage(value, label, schema=None):
    value = _mapping(value, label)
    for key, expected in STORAGE.items():
        _integer(value.get(key), label + " " + key, expected)
    for key in ("alias_partition_matches_factory", "bitwise_alias_values_consistent"):
        require(value.get(key) is True, label + " " + key + " is not true")
    groups = value.get("alias_groups")
    require(type(groups) is list and len(groups) == 244, label + " alias list differs")
    seen = set()
    byte_sum = 0
    first_keys = []
    rows = {row["key"]: row for row in schema} if schema is not None else None
    for group in groups:
        group = _mapping(group, label + " alias group")
        keys = group.get("keys")
        require(type(keys) is list and len(keys) == 2 and
                all(type(key) is str and key for key in keys) and keys == sorted(set(keys)) and
                not seen.intersection(keys), label + " alias key partition differs")
        seen.update(keys)
        first_keys.append(keys[0])
        byte_sum += _integer(group.get("storage_bytes"), label + " alias storage bytes", minimum=1)
        require(group.get("exact_same_view") is True and group.get("bitwise_equal") is True,
                label + " alias view/value receipt is not true")
        if rows is not None:
            require(all(key in rows for key in keys), label + " alias key absent from tensor schema")
            one, two = (rows[key] for key in keys)
            require(all(one[field] == two[field] for field in ("shape", "dtype", "numel", "bytes"))
                    and one["bytes"] == group["storage_bytes"], label + " alias tensor schema differs")
    require(first_keys == sorted(first_keys) and byte_sum == STORAGE["duplicate_named_bytes"],
            label + " alias order or byte accounting differs")
    return value


def _initializer(value, plan):
    value = _mapping(value, "Initializer contract")
    initial = plan["initializer"]
    _same_record(value, initial, "Initializer artifact")
    for actual, expected in (("label", "label"), ("model_repository", "repository"),
                             ("model_revision", "revision"), ("repository_path", "repository_path")):
        require(value.get(actual) == initial.get(expected), "Initializer provenance differs: " + actual)
    require(value.get("deserialization") == "torch.load(map_location='cpu', weights_only=True)" and
            value.get("strict_state_match") is True and value.get("missing_keys") == [] and
            value.get("unexpected_keys") == [], "Initializer safe strict-load receipt differs")
    require(value.get("weight_container") == "model",
            "Initializer selected weight container differs")
    _integer(value.get("step"), "Initializer embedded step", 74999)
    _integer(value.get("filename_step_label"), "Initializer filename step", 75000)
    _integer(initial.get("embedded_step"), "Plan initializer embedded step", 74999)
    _integer(initial.get("filename_step_label"), "Plan initializer filename step", 75000)
    _integer(value.get("parameter_count"), "Initializer parameters", minimum=1)
    require(value["parameter_count"] <= 40000000, "Initializer parameter cap differs")
    _integer(value.get("selected_state_bytes"), "Initializer named bytes", STORAGE["named_tensor_bytes"])
    _integer(value.get("selected_unique_storage_bytes"), "Initializer unique bytes", STORAGE["unique_storage_bytes"])
    schema = value.get("tensor_schema")
    require(type(schema) is list and len(schema) == 508, "Initializer tensor schema count differs")
    keys = []
    named_bytes = 0
    for row in schema:
        row = _mapping(row, "Initializer tensor schema row")
        require(set(row) == {"key", "shape", "dtype", "numel", "bytes"} and
                type(row["key"]) is str and row["key"], "Initializer tensor row keys differ")
        keys.append(row["key"])
        shape = row["shape"]
        require(type(shape) is list and all(type(v) is int and v > 0 for v in shape),
                "Initializer tensor shape differs")
        require(row["dtype"] == "torch.float32", "Initializer tensor dtype differs")
        _integer(row["numel"], "Initializer tensor numel", math.prod(shape), minimum=1)
        named_bytes += _integer(row["bytes"], "Initializer tensor bytes", 4 * row["numel"], minimum=1)
    require(keys == sorted(set(keys)) and named_bytes == STORAGE["named_tensor_bytes"],
            "Initializer tensor key order or named-byte sum differs")
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    require(_sha(value.get("tensor_schema_sha256"), "Initializer schema hash") == hashlib.sha256(encoded).hexdigest(),
            "Initializer canonical tensor-schema SHA256 differs")
    _storage(value.get("storage_contract"), "Initializer storage", schema)
    config = _mapping(value.get("embedded_config"), "Initializer embedded config")
    require(config.get("mode") == "flat" and config.get("model_type") == "vesuvius_unet_3d_stem_2d" and
            config.get("patch_size") == [17, 128, 128] and config.get("crop_size", [17, 128, 128]) == [17, 128, 128] and
            type(config.get("in_channels")) is int and config["in_channels"] == 1,
            "Initializer architecture/input receipt differs")
    require(config.get("model_config") == {"autoconfigure": True, "basic_encoder_block": "BasicBlockD",
            "basic_decoder_block": "ConvBlock", "stem_channels": 16, "z_projection_mode": "none"} and
            config.get("targets") == {"ink": {"activation": "none", "out_channels": 1, "z_projection_mode": "none"}} and
            config.get("image_normalization") == {"mode": "robust_mad", "percentile_lower": 1., "percentile_upper": 99.},
            "Initializer factory/target/normalizer receipt differs")
    loss = _mapping(config.get("loss"), "Initializer loss config")
    ema = _mapping(config.get("ema", {}), "Initializer EMA config")
    require(loss.get("bce_label_smoothing") == .5 and loss.get("dice_label_smoothing") == 0 and
            not config.get("use_stitched_forward", False) and not ema.get("enabled", False),
            "Initializer smoothed-score configuration differs")
    require(value.get("model_type") == "vesuvius_unet_3d_stem_2d", "Initializer model type differs")
    runtime_inputs = plan.get("transitive_runtime_inputs")
    require(type(runtime_inputs) is list, "Transitive runtime records are missing")
    records = [_record(item, "Transitive runtime input") for item in runtime_inputs]
    expected = {posixpath.basename(item["path"]): item for item in records}
    require({"run_ink_reference.py", "source-manifest.json"} <= set(expected),
            "Initializer adapter/factory source records are missing")
    require(value.get("reference_adapter_sha256") == expected["run_ink_reference.py"]["sha256"] and
            value.get("source_manifest_sha256") == expected["source-manifest.json"]["sha256"],
            "Initializer adapter/factory source receipt differs")
    return value


def verify_execution_receipts(plan, plan_record, results, results_record,
                              barrier, barrier_record, workers, events):
    """Validate already loaded JSON receipts; no paths are opened or followed."""
    for value, label in ((plan, "Plan"), (results, "Results"), (barrier, "Barrier"), (workers, "Workers")):
        _mapping(value, label)
    plan_record = _record(plan_record, "Plan record")
    results_record = _record(results_record, "Results record")
    barrier_record = _record(barrier_record, "Barrier record")
    require(plan.get("kind") == "Paris-fragments-fixed-state-calibration-transfer-v1" and
            plan.get("ready_for_execution") is True and plan.get("roles") == list(ROLES), "Frozen ready plan scope differs")
    _loads(plan.get("limits"), LIMITS, "Frozen limits")
    _loads(plan.get("planned_model_load_operations"), TOTAL_LOADS, "Planned loads")
    require(type(plan.get("cases")) is list and all(type(case) is dict for case in plan["cases"]) and
            [case.get("id") for case in plan["cases"]] ==
            list(CASE_IDS["calibration"] + CASE_IDS["evaluation"]), "Frozen case order differs")
    models = _mapping(plan.get("models"), "Plan models")
    require(set(models) == set(ROLES), "Fixed model roles differ")
    for role in ROLES:
        model = _mapping(models[role], "Plan role model")
        require(model.get("role") == role, "Plan model role differs")
        _sha(model.get("state_sha256"), "Plan model state")
        if role == "frozen_seed42":
            require(model.get("checkpoint") is None, "Baseline must use only the retained initializer")
        else:
            _record(model.get("checkpoint"), "Candidate checkpoint record")
    require(results.get("status") == barrier.get("status") == "PASS", "Completed results/barrier must PASS")
    _same_record(results.get("plan"), plan_record, "Results plan")
    _same_record(barrier.get("plan"), plan_record, "Barrier plan")
    _same_record(results.get("locks_ready"), barrier_record, "Controller barrier")
    root = posixpath.dirname(results_record["path"])
    require(results_record["path"] == root + "/results.json" and
            barrier_record["path"] == root + "/locks-ready.json", "Controller/barrier output layout differs")
    phase_record = _record(results.get("phase_order"), "Phase-order JSONL record")
    require(phase_record["path"] == root + "/phase-order.jsonl", "Phase-order file lineage differs")
    _integer(results.get("retries"), "Retries", 0)
    _integer(results.get("tile_forward_count"), "Controller forwards", 450)
    _loads(results.get("model_load_operations"), TOTAL_LOADS, "Controller loads")
    require(results.get("all_fixed_states_and_contexts_retained") is True, "Complete fixed output receipt is missing")
    controller_wall = _number(results.get("wall_seconds"), "Controller wall", 600)
    model_results = _mapping(results.get("models"), "Controller model results")
    require(set(model_results) == set(workers) == set(PHASES), "Exactly two worker phases required")
    require(barrier.get("roles") == list(ROLES) and barrier.get("all_three_locks_written_before_evaluation") is True,
            "All-three cutoff barrier receipt differs")
    _integer(barrier.get("calibration_tile_forwards"), "Barrier calibration forwards", 150)
    _integer(barrier.get("evaluation_tile_forwards"), "Barrier evaluation forwards", 0)
    barrier_models = _mapping(barrier.get("models"), "Barrier calibration models")
    locks = _mapping(barrier.get("threshold_locks"), "Barrier threshold locks")
    require(set(barrier_models) == set(locks) == set(ROLES), "Barrier must bind all three completed models/locks")
    paired = results.get("workers")
    require(type(paired) is list and len(paired) == 6 and type(events) is list and len(events) == 12,
            "Exactly six workers and twelve parsed JSONL events required")
    require(all(type(pair) is dict and set(pair) == {"launch", "complete"} for pair in paired),
            "Worker event pair schema differs")
    require(events == [event for pair in paired for event in (pair["launch"], pair["complete"])],
            "Parsed JSONL events differ from the controller's worker pairs")
    require(barrier.get("calibration_workers") == paired[:3], "Barrier calibration timeline differs")
    aggregate = dict.fromkeys(TOTAL_LOADS, 0)
    phase_forwards = dict.fromkeys(PHASES, 0)
    timeline = []
    reference_initializer = None
    python_command = None
    for index, (phase, role) in enumerate((p, r) for p in PHASES for r in ROLES):
        _mapping(workers[phase], "Phase workers")
        _mapping(model_results[phase], "Phase result records")
        require(set(workers[phase]) == set(model_results[phase]) == set(ROLES), "Exactly three workers per phase required")
        receipt = _mapping(workers[phase][role], "Worker receipt")
        require(receipt.get("status") == "PASS" and receipt.get("phase") == phase and receipt.get("role") == role,
                "Worker status/phase/role differs")
        _same_record(receipt.get("plan"), plan_record, "Worker plan")
        require(receipt.get("source_model") == models[role] and
                receipt.get("state_sha256") == models[role]["state_sha256"], "Worker source-model/state receipt differs")
        for field in ("model_state_unchanged", "parameter_gradients_absent", "source_pins_unchanged_before_after"):
            require(receipt.get(field) is True, "Worker " + field + " is not true")
        for field in ("training_calls", "network_calls"):
            _integer(receipt.get(field), "Worker " + field, 0)
        forwards = 50 if phase == "calibration" else 100
        _integer(receipt.get("tile_forward_count"), "Worker forwards", forwards)
        phase_forwards[phase] += forwards
        runs = _mapping(receipt.get("runs"), "Worker runs")
        require(set(runs) == set(CASE_IDS[phase]), "Worker fixed case participation differs")
        wall = _number(receipt.get("wall_seconds"), "Worker wall", 120)
        _integer(receipt.get("peak_rss_bytes"), "Worker peak RSS", minimum=1)
        require(receipt["peak_rss_bytes"] <= 3 * 1024**3, "Worker RSS exceeds acceptance")
        initializer = _initializer(receipt.get("initializer_contract"), plan)
        if reference_initializer is None:
            reference_initializer = initializer
        require(initializer == reference_initializer, "Six workers must share the identical initializer contract")
        if role == "frozen_seed42":
            require(receipt.get("saved_state_storage_contract") is None, "Baseline has an unauthorized candidate load")
        else:
            _storage(receipt.get("saved_state_storage_contract"), "Candidate storage", initializer["tensor_schema"])
            require(receipt["saved_state_storage_contract"] == initializer["storage_contract"],
                    "Candidate fixed factory storage schema differs")
        replacement = int(role != "frozen_seed42")
        expected_loads = {"final_role_worker_constructions": 1, "initializer_strict_loads": 1,
            "candidate_raw_state_strict_loads": replacement, "total_load_state_dict_calls": 1 + replacement}
        _loads(receipt.get("model_load_operations"), expected_loads, "Worker loads")
        for field, count in expected_loads.items():
            aggregate[field] += count
        launch, complete = (_mapping(paired[index][key], "Worker " + key) for key in ("launch", "complete"))
        for event, name in ((launch, "launch"), (complete, "complete")):
            require(event.get("event") == name and event.get("phase") == phase and event.get("role") == role,
                    "Exact baseline/A/B calibration-then-evaluation event order differs")
            _integer(event.get("index"), "Worker event index", index)
            elapsed = _number(event.get("elapsed_seconds"), "Worker event elapsed", controller_wall)
            timestamp = _utc(event.get("utc"), "Worker event UTC")
            timeline.append((elapsed, timestamp))
        _integer(complete.get("exit_code"), "Worker exit", 0)
        expected_result = _record(model_results[phase][role], "Worker result record")
        require(expected_result["path"] == root + "/" + phase + "/" + role + "/result.json",
                "Worker result path/role/phase differs")
        _same_record(complete.get("result"), expected_result, "Completed worker result")
        require(wall <= complete["elapsed_seconds"] - launch["elapsed_seconds"] + 1e-6,
                "Worker reported wall exceeds its controller interval")
        command = launch.get("command")
        require(type(command) is list and all(type(x) is str and x for x in command) and
                len(command) == (10 if phase == "calibration" else 14), "Worker launch command schema differs")
        require(command[0].startswith("/"), "Worker Python executable must be an absolute path")
        if python_command is None:
            python_command = command[0]
        require(command[0] == python_command and command[1] == plan["code"]["run_comparison.py"]["path"] and
                command[2:10] == ["--plan", plan_record["path"], "--worker-role", role, "--phase", phase,
                                 "--output-dir", posixpath.dirname(expected_result["path"])], "Worker launch command lineage differs")
        if phase == "calibration":
            require(launch.get("calibration_barrier") is None and receipt.get("calibration_barrier") is None,
                    "Calibration worker must precede the all-three barrier")
            _same_record(barrier_models[role], expected_result, "Barrier completed calibration model")
            _same_record(receipt.get("threshold_lock"), locks[role], "Worker cutoff lock")
            require(_record(locks[role], "Cutoff lock")["path"] == posixpath.dirname(expected_result["path"]) + "/thresholds.json",
                    "Calibration lock path differs")
        else:
            require(receipt.get("threshold_lock") is None, "Evaluation worker may not create a fitted cutoff lock")
            _same_record(receipt.get("calibration_barrier"), barrier_record, "Evaluation worker barrier")
            _same_record(launch.get("calibration_barrier"), barrier_record, "Evaluation launch barrier")
            require(command[10:] == ["--locks-ready", barrier_record["path"], "--locks-ready-sha256", barrier_record["sha256"]],
                    "Evaluation launch did not bind the locked barrier")
    require(phase_forwards == {"calibration": 150, "evaluation": 300} and aggregate == TOTAL_LOADS,
            "Complete phase-forward or load accounting differs")
    require(all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(timeline, timeline[1:])),
            "Worker elapsed or UTC timeline is not monotonic")
    locked_elapsed = _number(barrier.get("locked_at_elapsed_seconds"), "Barrier lock elapsed", controller_wall)
    locked_utc = _utc(barrier.get("locked_at_utc"), "Barrier lock UTC")
    require(timeline[5][0] <= locked_elapsed <= timeline[6][0] and
            timeline[5][1] <= locked_utc <= timeline[6][1], "All-three locks did not precede evaluation in both clocks")
    return {"status": "PASS", "workers_verified": 6, "jsonl_events_verified": 12,
            "worker_order": [[p, r] for p in PHASES for r in ROLES], "phase_forwards": phase_forwards,
            "tile_forwards": 450, "model_load_operations": aggregate,
            "initializer_contracts_identical": True, "cutoff_barrier_before_evaluation_elapsed_and_UTC": True,
            "controller_wall_seconds": controller_wall, "model_calls": 0, "checkpoint_reads": 0,
            "array_reads": 0, "network_calls": 0,
            "limitation": "Model identity, unchanged state, gradients and operation counts are validated retained worker/source receipts, not authenticated weight replay. CPU/address-space caps are frozen plan/source fields; measured CPU usage is not present in these worker receipts."}


def synthetic_fixture_inputs():
    """Construct plausible JSON objects only, with no file paths opened."""
    def pin(path, seed="a", size=1):
        return {"path": path, "size_bytes": size, "sha256": seed * 64}
    plan_record = pin("/synthetic/comparison-plan.json")
    results_record = pin("/synthetic/run/results.json", "b")
    barrier_record = pin("/synthetic/run/locks-ready.json", "c")
    schema, groups = [], []
    for i in range(264):
        numel = (33628685 if i == 0 else 1) if i < 244 else (917551 if i == 244 else 1)
        keys = ["g" + str(i).zfill(3) + ".a"] + (["g" + str(i).zfill(3) + ".b"] if i < 244 else [])
        for key in keys:
            schema.append({"key": key, "shape": [numel], "dtype": "torch.float32", "numel": numel, "bytes": 4 * numel})
        if i < 244:
            groups.append({"keys": keys, "storage_bytes": 4 * numel, "exact_same_view": True, "bitwise_equal": True})
    storage = {**STORAGE, "alias_groups": groups, "alias_partition_matches_factory": True,
               "bitwise_alias_values_consistent": True}
    config = {"mode": "flat", "model_type": "vesuvius_unet_3d_stem_2d", "patch_size": [17, 128, 128], "in_channels": 1,
              "model_config": {"autoconfigure": True, "basic_encoder_block": "BasicBlockD", "basic_decoder_block": "ConvBlock", "stem_channels": 16, "z_projection_mode": "none"},
              "targets": {"ink": {"activation": "none", "out_channels": 1, "z_projection_mode": "none"}},
              "image_normalization": {"mode": "robust_mad", "percentile_lower": 1., "percentile_upper": 99.},
              "loss": {"bce_label_smoothing": .5, "dice_label_smoothing": 0}}
    initial = {**pin("/synthetic/initializer.pth", "d", 138360039), "label": "seed42", "repository": "scrollprize/ink_9um",
               "revision": "synthetic-retained-revision", "repository_path": "hybrid_3d2d-seed42/step-075000.pth",
               "embedded_step": 74999, "filename_step_label": 75000}
    plan = {"kind": "Paris-fragments-fixed-state-calibration-transfer-v1", "ready_for_execution": True, "roles": list(ROLES),
            "limits": dict(LIMITS), "planned_model_load_operations": dict(TOTAL_LOADS), "initializer": initial,
            "cases": [{"id": c} for phase in PHASES for c in CASE_IDS[phase]],
            "code": {"run_comparison.py": pin("/synthetic/run_comparison.py")},
            "transitive_runtime_inputs": [pin("/synthetic/run_ink_reference.py", "e"), pin("/synthetic/source-manifest.json", "f")],
            "models": {role: {"role": role, "checkpoint": None if role == ROLES[0] else pin("/synthetic/" + role + ".pth"),
                              "state_sha256": str(i + 1) * 64} for i, role in enumerate(ROLES)}}
    initializer = {**initial, "model_repository": initial["repository"], "model_revision": initial["revision"],
                   "deserialization": "torch.load(map_location='cpu', weights_only=True)", "strict_state_match": True,
                   "missing_keys": [], "unexpected_keys": [], "weight_container": "model", "step": 74999,
                   "parameter_count": 34546498, "selected_state_bytes": STORAGE["named_tensor_bytes"],
                   "selected_unique_storage_bytes": STORAGE["unique_storage_bytes"], "storage_contract": storage,
                   "tensor_schema": schema, "tensor_schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                   "embedded_config": config, "model_type": "vesuvius_unet_3d_stem_2d",
                   "reference_adapter_sha256": "e" * 64, "source_manifest_sha256": "f" * 64}
    results = {"status": "PASS", "plan": plan_record, "locks_ready": barrier_record, "models": {p: {} for p in PHASES},
               "workers": [], "wall_seconds": 100., "tile_forward_count": 450, "model_load_operations": dict(TOTAL_LOADS),
               "retries": 0, "phase_order": pin("/synthetic/run/phase-order.jsonl"), "all_fixed_states_and_contexts_retained": True}
    workers = {p: {} for p in PHASES}
    locks = {}
    start = datetime(2026, 10, 8, tzinfo=timezone.utc)
    for i, (phase, role) in enumerate((p, r) for p in PHASES for r in ROLES):
        target = "/synthetic/run/" + phase + "/" + role
        result_pin = pin(target + "/result.json")
        lock = pin(target + "/thresholds.json") if phase == "calibration" else None
        if lock is not None:
            locks[role] = lock
        replacement = int(role != ROLES[0])
        counts = {"final_role_worker_constructions": 1, "initializer_strict_loads": 1,
                  "candidate_raw_state_strict_loads": replacement, "total_load_state_dict_calls": 1 + replacement}
        workers[phase][role] = {"status": "PASS", "role": role, "phase": phase, "plan": plan_record,
            "source_model": plan["models"][role], "state_sha256": plan["models"][role]["state_sha256"],
            "initializer_contract": copy.deepcopy(initializer), "saved_state_storage_contract": None if role == ROLES[0] else copy.deepcopy(storage),
            "runs": {c: {} for c in CASE_IDS[phase]}, "threshold_lock": lock,
            "calibration_barrier": None if phase == "calibration" else barrier_record,
            "tile_forward_count": 50 if phase == "calibration" else 100, "wall_seconds": 5., "peak_rss_bytes": 1024**3,
            "model_state_unchanged": True, "model_load_operations": counts, "parameter_gradients_absent": True,
            "training_calls": 0, "network_calls": 0, "source_pins_unchanged_before_after": True}
        launch_time, complete_time = 1. + 15. * i, 11. + 15. * i
        command = ["/synthetic/python", "/synthetic/run_comparison.py", "--plan", plan_record["path"], "--worker-role", role,
                   "--phase", phase, "--output-dir", target]
        if phase == "evaluation":
            command += ["--locks-ready", barrier_record["path"], "--locks-ready-sha256", barrier_record["sha256"]]
        launch = {"event": "launch", "role": role, "phase": phase, "index": i,
                  "utc": (start + timedelta(seconds=launch_time)).isoformat(), "elapsed_seconds": launch_time,
                  "calibration_barrier": None if phase == "calibration" else barrier_record, "command": command}
        complete = {"event": "complete", "role": role, "phase": phase, "index": i, "exit_code": 0, "result": result_pin,
                    "utc": (start + timedelta(seconds=complete_time)).isoformat(), "elapsed_seconds": complete_time}
        results["workers"].append({"launch": launch, "complete": complete})
        results["models"][phase][role] = result_pin
    barrier = {"status": "PASS", "plan": plan_record, "roles": list(ROLES), "models": copy.deepcopy(results["models"]["calibration"]),
               "threshold_locks": locks, "calibration_tile_forwards": 150, "evaluation_tile_forwards": 0,
               "all_three_locks_written_before_evaluation": True, "locked_at_elapsed_seconds": 43.,
               "locked_at_utc": (start + timedelta(seconds=43.)).isoformat(), "calibration_workers": copy.deepcopy(results["workers"][:3])}
    events = copy.deepcopy([event for pair in results["workers"] for event in (pair["launch"], pair["complete"])])
    return [plan, plan_record, results, results_record, barrier, barrier_record, workers, events]


def synthetic_fixture_checks():
    """Return meaningful in-memory mutation checks; never write a receipt."""
    checks = []
    original = synthetic_fixture_inputs()
    verify_execution_receipts(*original)
    checks.append({"case": "six_workers_twelve_events_full_success", "status": "PASS"})

    def reject(name, mutation):
        values = copy.deepcopy(original)
        mutation(values)
        try:
            verify_execution_receipts(*values)
        except (ValueError, KeyError, TypeError) as error:
            checks.append({"case": name, "status": "PASS", "rejected_as_expected": type(error).__name__ + ": " + str(error)})
        else:
            raise AssertionError("Expected rejection did not occur: " + name)

    def set_path(values, path, value):
        parent = values
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value

    mutations = [
        ("missing_barrier_rejected", [4], None),
        ("missing_worker_rejected", [6, "evaluation", "B_preserve"], None),
        ("draft_plan_rejected", [0, "ready_for_execution"], False),
        ("load_accounting_six_physical_calls_rejected", [2, "model_load_operations", "total_load_state_dict_calls"], 6),
        ("planned_load_accounting_missing_replacement_rejected", [0, "planned_model_load_operations", "candidate_raw_state_strict_loads"], 0),
        ("worker_boolean_load_count_rejected", [6, "calibration", "frozen_seed42", "model_load_operations", "candidate_raw_state_strict_loads"], False),
        ("worker_float_forward_count_rejected", [6, "evaluation", "A_preserve", "tile_forward_count"], 100.),
        ("controller_wrong_forward_total_rejected", [2, "tile_forward_count"], 449),
        ("barrier_premature_evaluation_count_rejected", [4, "evaluation_tile_forwards"], 1),
        ("barrier_boolean_calibration_count_rejected", [4, "calibration_tile_forwards"], True),
        ("worker_state_identity_changed_rejected", [6, "evaluation", "A_preserve", "state_sha256"], "f" * 64),
        ("worker_source_model_missing_rejected", [6, "evaluation", "B_preserve", "source_model"], None),
        ("worker_state_changed_rejected", [6, "evaluation", "frozen_seed42", "model_state_unchanged"], False),
        ("worker_gradient_absence_not_true_rejected", [6, "calibration", "A_preserve", "parameter_gradients_absent"], 1),
        ("worker_training_call_rejected", [6, "calibration", "B_preserve", "training_calls"], 1),
        ("worker_network_call_rejected", [6, "evaluation", "A_preserve", "network_calls"], 1),
        ("worker_source_pin_receipt_missing_rejected", [6, "evaluation", "B_preserve", "source_pins_unchanged_before_after"], None),
        ("initializer_unsafe_deserialization_rejected", [6, "calibration", "A_preserve", "initializer_contract", "deserialization"], "torch.load(weights_only=False)"),
        ("initializer_nonstrict_load_rejected", [6, "calibration", "B_preserve", "initializer_contract", "strict_state_match"], False),
        ("initializer_none_loss_config_rejected", [6, "calibration", "B_preserve", "initializer_contract", "embedded_config", "loss"], None),
        ("initializer_none_ema_config_rejected", [6, "calibration", "B_preserve", "initializer_contract", "embedded_config", "ema"], None),
        ("initializer_schema_digest_changed_rejected", [6, "evaluation", "A_preserve", "initializer_contract", "tensor_schema_sha256"], "0" * 64),
        ("initializer_schema_bool_numel_rejected", [6, "calibration", "A_preserve", "initializer_contract", "tensor_schema", 0, "numel"], True),
        ("initializer_schema_dtype_changed_rejected", [6, "calibration", "B_preserve", "initializer_contract", "tensor_schema", 0, "dtype"], "torch.float64"),
        ("candidate_storage_missing_rejected", [6, "evaluation", "A_preserve", "saved_state_storage_contract"], None),
        ("candidate_storage_bool_bytes_rejected", [6, "calibration", "B_preserve", "saved_state_storage_contract", "named_tensor_bytes"], True),
        ("candidate_alias_values_not_true_rejected", [6, "evaluation", "B_preserve", "saved_state_storage_contract", "alias_groups", 0, "bitwise_equal"], False),
        ("candidate_alias_partition_changed_rejected", [6, "calibration", "A_preserve", "saved_state_storage_contract", "alias_partition_matches_factory"], False),
        ("worker_wall_none_rejected", [6, "calibration", "frozen_seed42", "wall_seconds"], None),
        ("worker_wall_over_limit_rejected", [6, "evaluation", "B_preserve", "wall_seconds"], 120.001),
        ("worker_rss_float_rejected", [6, "calibration", "A_preserve", "peak_rss_bytes"], 1.),
        ("worker_rss_over_limit_rejected", [6, "evaluation", "B_preserve", "peak_rss_bytes"], 3 * 1024**3 + 1),
        ("controller_wall_nan_rejected", [2, "wall_seconds"], float("nan")),
        ("controller_wall_over_limit_rejected", [2, "wall_seconds"], 600.001),
        ("controller_wall_before_completion_rejected", [2, "wall_seconds"], 85.),
        ("barrier_elapsed_before_all_calibration_rejected", [4, "locked_at_elapsed_seconds"], 40.),
        ("barrier_elapsed_after_evaluation_launch_rejected", [4, "locked_at_elapsed_seconds"], 47.),
        ("barrier_utc_before_all_calibration_rejected", [4, "locked_at_utc"], "2026-10-08T00:00:40+00:00"),
        ("barrier_utc_after_evaluation_launch_rejected", [4, "locked_at_utc"], "2026-10-08T00:00:47+00:00"),
        ("barrier_naive_utc_rejected", [4, "locked_at_utc"], "2026-10-08T00:00:43"),
        ("barrier_bad_binding_rejected", [4, "models", "A_preserve", "sha256"], "0" * 64),
        ("evaluation_worker_barrier_missing_rejected", [6, "evaluation", "frozen_seed42", "calibration_barrier"], None),
        ("evaluation_new_threshold_fit_rejected", [6, "evaluation", "B_preserve", "threshold_lock"], {"unexpected": True}),
        ("retry_receipt_rejected", [2, "retries"], 1),
        ("worker_calibration_case_substitution_rejected", [6, "calibration", "A_preserve", "runs"], {"frag1-context1": {}}),
        ("events_jsonl_mismatch_rejected", [7, 0, "index"], 1),
        ("none_event_elapsed_rejected", [2, "workers", 3, "launch", "elapsed_seconds"], None),
        ("event_boolean_index_rejected", [2, "workers", 0, "launch", "index"], False),
        ("changed_evaluation_command_barrier_rejected", [2, "workers", 3, "launch", "command", 13], "0" * 64),
        ("changed_model_initializer_artifact_rejected", [6, "evaluation", "B_preserve", "initializer_contract", "sha256"], "0" * 64),
    ]
    for name, path, value in mutations:
        reject(name, lambda values, path=path, value=value: set_path(values, path, value))
    def alternate_container(values):
        for phase in PHASES:
            for role in ROLES:
                values[6][phase][role]["initializer_contract"]["weight_container"] = "state_dict"
    reject("all_six_synchronized_alternate_weight_container_rejected", alternate_container)
    inclusive = copy.deepcopy(original)
    inclusive[2]["wall_seconds"] = 600.
    inclusive[6]["evaluation"]["B_preserve"]["peak_rss_bytes"] = 3 * 1024**3
    verify_execution_receipts(*inclusive)
    checks.append({"case": "controller_600_seconds_and_RSS_3GiB_inclusive_limits", "status": "PASS"})
    # Synchronize redundant logs to make these mutations reach timeline/order checks.
    def synchronized_mutation(values, path, value):
        set_path(values, path, value)
        values[7] = copy.deepcopy([event for pair in values[2]["workers"] for event in (pair["launch"], pair["complete"])])
        values[4]["calibration_workers"] = copy.deepcopy(values[2]["workers"][:3])
    for name, path, value in [
        ("synchronized_wrong_worker_role_order_rejected", [2, "workers", 1, "launch", "role"], "B_preserve"),
        ("synchronized_nonmonotone_elapsed_rejected", [2, "workers", 4, "launch", "elapsed_seconds"], 50.),
        ("synchronized_nonmonotone_utc_rejected", [2, "workers", 4, "launch", "utc"], "2026-10-08T00:00:50+00:00"),
        ("synchronized_boolean_worker_index_rejected", [2, "workers", 0, "launch", "index"], False),
        ("synchronized_wrong_evaluation_barrier_command_rejected", [2, "workers", 3, "launch", "command", 13], "0" * 64),
        ("synchronized_worker_exit_failure_rejected", [2, "workers", 5, "complete", "exit_code"], 1),
        ("synchronized_worker_wall_longer_than_interval_rejected", [2, "workers", 5, "complete", "elapsed_seconds"], 80.),
    ]:
        reject(name, lambda values, path=path, value=value: synchronized_mutation(values, path, value))
    return {"status": "PASS", "cases": len(checks), "checks": checks,
            "fixture_success_summary": verify_execution_receipts(*original), "real_input_reads": 0,
            "array_io": 0, "model_calls": 0, "Torch_imports": 0, "checkpoint_reads": 0,
            "network_calls": 0, "module_file_io": 0}

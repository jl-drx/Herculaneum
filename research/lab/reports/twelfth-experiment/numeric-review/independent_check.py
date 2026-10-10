"""Independent NumPy loss/oracle reconstruction; no audit helper/model imports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "1"

def offline(*args, **kwargs):
    raise RuntimeError("No network is permitted in the independent numeric review")

socket.create_connection = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
import numpy as np

ARMS = ["A_preserve", "B_preserve"]
WINDOWS = {"1-8": list(range(1, 9)), "9-16": list(range(9, 17)),
           "17-24": list(range(17, 25)), "25-32": list(range(25, 33)),
           "all-1-32": list(range(1, 33))}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def record(path):
    path = Path(path).resolve(strict=True)
    require(path.suffix not in {".pth", ".pt", ".ckpt", ".safetensors"}, "Checkpoint access is forbidden")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}

def verify(item):
    require(record(item["path"]) == item, "Source pin changed: " + item["path"])
    return Path(item["path"])

def read_json(item):
    return json.loads(verify(item).read_text())

def avg(values):
    return float(np.sum(values, dtype=np.float64) / values.size)

def terms(student, teacher, hard):
    """Use log-probability cross entropy/KL and entropy oracle, not Bregman code."""
    z, t, y = [np.asarray(a, dtype=np.float64) for a in [student, teacher, hard]]
    require(z.shape == t.shape == y.shape and z.size > 0, "Nonempty aligned support required")
    require(np.isfinite(z).all() and np.isfinite(t).all() and np.isin(y, [0, 1]).all(), "Invalid logit/target")
    log_p, log_pc = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    log_q, log_qc = -np.logaddexp(0, -t), -np.logaddexp(0, t)
    q, qc = np.exp(log_q), np.exp(log_qc)
    soft = 0.25 + 0.5 * y
    bce_z, bce_t = -soft * log_p - (1 - soft) * log_pc, -soft * log_q - (1 - soft) * log_qc
    kl = q * (log_q - log_p) + qc * (log_qc - log_pc)
    linear = (q - soft) * (z - t)
    optimum = (soft + q) / 2
    entropy_r = -optimum * np.log(optimum) - (1 - optimum) * np.log1p(-optimum)
    entropy_q = -q * log_q - qc * log_qc
    optimum_full = 2 * entropy_r - entropy_q
    base_delta = avg(bce_z) - avg(bce_t)
    full_z = avg(bce_z) + avg(kl)
    full_delta = full_z - avg(bce_t)
    return {"count": int(y.size), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "student_smoothed_BCE": avg(bce_z), "initializer_smoothed_BCE": avg(bce_t),
            "student_hard_BCE": avg(-y * log_p - (1 - y) * log_pc),
            "initializer_hard_BCE": avg(-y * log_q - (1 - y) * log_qc),
            "KL": avg(kl), "linear_drift": avg(linear), "base_delta": base_delta,
            "full_delta": full_delta, "student_full": full_z, "initializer_full": avg(bce_t),
            "base_decomposition_residual": base_delta - avg(kl) - avg(linear),
            "full_decomposition_residual": full_delta - 2 * avg(kl) - avg(linear),
            "free_logit_optimum_full": avg(optimum_full),
            "student_gap_to_free_optimum": full_z - avg(optimum_full),
            "initializer_gap_to_free_optimum": avg(bce_t) - avg(optimum_full),
            "optimum_probability_min": float(optimum.min()), "optimum_probability_max": float(optimum.max())}

def d4(array, code):
    """Destination-to-source coordinate map; no rot90/flip calls."""
    require(type(code) is int and 0 <= code < 8 and array.shape[-2] == array.shape[-1], "Square D4 input/code required")
    n = array.shape[-1]
    yy, xx = np.indices((n, n))
    u = xx if code < 4 else n - 1 - xx
    mappings = [(yy, u), (u, n - 1 - yy), (n - 1 - yy, n - 1 - u), (n - 1 - u, yy)]
    source_y, source_x = mappings[code % 4]
    return np.ascontiguousarray(array[..., source_y, source_x])

def decode_region(case):
    raw = {key: np.load(verify(item), mmap_mode="r", allow_pickle=False) for key, item in case["arrays"].items()}
    require(all(a.shape == (384, 384) for a in raw.values()), "Regional canvas differs")
    require(all(raw[k].dtype == np.uint8 for k in ["inklabels", "supervision_mask", "validation_mask"]), "Annotation dtype differs")
    require(all(np.isin(raw[k], [0, 1]).all() for k in ["supervision_mask", "validation_mask"]), "Mask encoding differs")
    require(raw["observed"].dtype == raw["teacher43_evaluated"].dtype == np.bool_, "Acquisition/evaluated mask dtype differs")
    require(raw["teacher43_scores"].dtype == np.float32, "Pseudo-source dtype differs")
    observed = raw["observed"]
    manual = np.logical_and.reduce([observed, raw["supervision_mask"] == 1, raw["validation_mask"] == 0])
    scores = raw["teacher43_scores"]
    finite = np.isfinite(scores)
    require(np.all((scores[finite] >= 0) & (scores[finite] <= 1)), "Pseudo scores outside probability range")
    pseudo = np.logical_and.reduce([observed, raw["teacher43_evaluated"], finite, np.logical_not(manual)])
    require(not (manual & pseudo).any(), "Supports overlap")
    require(np.isin(raw["inklabels"][manual], [0, 1]).all(), "Manual hard labels differ")
    return {"manual_mask": manual, "pseudo_mask": pseudo,
            "manual_target": raw["inklabels"], "pseudo_target": scores >= 0.5}

def batch(regions, step):
    require([p["region"] for p in step["patches"]] == [0, 1], "Batch region order differs")
    selected = {key: [] for key in ["manual_mask", "pseudo_mask", "manual_target", "pseudo_target"]}
    exposures = []
    for patch in step["patches"]:
        r = patch["region"]
        x, y, w, h = patch["bbox"]
        require(w == h == 128 and 0 <= x <= 256 and 0 <= y <= 256 and x % 64 == y % 64 == 0, "Tile placement differs")
        entry = {"region": r}
        for support in ["manual", "pseudo"]:
            mask = d4(regions[r][support + "_mask"][y:y + h, x:x + w], patch["augmentation"])
            target = d4(regions[r][support + "_target"][y:y + h, x:x + w], patch["augmentation"])
            selected[support + "_mask"].append(mask)
            selected[support + "_target"].append(target)
            entry[support] = {"positive": int((mask & (target == 1)).sum()), "negative": int((mask & (target == 0)).sum())}
        require(min(entry["manual"].values()) >= 256, "Manual class gate differs")
        exposures.append(entry)
    return {key: np.stack(value) for key, value in selected.items()}, exposures

def values_for_step(manual, pseudo, coefficient, historical):
    def weighted(key):
        return manual[key] + coefficient * pseudo[key]
    values = {"U_student": manual["student_smoothed_BCE"], "U_initializer": manual["initializer_smoothed_BCE"],
              "P_student": pseudo["student_smoothed_BCE"], "P_initializer": pseudo["initializer_smoothed_BCE"]}
    for label, key in [("hard_base", "hard_BCE"), ("base", "smoothed_BCE"), ("full", "full")]:
        values[label + "_student"] = weighted("student_" + key)
        values[label + "_initializer"] = weighted("initializer_" + key)
    for key in ["base_delta", "full_delta", "student_gap_to_free_optimum", "initializer_gap_to_free_optimum",
                "base_decomposition_residual", "full_decomposition_residual"]:
        values[key] = weighted(key)
    values.update(weighted_KL=weighted("KL"), weighted_linear_drift=weighted("linear_drift"),
                  free_optimum_full=weighted("free_logit_optimum_full"), ninth_base_receipt=historical["total_loss"],
                  ninth_U_receipt=historical["loss_U"], ninth_P_receipt=historical["loss_P"])
    values["ninth_base_delta_vs_same_input_initializer_receipt"] = historical["total_loss"] - values["base_initializer"]
    values["base_delta_vs_ninth_receipt"] = values["base_student"] - historical["total_loss"]
    return values

def fixture_checks():
    array = np.arange(1, 10).reshape(3, 3)
    gold = [[[1,2,3],[4,5,6],[7,8,9]], [[3,6,9],[2,5,8],[1,4,7]],
            [[9,8,7],[6,5,4],[3,2,1]], [[7,4,1],[8,5,2],[9,6,3]],
            [[3,2,1],[6,5,4],[9,8,7]], [[9,6,3],[8,5,2],[7,4,1]],
            [[7,8,9],[4,5,6],[1,2,3]], [[1,4,7],[2,5,8],[3,6,9]]]
    for code in range(8):
        require(np.array_equal(d4(array, code), gold[code]), "Golden D4 fixture failed")
    z, t, y = np.array([-7., -1., 0., 3., 6.]), np.array([-2., 2., 0., -1., 4.]), np.array([0., 1., 0., 1., 0.])
    result = terms(z, t, y)
    require(abs(result["base_decomposition_residual"]) < 1e-12 and abs(result["full_decomposition_residual"]) < 1e-12, "Independent decomposition fixture failed")
    q = 1 / (1 + np.exp(-t)); optimum = (0.25 + 0.5 * y + q) / 2
    optimum_logits = np.log(optimum / (1 - optimum))
    at_optimum = terms(optimum_logits, t, y)
    require(abs(at_optimum["student_gap_to_free_optimum"]) < 1e-12, "Entropy oracle equality fixture failed")
    for offset in [-10., -.5, .5, 10.]:
        require(terms(optimum_logits + offset, t, y)["student_gap_to_free_optimum"] > 0, "Oracle perturbation fixture failed")
    equality = terms(t, t, y)
    require(equality["KL"] == equality["base_delta"] == equality["linear_drift"] == 0, "Initializer identity fixture failed")
    extreme = terms(np.array([-1000., 1000.]), np.array([1000., -1000.]), np.array([0., 1.]))
    require(all(math.isfinite(v) for v in extreme.values()), "Extreme-logit finiteness fixture failed")
    return {"status": "PASS", "golden_D4_cases": 8, "decomposition_cases": 1,
            "oracle_equality_cases": 1, "oracle_perturbation_cases": 4,
            "initializer_identity_cases": 1, "extreme_logit_cases": 1,
            "implementation": "log-probability BCE/KL; entropy oracle; coordinate-index D4"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--audit-sha256")
    parser.add_argument("--authorized-retained-read", action="store_true")
    args = parser.parse_args()
    require("torch" not in sys.modules, "Torch must not be imported")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fixtures = fixture_checks()
    if args.self_test:
        fixtures["code"] = record(Path(__file__))
        (args.output_dir / "synthetic-fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
        print(json.dumps(fixtures))
        return
    require(args.authorized_retained_read, "Root authorization for actual retained arrays is required")
    plan_pin, audit_pin = record(args.plan), record(args.audit)
    require(plan_pin["sha256"] == args.plan_sha256 and audit_pin["sha256"] == args.audit_sha256, "Root execution pins differ")
    plan, audit = read_json(plan_pin), read_json(audit_pin)
    require(plan["ready_for_execution"] is True and audit["status"] == "PASS" and audit["plan"] == plan_pin, "Completed frozen audit required")
    require(plan["arms"] == ARMS and plan["windows"] == WINDOWS, "Scope changed")
    require(plan["tolerances"] == {"float32_scalar_absolute": 2e-6, "float64_identity_absolute": 1e-12,
            "KL_mean_floor": -1e-6, "first_logit_absolute": 1e-7}, "Frozen tolerances changed")
    code_pin = record(Path(__file__))
    before = {item["path"]: item for item in plan["verified_inputs"]}
    for item in before.values():
        verify(item)
    regions = [decode_region(case) for case in plan["regions"]]
    require(len(regions) == 2, "Two regions required")
    schedule = read_json(plan["schedule"])
    require([s["step"] for s in schedule] == list(range(1, 33)), "Schedule changed")
    maximum = {}
    comparison_count = 0
    def close(actual, expected, name, tolerance=1e-12):
        nonlocal comparison_count
        require(math.isfinite(actual) and math.isfinite(expected), "Nonfinite scalar: " + name)
        error = abs(float(actual) - float(expected))
        require(error <= tolerance, "Independent scalar mismatch: " + name + f" error={error}")
        maximum[name] = max(maximum.get(name, 0.), error)
        comparison_count += 1
    def check_terms(actual, expected):
        require(set(actual) == set(expected), "Support fields changed")
        for key, expected_value in expected.items():
            if key in {"count", "positive", "negative"}:
                require(actual[key] == expected_value, "Support count mismatch")
            else:
                close(actual[key], expected_value, "terms/" + key)
        require(expected["KL"] >= -1e-6 and expected["student_gap_to_free_optimum"] >= -1e-12
                and expected["initializer_gap_to_free_optimum"] >= -1e-12, "KL/oracle nonnegativity failed")
    all_steps = {}
    teacher_reference = None
    maximum_logged = {}
    for arm in ARMS:
        coefficient = 0. if arm == "A_preserve" else .25
        inputs = plan["arm_inputs"][arm]
        z, t = [np.load(verify(inputs[key]), mmap_mode="r", allow_pickle=False) for key in ["student_logits", "teacher_logits"]]
        require(all(a.dtype == np.float32 and a.shape == (32, 2, 1, 128, 128) and np.isfinite(a).all() for a in [z, t]), "Retained logits contract changed")
        require(float(np.max(np.abs(z[0].astype(np.float64) - t[0]))) <= 1e-7, "First student/teacher discrepancy")
        if teacher_reference is None:
            teacher_reference = t
        else:
            require(np.array_equal(t, teacher_reference), "Teacher logits differ across arms")
        logs = [json.loads(line) for line in verify(inputs["steps"]).read_text().splitlines()]
        historical = [json.loads(line) for line in verify(inputs["ninth_steps"]).read_text().splitlines()]
        reported = audit["arms"][arm]
        require(reported["pseudo_coefficient"] == coefficient, "Pseudo coefficient differs")
        require([r["step"] for r in logs] == [r["step"] for r in historical] == [r["step"] for r in reported["steps"]] == list(range(1, 33)), "All32 steps required")
        reconstructed = []
        for index, (scheduled, log, old, output) in enumerate(zip(schedule, logs, historical, reported["steps"])):
            require(scheduled["patches"] == log["patches"] == old["patches"] == output["patches"], "Patch/D4 recipe changed")
            selected, exposure = batch(regions, scheduled)
            require(exposure == log["exposures"] == old["exposures"] == output["exposures"], "Exact support exposure mismatch")
            zz, tt = z[index, :, 0].astype(np.float64), t[index, :, 0].astype(np.float64)
            supports = {}
            for name in ["manual", "pseudo"]:
                mask, target = selected[name + "_mask"], selected[name + "_target"]
                supports[name] = terms(zz[mask], tt[mask], target[mask])
                check_terms(output[name], supports[name])
            manual, pseudo = supports["manual"], supports["pseudo"]
            balanced_groups = []
            expected_strata = []
            for region in [0, 1]:
                for name in ["manual", "pseudo"]:
                    mask, target = selected[name + "_mask"], selected[name + "_target"]
                    for label in [0, 1]:
                        participant = mask[region] & (target[region] == label)
                        count = int(participant.sum())
                        expected_term = terms(zz[region][participant], tt[region][participant], target[region][participant]) if count else None
                        expected_strata.append({"region": region, "support": name, "hard_class": label, "count": count,
                            "fraction_of_own_support": count / int(mask.sum()), "terms": expected_term})
                        if name == "manual":
                            require(expected_term is not None, "Empty manual stratum")
                            balanced_groups.append(expected_term["student_smoothed_BCE"])
            require(len(output["strata"]) == len(expected_strata) == 8, "Eight strata required")
            for actual, expected in zip(output["strata"], expected_strata):
                for key in ["region", "support", "hard_class", "count"]:
                    require(actual[key] == expected[key], "Stratum identity/count differs")
                close(actual["fraction_of_own_support"], expected["fraction_of_own_support"], "strata/fraction")
                if expected["terms"] is None:
                    require(actual["terms"] is None, "Empty class must remain undefined")
                else:
                    check_terms(actual["terms"], expected["terms"])
            values = values_for_step(manual, pseudo, coefficient, old)
            require(set(output["values"]) == set(values), "Objective fields changed")
            for key, value in values.items():
                close(output["values"][key], value, "values/" + key)
            expected_logs = {"loss_U": manual["student_smoothed_BCE"], "loss_P": pseudo["student_smoothed_BCE"],
                "loss_R": sum(balanced_groups) / 4, "teacher_U": manual["initializer_smoothed_BCE"],
                "teacher_P": pseudo["initializer_smoothed_BCE"], "loss_D_M": manual["KL"], "loss_D_P": pseudo["KL"],
                "original_objective": values["base_student"], "total_loss": values["full_student"]}
            require(set(output["reconstruction_absolute_errors"]) == set(expected_logs), "Logged reconstruction fields changed")
            for key, value in expected_logs.items():
                tolerance = 1e-12 if key in {"loss_D_M", "loss_D_P"} else 2e-6
                close(log[key], value, "logged/" + key, tolerance)
                error = abs(log[key] - value)
                maximum_logged[key] = max(maximum_logged.get(key, 0.), error)
                close(output["reconstruction_absolute_errors"][key], error, "reported_reconstruction_error/" + key)
            close(old["total_loss"], old["loss_U"] + coefficient * old["loss_P"], "ninth/scalar_coefficient", 2e-6)
            require(log["unknown_output_gradient_max"] == 0 and math.isfinite(log["preclip_global_gradient_norm"])
                    and log["preclip_global_gradient_norm"] > 0, "Historical gradient receipt differs")
            active_count = manual["count"] + (pseudo["count"] if coefficient else 0)
            require(output["unknown_active_output_count"] == 2 * 128 * 128 - active_count, "Unknown output count differs")
            reconstructed.append({"step": index + 1, "manual": manual, "pseudo": pseudo, "values": values})
        windows = {}
        require(set(reported["windows"]) == set(WINDOWS), "Window scope changed")
        for name, steps in WINDOWS.items():
            source = [reconstructed[i - 1] for i in steps]
            expected = {key: math.fsum(row["values"][key] for row in source) / len(source) for key in values}
            actual = reported["windows"][name]
            require(actual["steps"] == steps and set(actual["equal_step_means"]) == set(expected), "Window recipe differs")
            for key, value in expected.items():
                close(actual["equal_step_means"][key], value, "window/" + key)
            counts = {support: {key: sum(row[support][key] for row in source) for key in ["count", "positive", "negative"]}
                      for support in ["manual", "pseudo"]}
            require(actual["support_counts"] == counts, "Window support counts differ")
            positive_base = sum(row["values"]["base_delta"] > 0 for row in source)
            positive_full = sum(row["values"]["full_delta"] > 0 for row in source)
            require(actual["steps_with_positive_base_delta"] == positive_base and actual["steps_with_positive_full_delta"] == positive_full, "Window paired-delta signs differ")
            windows[name] = {"steps": steps, "equal_step_means": expected, "support_counts": counts,
                             "steps_with_positive_base_delta": positive_base, "steps_with_positive_full_delta": positive_full}
        all_steps[arm] = {"steps_verified": 32, "strata_verified": 256, "windows": windows}
    for key, expected in maximum_logged.items():
        close(audit["maximum_logged_scalar_reconstruction_errors"][key], expected, "maximum_logged_error/" + key)
    require(audit["teacher_logits_bitwise_identical_between_arms"] is True, "Teacher summary differs")
    for key in ["checkpoint_loads", "model_calls", "backward_calls", "optimizer_calls", "network_calls"]:
        require(audit[key] == 0, "Forbidden audit action reported")
    csv_rows = list(csv.DictReader(args.audit.parent.joinpath("steps.csv").open()))
    require(len(csv_rows) == 64, "Audit CSV must contain64 steps")
    for row in csv_rows:
        arm, step = row["arm"], int(row["step"])
        require(arm in ARMS and 1 <= step <= 32, "CSV identity differs")
        expected = audit["arms"][arm]["steps"][step - 1]["values"]
        for key, value in expected.items():
            close(float(row[key]), value, "csv/" + key)
    require(len({(r["arm"], int(r["step"])) for r in csv_rows}) == 64, "CSV duplicates/missing steps")
    for item in before.values():
        verify(item)
    require(record(args.plan) == plan_pin and record(args.audit) == audit_pin and record(Path(__file__)) == code_pin,
            "Plan/audit/checker changed during review")
    require("torch" not in sys.modules, "Torch must not be imported")
    output = {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "plan": plan_pin,
              "audit": audit_pin, "checker": code_pin, "synthetic_fixtures": fixtures,
              "independent_implementations": ["log-probability cross entropy and teacher-to-student KL", "entropy-form free-output oracle", "explicit coordinate-index D4", "math.fsum equal-step window means"],
              "arm_checks": all_steps, "total_steps_verified": 64, "total_strata_verified": 512,
              "scalar_comparisons": comparison_count, "maximum_absolute_errors": maximum,
              "maximum_logged_scalar_errors": maximum_logged, "CSV_rows_verified": 64,
              "source_pins_unchanged": True, "teacher_bitwise_equality_verified": True,
              "actual_operations": {"retained_NPY_and_JSON_reads": True, "Torch_imports": 0,
                  "checkpoint_reads": 0, "models": 0, "network": 0, "new_training_or_predictions": 0},
              "scope": "Independent numerical reconstruction only; root owns scientific interpretation. Ninth evidence remains scalar receipts, not invented raw logits."}
    (args.output_dir / "independent-verification.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "result": record(args.output_dir / "independent-verification.json"),
                      "steps": 64, "strata": 512, "scalar_comparisons": comparison_count}))

if __name__ == "__main__":
    main()

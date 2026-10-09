"""Independent offline panel review; never import benchmark metric functions."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import numpy as np

ROLES = ["russo_ex01675k", "russo_control75k", "seed42", "seed43"]
LOCK_SHA = "3bb59b4a15caaa6c0fe433635535edf549719d35f25a644a9e218c2927236ff1"
RANK_PATH = Path("/workspace/herculaneum-lab/reports/sixth-experiment/independent-metric-review/review_metrics.py")
RANK_SHA = "e3cad20abca6f4daa82c34f25c3237ba4626fbda43c90cd36e0735f4923c0452"
METRIC_FIELDS = ("roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "supplied_positive_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(plan_path, expected_sha, run_dir, output):
    require(not output.exists(), "Independent review output must be fresh")
    require(sha(plan_path) == expected_sha, "Final panel plan SHA differs")
    records = {}
    maximum_error = 0.0

    def pin(path, expected=None):
        path = Path(path).resolve()
        item = dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)
        if expected is not None:
            require(item["sha256"] == expected["sha256"] and item["size_bytes"] == expected.get("size_bytes", expected.get("byte_length")), "Recorded hash/size differs: " + path.name)
        require(str(path) not in records or records[str(path)] == item, "Conflicting input receipts")
        records[str(path)] = item
        return path

    def collect(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys() and ("size_bytes" in value or "byte_length" in value):
                pin(value["path"], value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    def read(path):
        value = json.loads(pin(path).read_text())
        collect(value)
        return value

    def close(actual, expected, name):
        nonlocal maximum_error
        require((actual is None) == (expected is None), "Undefined metric mismatch: " + name)
        if actual is not None:
            error = abs(actual - expected)
            maximum_error = max(maximum_error, error)
            require(error <= 1e-12, "Metric mismatch: " + name)

    pin(__file__)
    require(sha(RANK_PATH) == RANK_SHA, "Immutable sixth independent ranking method changed")
    pin(RANK_PATH)
    spec = importlib.util.spec_from_file_location("independent_sixth_rank_formula", RANK_PATH)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    rank = independent.ranked_metrics
    plan = read(plan_path)
    require(plan["ready_for_execution"] is True and plan["model_order"] == ROLES, "Final four-arm panel plan required")
    require((run_dir / "frozen-plan.json").read_bytes() == plan_path.read_bytes(), "Run frozen plan differs")
    execution = read(run_dir / "execution.json")
    require(execution["status"] == "complete", "Benchmark did not complete")
    pairs = [(r["case_id"], r["role"]) for r in execution["runs"]]
    require(len(pairs) == len(plan["cases"]) * 4 and set(pairs) == {(c["id"], r) for c in plan["cases"] for r in ROLES}, "Missing/duplicate executed arm")
    require(all(r["exit_code"] == 0 for r in execution["runs"]), "An arm failed")
    lock_path = pin(plan["threshold_lock"]["path"], plan["threshold_lock"])
    require(sha(lock_path) == LOCK_SHA, "Actual threshold lock differs from independently reviewed lock")
    require((run_dir / "threshold-lock.json").read_bytes() == lock_path.read_bytes(), "Run threshold copy differs")
    lock = read(lock_path)
    require(lock["model_order"] == ROLES and lock["development_case"] == "pherc1667-w029", "Development model/data identity differs")
    report = read(run_dir / "analysis/metrics.json")
    integrity = read(run_dir / "analysis/final-integrity.json")
    require(integrity["status"] == "passed" and integrity["verified_after_analysis"], "Final integrity not passed")
    require(report["plan"]["sha256"] == expected_sha and report["threshold_lock"] == plan["threshold_lock"], "Reported plan/threshold identity differs")
    for path in run_dir.rglob("*"):
        if path.is_file():
            pin(path)
    require(set(report["contexts"]) == {c["id"] for c in plan["cases"]}, "Context set differs")
    audit = read(plan["model_contract_audit"]["path"])
    require(audit["status"] == "passed", "Model contract audit did not pass")
    for artifact in plan["models"]:
        role = artifact["label"]
        require(artifact["sha256"] == lock["models_and_maps"][role]["checkpoint"]["sha256"] == audit["models"][role]["sha256"], "Checkpoint identity differs")
        pin(artifact["path"], artifact)

    def operating(y, s, threshold):
        predicted = s.astype(np.float64) >= threshold
        positive, negative = int(y.sum()), int((~y).sum())
        tp, fp = int((y & predicted).sum()), int((~y & predicted).sum())
        return dict(threshold=threshold, pixel_count=int(y.size), supplied_positive_pixels=positive,
                    supplied_negative_pixels=negative, confusion=dict(tp=tp, fp=fp, tn=negative-fp, fn=positive-tp),
                    sensitivity=tp/positive if positive else None, specificity=(negative-fp)/negative if negative else None,
                    precision=tp/(tp+fp) if tp+fp else None,
                    balanced_accuracy=(tp/positive+(negative-fp)/negative)/2 if positive and negative else None)

    def compare_operating(actual, expected, name):
        for key in ("threshold", "pixel_count", "supplied_positive_pixels", "supplied_negative_pixels", "confusion"):
            require(actual[key] == expected[key], "Locked operating count differs: " + name + "/" + key)
        require(expected["threshold_operator"] == "float64(score) >= threshold", "Locked threshold operator differs")
        for key in ("sensitivity", "specificity", "precision", "balanced_accuracy"):
            close(actual[key], expected[key], name + "/" + key)

    # Reconstruct development thresholds without the operating-point helper:
    # kth negative boundary via partition, then increment its positive float32 bits.
    dev_y = np.load(lock["source_inputs"]["inklabels"]["path"], allow_pickle=False) == 1
    dev_mask = np.load(lock["primary_mask"]["path"], allow_pickle=False)
    require(dev_mask.dtype == bool and int(dev_mask.sum()) == 16384, "Development mask differs")
    dev_y = dev_y[dev_mask]
    development = {}
    for role in ROLES:
        s = np.load(lock["models_and_maps"][role]["scores"]["path"], allow_pickle=False)[dev_mask]
        require(s.dtype == np.float32 and np.isfinite(s).all(), "Development scores invalid")
        negatives = s[~dev_y]
        allowed = negatives.size // 20
        boundary = np.partition(negatives, negatives.size - 1 - allowed)[negatives.size - 1 - allowed]
        bits = int(np.asarray(boundary, dtype=np.float32).view(np.uint32))
        threshold = float(np.asarray(1 if boundary == 0 else bits + 1, dtype=np.uint32).view(np.float32))
        fixed = lock["thresholds"][role]
        require(fixed["threshold"] == threshold and fixed["excluded_negative_boundary"] == float(boundary), "Development threshold reconstruction differs")
        require(fixed["maximum_false_positives"] == allowed and fixed["negative_count"] == negatives.size, "Development FP budget differs")
        require(fixed["boundary_tie_count"] == int((negatives == boundary).sum()), "Development tie count differs")
        require(int((negatives.astype(np.float64) >= float(boundary)).sum()) > allowed, "Threshold is not crossing the declared FP budget")
        value = operating(dev_y, s, threshold)
        require(value["confusion"]["fp"] == fixed["development_false_positives"] <= allowed, "Development FP bound differs")
        compare_operating(value, fixed["development_operating_metrics"], "development/" + role)
        development[role] = dict(threshold=threshold, boundary=float(boundary), maximum_fp=int(allowed), metrics=value)

    contexts = {}
    for case in plan["cases"]:
        ident = case["id"]
        arrays = {k: np.load(v["path"], allow_pickle=False) for k, v in case["inputs"].items()}
        surface = arrays["surface"]
        require(surface.dtype == np.uint8 and list(surface.shape) == case["data"]["shape"] == [21,384,384], "Surface contract differs")
        labels = {}
        for key in ("inklabels", "supervision_mask", "validation_mask"):
            a = arrays[key]
            foreground = case["data"]["binary_foreground"][key]
            require(a.dtype == np.uint8 and a.shape == (384,384) and set(np.unique(a)) <= {0,foreground}, "Annotation encoding differs")
            labels[key] = a == foreground
        y = labels["inklabels"]
        observed, occupancy = arrays["acquisition_observed_mask"], arrays["source_observed_nonzero_mask"]
        require(observed.dtype == occupancy.dtype == bool and observed.shape == occupancy.shape == y.shape and observed.all(), "Observed support differs")
        require(np.array_equal(occupancy, np.any(surface[2:19] != 0, axis=0)), "Occupancy differs")
        common = np.ones(y.shape, bool)
        scores = {}
        for role in ROLES:
            metadata = read(run_dir / ident / role / "metadata.json")
            require(metadata["case_id"] == ident and metadata["role"] == role and metadata["plan"]["sha256"] == expected_sha, "Output identity differs")
            require(metadata["source_inputs"] == case["inputs"] and metadata["source_provenance"] == case["provenance"] and metadata["data_contract"] == case["data"], "Source identity differs")
            require(metadata["threshold_lock"] == plan["threshold_lock"] and metadata["execution_code"] == plan["code"], "Execution or threshold provenance differs")
            model = metadata["model"]
            require(model["sha256"] == lock["models_and_maps"][role]["checkpoint"]["sha256"] and model["step"] == 74999 and model["strict_state_match"] and not model["missing_keys"] and not model["unexpected_keys"], "Strict model identity differs")
            s = np.load(metadata["raw_scores"]["path"], allow_pickle=False)
            evaluated = np.load(metadata["evaluated_mask"]["path"], allow_pickle=False)
            require(s.dtype == np.float32 and s.shape == y.shape and evaluated.dtype == bool and evaluated.shape == y.shape, "Raw output shape/dtype differs")
            require(np.all((s[np.isfinite(s)] >= 0) & (s[np.isfinite(s)] <= 1)), "Raw score range differs")
            scores[role] = s
            common &= evaluated & np.isfinite(s)
        interior = np.zeros(y.shape, bool)
        interior[64:-64,64:-64] = True
        before = dict(primary=labels["validation_mask"] & observed & interior, secondary=labels["validation_mask"] & observed)
        masks = {k: v & common for k, v in before.items()}
        saved_context = report["contexts"][ident]
        support = read(run_dir / "analysis" / ident / "common-support.json")
        local_report = read(run_dir / "analysis" / ident / "metrics.json")
        require(local_report == saved_context, "Context report differs from final report")
        require(saved_context["physical_group"] == case["physical_group"] and saved_context["data"] == case["data"], "Context physical/data identity differs")
        context = dict(physical_group=case["physical_group"], domains={})
        for name, mask in masks.items():
            require(np.array_equal(np.load(saved_context["masks"][name]["path"], allow_pickle=False), mask), "Saved common domain differs")
            counts = dict(pixel_count=int(mask.sum()), supplied_positive_pixels=int((mask & y).sum()), supplied_negative_pixels=int((mask & ~y).sum()))
            require(counts == support["final"][name] == saved_context["domains"][name]["counts"], "Final domain counts differ")
            require(support[name + "_lost_pixels"] == int((before[name] & ~common).sum()), "Common support losses differ")
            if name == "primary":
                require(min(counts["supplied_positive_pixels"], counts["supplied_negative_pixels"]) >= 256, "Primary class gate failed")
            models = {}
            for role in ROLES:
                r = rank(y[mask], scores[role][mask])
                saved = saved_context["domains"][name]["models"][role]
                for field in ("pixel_count", "supplied_positive_pixels", "supplied_negative_pixels"):
                    require(r[field] == saved[field], "Ranking class counts differ")
                for field in METRIC_FIELDS:
                    close(r[field], saved[field], ident + "/" + name + "/" + role + "/" + field)
                require({k:r["fixed_threshold"][k] for k in ("tp","fp","tn","fn")} == saved["confusion"], "Midpoint confusion differs")
                require(r["unique_score_count"] == saved["distinct_score_groups"] and r["tied_score_pixel_count"] == saved["tied_score_pixel_count"], "Score tie counts differ")
                locked = operating(y[mask], scores[role][mask], development[role]["threshold"])
                compare_operating(locked, saved["development_locked_operating_point"], ident + "/" + name + "/" + role)
                q = locked["supplied_negative_pixels"]
                transfer = locked["confusion"]["tn"] * 20 >= q * 19 if q else None
                require(saved["development_locked_operating_point"]["reserved_specificity_meets95_percent"] == transfer, "Specificity transfer flag differs")
                r["development_locked_operating_point"] = {**locked, "reserved_specificity_meets95_percent": transfer}
                strata = {}
                for stratum, subset in (("nonzero_profiles", mask & occupancy), ("observed_zero_profiles", mask & ~occupancy)):
                    value = rank(y[subset], scores[role][subset])
                    stratum_saved = saved["occupancy_strata"][stratum]
                    for field in ("pixel_count", "supplied_positive_pixels", "supplied_negative_pixels"):
                        require(value[field] == stratum_saved[field], "Occupancy class counts differ")
                    for field in METRIC_FIELDS:
                        close(value[field], stratum_saved[field], ident + "/" + name + "/" + role + "/" + stratum + "/" + field)
                    require({k:value["fixed_threshold"][k] for k in ("tp","fp","tn","fn")} == stratum_saved["confusion"], "Occupancy midpoint confusion differs")
                    value["development_locked_operating_point"] = operating(y[subset], scores[role][subset], development[role]["threshold"])
                    compare_operating(value["development_locked_operating_point"], stratum_saved["development_locked_operating_point"], "occupancy/" + stratum)
                    strata[stratum] = value
                r["occupancy_strata"] = strata
                models[role] = r
            differences = {}
            for field in ("roc_auc", "average_precision"):
                key = "ex016_minus_control_" + field
                differences[key] = models[ROLES[0]][field] - models[ROLES[1]][field]
                close(differences[key], saved_context["domains"][name]["paired_ex016_control"][key], key)
            for field in ("sensitivity", "specificity"):
                key = "ex016_minus_control_locked_" + field
                differences[key] = models[ROLES[0]]["development_locked_operating_point"][field] - models[ROLES[1]]["development_locked_operating_point"][field]
                close(differences[key], saved_context["domains"][name]["paired_ex016_control"][key], key)
            context["domains"][name] = dict(counts=counts, common_mask_formula_matches=True, models=models, paired_ex016_control=differences)
        contexts[ident] = context

    def mean_range(values):
        return dict(mean=math.fsum(values)/len(values), min=min(values), max=max(values))

    groups = {}
    for group in sorted({c["physical_group"] for c in contexts.values()}):
        ids = [k for k, v in contexts.items() if v["physical_group"] == group]
        saved = report["physical_group_summaries"][group]
        require(saved["contexts"] == ids and saved["context_count"] == len(ids), "Specimen grouping differs")
        models = {}
        for role in ROLES:
            values = [contexts[k]["domains"]["primary"]["models"][role] for k in ids]
            out = {}
            for field in ("roc_auc", "average_precision", "supplied_positive_fraction", "locked_sensitivity", "locked_specificity"):
                samples = [v["development_locked_operating_point"][field[7:]] if field.startswith("locked_") else v[field] for v in values]
                out[field] = mean_range(samples)
                for key in ("mean", "min", "max"):
                    close(out[field][key], saved["equal_context_macro"][role][field][key], "equal-context/" + group + "/" + role + "/" + field + "/" + key)
            models[role] = out
        paired = {}
        for field in ("ex016_minus_control_roc_auc", "ex016_minus_control_average_precision", "ex016_minus_control_locked_sensitivity", "ex016_minus_control_locked_specificity"):
            paired[field] = mean_range([contexts[k]["domains"]["primary"]["paired_ex016_control"][field] for k in ids])
            for key in ("mean", "min", "max"):
                close(paired[field][key], saved["paired_ex016_control"][field][key], "paired/equal-context")
        groups[group] = dict(contexts=ids, context_count=len(ids), equal_context_macro=models, paired_ex016_control=paired)
    require(set(groups) == {"PHerc0139", "PHerc0814"} == set(report["physical_group_summaries"]), "Physical group set differs")
    across = {}
    for role in ROLES:
        across[role] = {}
        for field in ("roc_auc", "average_precision", "locked_sensitivity", "locked_specificity"):
            value = math.fsum(g["equal_context_macro"][role][field]["mean"] for g in groups.values()) / len(groups)
            close(value, report["equal_specimen_macro"][role][field], "equal-specimen/" + role + "/" + field)
            across[role][field] = value
    for path, record in records.items():
        require(sha(path) == record["sha256"] and Path(path).stat().st_size == record["size_bytes"], "Input/output changed during independent review")
    output.mkdir(parents=True)
    result = dict(status="PASS", plan_sha256=expected_sha, threshold_lock_sha256=LOCK_SHA,
                  independent_rank_method_sha256=RANK_SHA, reviewer_code_sha256=sha(__file__),
                  verified_input_output_records_before_and_after=list(records.values()), verified_unique_records=len(records),
                  tolerance=1e-12, maximum_absolute_metric_difference=maximum_error, development_threshold_reconstruction=development,
                  contexts=contexts, physical_group_summaries=groups, equal_specimen_macro=across,
                  benchmark_metric_code_imported=False, inference_calls=0, network_requests=0,
                  interpretation="Descriptive supplied-reference agreement, equal context then equal specimen weighting. No pixel CI, calibrated confidence, ink authentication or historical training/teacher holdout assertion.")
    target = output / "independent-verification.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], unique_records=len(records), maximum_absolute_difference=maximum_error, path=str(target), sha256=sha(target))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    def blocked(*args, **kwargs):
        raise RuntimeError("Independent metric review attempted network access")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.socket.sendto = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    review(args.plan.resolve(), args.plan_sha256, args.run_dir.resolve(), args.output_dir.resolve())

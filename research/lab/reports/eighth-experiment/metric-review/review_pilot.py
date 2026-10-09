"""One offline independent receipt/metric review; no forward or backward calls."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import numpy as np

ARMS = ["frozen_seed42", "A", "B", "C"]
RANK_PATH = Path("/workspace/herculaneum-lab/reports/sixth-experiment/independent-metric-review/review_metrics.py")
RANK_SHA = "e3cad20abca6f4daa82c34f25c3237ba4626fbda43c90cd36e0735f4923c0452"
FIELDS = ("roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "supplied_positive_fraction")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(plan_path, expected_sha, run_dir, analysis_dir, output):
    require(not output.exists(), "Fresh independent output required")
    require(sha(plan_path) == expected_sha, "Final plan SHA differs")
    pins = {}
    max_error = 0.0
    max_loss_error = 0.0

    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path))
        if expected is not None:
            require(actual["sha256"] == expected["sha256"] and actual["size_bytes"] == expected.get("size_bytes", expected.get("byte_length")), "Hash/size mismatch: " + path.name)
        require(str(path) not in pins or pins[str(path)] == actual, "Conflicting receipts")
        pins[str(path)] = actual
        return path

    def collect(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys() and ("size_bytes" in value or "byte_length" in value):
                pin(value["path"], value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    def read(path):
        value = json.loads(pin(path).read_text())
        collect(value)
        return value

    def close(actual, expected, label):
        nonlocal max_error
        require((actual is None) == (expected is None), "Undefined metric differs: " + label)
        if actual is not None:
            error = abs(actual - expected)
            max_error = max(max_error, error)
            require(error <= 1e-12, "Metric differs: " + label)

    def load(item):
        return np.load(pin(item["path"], item), allow_pickle=False)

    pin(__file__)
    require(sha(RANK_PATH) == RANK_SHA, "Independent ranking source changed")
    pin(RANK_PATH)
    spec = importlib.util.spec_from_file_location("immutable_independent_rank", RANK_PATH)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    rank = independent.ranked_metrics
    plan = read(plan_path)
    require(plan["kind"] == "matched-continuation-v1" and plan["ready_for_execution"], "Ready matched plan required")
    require(read(run_dir / "frozen-plan.json") == plan, "Controller frozen plan differs")
    execution = read(run_dir / "execution.json")
    require(execution["plan"]["sha256"] == expected_sha and execution["retries"] == 0, "Execution plan/retry scope differs")
    require([w["arm"] for w in execution["workers"]] == ARMS and all(w["exit_code"] == 0 for w in execution["workers"]), "Four once-only successful workers required")
    report = read(analysis_dir / "metrics.json")
    require(report["status"] == "PASS" and report["plan"]["sha256"] == expected_sha, "Analysis plan/status differs")
    workers = {arm: read(run_dir / arm / "result.json") for arm in ARMS}
    for arm, worker in workers.items():
        require(worker["status"] == "PASS" and worker["arm"] == arm and worker["plan"]["sha256"] == expected_sha, "Worker identity differs")
        require(worker["initializer"] == plan["initializer"] and worker["model_contract"]["sha256"] == plan["initializer"]["sha256"], "Initializer artifact differs")
        require(worker["model_contract"]["strict_state_match"] and worker["threads"] == 4, "Strict initializer/runtime differs")
        require(worker["peak_rss_bytes"] <= plan["limits"]["rss_acceptance_bytes"] and worker["cpu_seconds"] <= plan["limits"]["cpu_seconds"] and worker["wall_seconds"] <= plan["limits"]["wall_seconds"], "Worker resource receipt exceeds bound")
        require(report["model_results"][arm]["sha256"] == sha(run_dir / arm / "result.json"), "Analysis worker receipt differs")
    require(len({w["initial_state_sha256"] for w in workers.values()}) == 1, "Initial states differ")
    for directory in (run_dir, analysis_dir):
        for path in directory.rglob("*"):
            if path.is_file():
                pin(path)
    require(sum(path.stat().st_size for directory in (run_dir,analysis_dir) for path in directory.rglob("*") if path.is_file()) <= plan["limits"]["output_bytes"], "Run/analysis output budget exceeded")

    def arrays(case):
        a = {k: load(v) for k, v in case["inputs"].items()}
        surface, observed = a["surface"], a["acquisition_observed_mask"]
        shape = surface.shape[1:]
        require(surface.dtype == np.uint8 and surface.shape[0] == 21 and shape in ((256,256),(384,384)), "Source shape/dtype differs")
        require(observed.dtype == bool and observed.shape == shape and observed.all(), "Acquired source differs")
        for key in ("inklabels", "supervision_mask", "validation_mask"):
            require(a[key].shape == shape and a[key].dtype == np.uint8 and set(np.unique(a[key])) <= {0,1}, "Binary reference encoding differs")
        require(a["source_observed_nonzero_mask"].dtype == bool and np.array_equal(a["source_observed_nonzero_mask"], np.any(surface[2:19] != 0, axis=0)), "Occupancy differs")
        return a

    def domain(case):
        a = arrays(case)
        role = case["evaluation_participation_role"]
        require(role in ("public_validation", "supplied_supervision"), "Evaluation role differs")
        part = a["validation_mask"] > 0 if role == "public_validation" else a["supervision_mask"] > 0
        require(np.array_equal(part, a["evaluation_participation_mask"] > 0), "Participation identity differs")
        mask = np.zeros(part.shape, bool)
        mask[64:-64,64:-64] = True
        mask &= part & a["acquisition_observed_mask"]
        return a, a["inklabels"] > 0, mask

    schedule = read(plan["schedule"]["path"])
    require(len(schedule) == 32, "32 steps required")
    training_sources = []
    for case in plan["adaptation"]:
        require(case["physical_group"] == "PHerc0139", "Adaptation specimen differs")
        a = arrays(case)
        manual = (a["supervision_mask"] > 0) & (a["validation_mask"] == 0) & a["acquisition_observed_mask"]
        teacher, evaluated = load(case["teacher_scores"]), load(case["teacher_evaluated"])
        require(teacher.dtype == np.float32 and evaluated.dtype == bool and teacher.shape == evaluated.shape == manual.shape, "Teacher map contract differs")
        pseudo = a["acquisition_observed_mask"] & evaluated & np.isfinite(teacher) & ~manual
        require(not (manual & pseudo).any(), "Manual and pseudo masks overlap")
        training_sources.append((manual, a["inklabels"] == 1, pseudo, teacher >= .5))
    require(len(training_sources) == 2, "Two adaptation contexts required")
    grid_support = []
    for index,(m,y0,p,t) in enumerate(training_sources):
        eligible = []
        covered = np.zeros(m.shape,bool)
        for y in range(0,257,64):
            for x in range(0,257,64):
                sl=np.s_[y:y+128,x:x+128]
                if min(int((m[sl]&y0[sl]).sum()),int((m[sl]&~y0[sl]).sum())) >= 256:
                    eligible.append([x,y,128,128]);covered[sl]=True
        actual=dict(id=plan["adaptation"][index]["id"],eligible_tiles=eligible,
                    unique_manual_pixels=int((covered&m).sum()),unique_pseudo_positive_pixels=int((covered&p&t).sum()),
                    unique_pseudo_negative_pixels=int((covered&p&~t).sum()),whole_context_pseudo_positive_pixels=int((p&t).sum()))
        require(actual == plan["exposure"][index], "Eligible-grid support differs")
        grid_support.append(actual)
    expected_exposure = []
    for index, step in enumerate(schedule, 1):
        require(step["step"] == index and [p["region"] for p in step["patches"]] == [0,1], "Matched batch region order differs")
        exposures = []
        for patch in step["patches"]:
            x,y,w,h = patch["bbox"]
            require(w == h == 128 and x % 64 == y % 64 == 0 and 0 <= x <= 256 and 0 <= y <= 256 and 0 <= patch["augmentation"] < 8, "Supported grid/D4 differs")
            require(patch["bbox"] in grid_support[patch["region"]]["eligible_tiles"], "Schedule patch not eligible")
            m, y0, p, t = training_sources[patch["region"]]
            sl = np.s_[y:y+128,x:x+128]
            # D4 is a bijection, so class-count receipts are invariant to its orientation.
            exposure = dict(region=patch["region"], manual=dict(positive=int((m[sl]&y0[sl]).sum()), negative=int((m[sl]&~y0[sl]).sum())),
                            pseudo=dict(positive=int((p[sl]&t[sl]).sum()), negative=int((p[sl]&~t[sl]).sum())))
            require(min(exposure["manual"].values()) >= 256, "Manual class support differs")
            exposures.append(exposure)
        expected_exposure.append(exposures)
    training_receipts = {}
    for arm in ("A", "B", "C"):
        worker = workers[arm]
        training = worker["training"]
        require(training["steps"] == 32 and training["schedule"] == plan["schedule"] and training["initial_state_sha256"] == worker["initial_state_sha256"], "Training receipt differs")
        require(training["exposure"] == plan["exposure"] and training["unsmoothed_targets"] is True and training["inherited_checkpoint_bce_label_smoothing"] == .5, "Training target/support receipt differs")
        rows = [json.loads(line) for line in pin(training["steps_log"]["path"], training["steps_log"]).read_text().splitlines()]
        require(len(rows) == 32, "Training update count differs")
        class_exposure = {role: dict(positive=0, negative=0) for role in ("manual", "pseudo")}
        for index, row in enumerate(rows):
            step = schedule[index]
            require(row["step"] == step["step"] and row["patches"] == step["patches"] and row["exposures"] == expected_exposure[index], "Per-step schedule/exposure differs")
            require(row["unknown_output_gradient_max"] == 0.0 and row["preclip_global_gradient_norm"] > 0 and math.isfinite(row["preclip_global_gradient_norm"]), "Gradient receipt differs")
            probe = row["parameter_update_probe"]
            require(math.isfinite(probe["before"]) and math.isfinite(probe["after"]) and probe["before"] != probe["after"], "Parameter update receipt differs")
            require(row["peak_rss_bytes"] <= plan["limits"]["rss_acceptance_bytes"], "Step memory exceeded acceptance")
            u,p,r = [row["loss_"+k] for k in ("U","P","R")]
            require(all(math.isfinite(v) and v >= 0 for v in (u,p,r,row["total_loss"])), "Loss receipt invalid")
            expected_total = float(np.float32(u)) if arm == "A" else float(np.float32(np.float32(u) + np.float32(.25)*np.float32(p))) if arm == "B" else float(np.float32(np.float32(np.float32(.5)*np.float32(u) + np.float32(.5)*np.float32(r)) + np.float32(.25)*np.float32(p)))
            error = abs(row["total_loss"] - expected_total)
            max_loss_error = max(max_loss_error, error)
            require(error <= 2e-6, "Per-step loss coefficient arithmetic differs")
            for exposure in row["exposures"]:
                for role in class_exposure:
                    for label in ("positive", "negative"):
                        class_exposure[role][label] += exposure[role][label]
        require(training["initial_parameter_sha256"] != training["final_parameter_sha256"], "No recorded weight change")
        training_receipts[arm] = dict(steps=32, identical_schedule_and_exposure=True, class_exposure_with_repetition=class_exposure,
                                      max_scalar_loss_formula_error=max_loss_error, actual_parameter_update_probes=32,
                                      loss_limit="Checks saved U/P/R coefficient arithmetic; raw training logits and four class BCE values were not retained, so these values are not independently recomputed")
    require(workers["frozen_seed42"]["training"] is None, "Frozen baseline trained")
    require(len({json.dumps(v["class_exposure_with_repetition"], sort_keys=True) for v in training_receipts.values()}) == 1, "Arm exposures differ")

    def operating(y, scores, threshold):
        predicted = scores.astype(np.float64) >= threshold
        p,q = int(y.sum()), int((~y).sum())
        tp,fp = int((y & predicted).sum()), int((~y & predicted).sum())
        return dict(threshold=float(threshold), pixel_count=int(y.size), supplied_positive_pixels=p, supplied_negative_pixels=q,
                    confusion=dict(tp=tp,fp=fp,tn=q-fp,fn=p-tp), sensitivity=tp/p if p else None, specificity=(q-fp)/q if q else None,
                    precision=tp/(tp+fp) if tp+fp else None, balanced_accuracy=(tp/p+(q-fp)/q)/2 if p and q else None)

    def check_operating(actual, saved, label):
        for key in ("threshold", "pixel_count", "supplied_positive_pixels", "supplied_negative_pixels", "confusion"):
            require(actual[key] == saved[key], "Operating count differs: " + label)
        for key in ("sensitivity", "specificity", "precision", "balanced_accuracy"):
            close(actual[key], saved[key], label+"/"+key)

    def check_ranking(actual, saved, label):
        for key in ("pixel_count", "supplied_positive_pixels", "supplied_negative_pixels"):
            require(actual[key] == saved[key], "Ranking counts differ: " + label)
        for key in FIELDS:
            close(actual[key], saved[key], label+"/"+key)
        require({k:actual["fixed_threshold"][k] for k in ("tp","fp","tn","fn")} == saved["confusion"], "Midpoint confusion differs")
        require(actual["unique_score_count"] == saved["distinct_score_groups"] and actual["tied_score_pixel_count"] == saved["tied_score_pixel_count"], "Score ties differ")

    locks = {}
    calibration = {}
    for arm, worker in workers.items():
        lock = read(worker["threshold_lock"]["path"])
        require(lock["arm"] == arm and lock["plan"]["sha256"] == expected_sha and lock["fresh_inference_started"] is False, "Pre-fresh lock identity differs")
        selections = {}
        scores_by_case = {}
        for case in plan["development"]:
            a,y,mask = domain(case)
            maps = worker["development"][case["id"]]
            if arm == "frozen_seed42":
                require(maps == case["baseline_maps"], "Frozen development maps were replaced")
            scores,evaluated = load(maps["scores"]), load(maps["evaluated"])
            require(scores.dtype == np.float32 and scores.shape == mask.shape and evaluated.dtype == bool and evaluated[mask].all() and np.isfinite(scores[mask]).all(), "Development support differs")
            neg = scores[mask & ~y]
            allowed = neg.size // 20
            boundary = np.partition(neg, neg.size - 1 - allowed)[neg.size - 1 - allowed]
            bits = int(np.asarray(boundary, dtype=np.float32).view(np.uint32))
            threshold = float(np.asarray(1 if boundary == 0 else bits+1, dtype=np.uint32).view(np.float32))
            saved_selection = lock["development"][case["id"]]["selection"]
            require(saved_selection["threshold"] == threshold and saved_selection["excluded_negative_boundary"] == float(boundary), "Development cutoff differs")
            require(saved_selection["negative_count"] == neg.size and saved_selection["maximum_false_positives"] == allowed and saved_selection["boundary_tie_count"] == int((neg == boundary).sum()), "Cutoff negative/tie budget differs")
            fp = int((neg.astype(np.float64) >= threshold).sum())
            require(saved_selection["development_false_positives"] == fp <= allowed, "Local development FP bound differs")
            require(int((neg.astype(np.float64) >= float(boundary)).sum()) > allowed, "Cutoff not at crossing boundary")
            selections[case["id"]] = threshold
            scores_by_case[case["id"]] = (y[mask],scores[mask])
        primary = max(selections.values())
        secondary = selections["pherc1667-w029"]
        require(lock["primary"] == primary and lock["secondary"] == secondary, "Max-of-three/secondary threshold differs")
        locks[arm] = dict(primary=primary,secondary=secondary,region_cutoffs=selections)
        calibration[arm] = {}
        for ident,(y,scores) in scores_by_case.items():
            r = rank(y,scores)
            saved = report["calibration"][arm][ident]
            check_ranking(r,saved["ranking"],"calibration/"+arm+"/"+ident)
            ops = {rule:operating(y,scores,locks[arm][rule]) for rule in ("primary","secondary")}
            for rule,value in ops.items():
                check_operating(value,saved["operating"][rule],"calibration/"+rule)
            require(ops["primary"]["confusion"]["fp"] <= ops["primary"]["supplied_negative_pixels"]//20, "Global cutoff failed a development region")
            calibration[arm][ident] = dict(ranking=r,operating=ops)

    fresh = {}
    for case in plan["fresh"]:
        a,y,before = domain(case)
        require(case["physical_group"] == "PHerc0814" and case["evaluation_participation_role"] == "supplied_supervision", "Fresh identity differs")
        scores_by_arm = {}
        mask = before.copy()
        for arm,worker in workers.items():
            maps = worker["fresh"][case["id"]]
            scores,evaluated = load(maps["scores"]),load(maps["evaluated"])
            require(scores.dtype == np.float32 and scores.shape == before.shape and evaluated.dtype == bool and evaluated.shape == before.shape, "Fresh array contract differs")
            require(np.all((scores[np.isfinite(scores)] >= 0) & (scores[np.isfinite(scores)] <= 1)), "Fresh score range differs")
            scores_by_arm[arm] = scores
            mask &= evaluated & np.isfinite(scores)
        stored_mask = np.load(pin(analysis_dir / case["id"] / "primary-common-mask.npy"),allow_pickle=False)
        require(stored_mask.dtype == bool and np.array_equal(mask,stored_mask), "Fresh common mask differs")
        saved = report["fresh"][case["id"]]
        require(saved["fixed_primary_count"] == int(before.sum()) and saved["common_primary_count"] == int(mask.sum()) and saved["positive"] == int((mask&y).sum()) and saved["negative"] == int((mask&~y).sum()), "Fresh class counts differ")
        require(min(saved["positive"],saved["negative"]) >= 256, "Common class gate failed")
        context = dict(physical_group="PHerc0814",bbox_source_xywh=case["data"]["bbox_source_xywh"],counts=dict(pixel_count=int(mask.sum()),positive=int((mask&y).sum()),negative=int((mask&~y).sum())),models={})
        for arm,scores in scores_by_arm.items():
            values,labels = scores[mask],y[mask]
            r = rank(labels,values)
            model = saved["models"][arm]
            check_ranking(r,model["midpoint"],"fresh/"+case["id"]+"/"+arm)
            close(r["roc_auc"],model["auc"],"freshAUC");close(r["average_precision"],model["ap"],"freshAP")
            ops = {rule:operating(labels,values,locks[arm][rule]) for rule in ("primary","secondary")}
            for rule,value in ops.items():
                check_operating(value,model["operating"][rule],"fresh/"+rule)
                met = value["confusion"]["fp"] <= value["supplied_negative_pixels"]//20
                require(model["operating"][rule]["specificity_target_met_exact"] == met, "Fresh specificity transfer flag differs")
                value["specificity_target_met_exact"] = met
            quantiles = dict(zip(("min","q05","median","q95","max"),map(float,np.quantile(values,[0,.05,.5,.95,1]))))
            for key,value in quantiles.items():close(value,model["score_quantiles"][key],"scoreQuantile")
            strata = {}
            for name,occupancy in (("observed_nonzero_profile",a["source_observed_nonzero_mask"]),("observed_zero_profile",~a["source_observed_nonzero_mask"])):
                submask = mask & occupancy
                sub = rank(y[submask],scores[submask])
                check_ranking(sub,model["strata"][name],"stratum/"+name)
                strata[name] = sub
            context["models"][arm] = dict(auc=r["roc_auc"],ap=r["average_precision"],midpoint=r,operating=ops,score_quantiles=quantiles,strata=strata)
        fresh[case["id"]] = context
    require(set(fresh) == set(report["fresh"]) and len(fresh) == 2, "Fresh context set differs")
    macro = {}
    for arm in ARMS:
        values = [c["models"][arm] for c in fresh.values()]
        macro[arm] = dict(auc=math.fsum(v["auc"] for v in values)/2,ap=math.fsum(v["ap"] for v in values)/2,
                         primary_sensitivity=math.fsum(v["operating"]["primary"]["sensitivity"] for v in values)/2,
                         primary_specificity=math.fsum(v["operating"]["primary"]["specificity"] for v in values)/2)
        for field,value in macro[arm].items():close(value,report["equal_context_macro"][arm][field],"macro/"+arm+"/"+field)
    gates = {}
    for comparator in plan["gate"]["comparators"]:
        differences = []
        exact_sensitivity = []
        constraints = []
        for case in fresh.values():
            c,other = case["models"]["C"],case["models"][comparator]
            positive = case["counts"]["positive"]
            delta = Fraction(c["operating"]["primary"]["confusion"]["tp"] - other["operating"]["primary"]["confusion"]["tp"],positive)
            row = dict(sensitivity=float(delta),auc=c["auc"]-other["auc"],ap=c["ap"]-other["ap"])
            exact_sensitivity.append(delta);differences.append(row)
            constraints.append(c["operating"]["primary"]["specificity_target_met_exact"] and delta >= Fraction(-1,50) and row["auc"] >= -.01 and row["ap"] >= -.02)
        mean_exact = sum(exact_sensitivity,Fraction())/2
        passed = all(constraints) and mean_exact >= Fraction(1,50)
        saved = report["gates"][comparator]
        require(saved["pass_screen"] == passed, "Predeclared screen differs")
        close(float(mean_exact),saved["mean_sensitivity_delta"],"gateMean")
        for actual,expected in zip(differences,saved["per_context_deltas"]):
            for field,value in actual.items():close(value,expected[field],"gateDelta")
        gates[comparator] = dict(pass_screen=passed,per_context_deltas=differences,mean_sensitivity_delta=float(mean_exact),sensitivity_gates_checked_as_exact_rational_counts=True)
    require(report["promising_candidate_beyond_cheap_controls"] == all(g["pass_screen"] for g in gates.values()), "Overall candidate gate differs")
    checkpoint_module_path = pin(Path(__file__).with_name("checkpoint_receipts.py"))
    spec = importlib.util.spec_from_file_location("independent_checkpoint_receipts",checkpoint_module_path)
    checkpoint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_module)
    checkpoints = checkpoint_module.verify_checkpoints(plan,workers)
    collect(checkpoints)
    for path,item in pins.items():
        require(Path(path).stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "Input/output changed during independent review")
    output.mkdir(parents=True)
    result = dict(status="PASS",plan_sha256=expected_sha,independent_rank_method_sha256=RANK_SHA,
                  independent_code_sha256=sha(__file__),checkpoint_review=checkpoints,verified_unique_records=len(pins),
                  verified_input_output_records_before_and_after=list(pins.values()),actual_training_updates=96,
                  training_receipts=training_receipts,eligible_grid_support=grid_support,max_saved_loss_formula_difference=max_loss_error,
                  development_thresholds=locks,calibration=calibration,fresh=fresh,equal_context_macro=macro,gates=gates,
                  promising_candidate_beyond_cheap_controls=all(g["pass_screen"] for g in gates.values()),
                  maximum_absolute_metric_difference=max_error,tolerance=1e-12,network_requests=0,forward_calls=0,backward_calls=0,
                  benchmark_metric_code_imported=False,interpretation="One-seed descriptive continuation on public reference annotations; same-specimen fresh contexts, unresolved historical/teacher exposure, no independent ink accuracy, pixel CI or novelty claim")
    target = output / "independent-verification.json"
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS",path=str(target),sha256=sha(target),unique_records=len(pins),max_metric_difference=max_error,gate=result["promising_candidate_beyond_cheap_controls"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--analysis-dir",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args = parser.parse_args()
    def blocked(*args,**kwargs):
        raise RuntimeError("Independent receipt review attempted network access")
    socket.socket.connect=blocked
    socket.socket.connect_ex=blocked
    socket.socket.sendto=blocked
    socket.create_connection=blocked
    socket.getaddrinfo=blocked
    review(args.plan.resolve(),args.plan_sha256,args.run_dir.resolve(),args.analysis_dir.resolve(),args.output_dir.resolve())

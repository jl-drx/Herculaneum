"""Independent source-only synthetic preparation; no real saved-array reads."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

import verify_inference as review


def fixtures():
    checks = []

    def passed(case, **details):
        checks.append(dict(case=case, status="PASS", **details))

    def reject(case, action):
        try:
            action()
        except ValueError as error:
            passed(case, rejected_as_expected=str(error))
        else:
            raise AssertionError("Expected rejection did not occur: " + case)

    arithmetic = review.pinned_module(review.MATH, "synthetic_independent_benchmark")
    blend = review.pinned_module(review.BLEND, "synthetic_independent_blend")
    original = arithmetic.fixtures()
    review.require(original["status"] == "PASS", "Retained independent arithmetic fixtures failed")
    passed("retained_independent_cutoff_ranking_operating_macro_screens", retained_fixture=original)
    review.require(blend.SCORE_TOLERANCE == 2e-6 and blend.expected_grid((384, 384)) ==
                   [[x, y, 128, 128] for y in (0, 64, 128, 192, 256) for x in (0, 64, 128, 192, 256)],
                   "Pinned blend contract differs")
    passed("retained_independent_blend_grid_and_tolerance")
    reject("author_module_import_rejected_before_read",
           lambda: review.pinned_module(dict(path=str(review.BASE / "inference/comparison_common.py"),
                                             size_bytes=1, sha256="0" * 64), "forbidden"))
    reject("checkpoint_read_rejected_before_open", lambda: review.record(review.HERE / "never-read.pth"))
    reject("raw_transport_binary_read_rejected", lambda: review.record(review.HERE / "never-read.bin"))
    reject("duplicate_json_key_rejected", lambda: review.decode_json('{"count":1,"count":2}'))
    reject("json_nan_rejected", lambda: review.decode_json('{"threshold":NaN}'))
    reject("json_infinity_rejected", lambda: review.decode_json('{"threshold":Infinity}'))
    pin = dict(path=str(review.HERE / "dummy.json"), size_bytes=10, byte_length=10, bytes=10, sha256="1" * 64)
    review.require(review.identity(pin)["size_bytes"] == 10, "Consistent file-size aliases differ")
    passed("file_identity_consistent_aliases")
    bad = dict(pin, byte_length=11)
    reject("file_identity_conflicting_aliases_rejected", lambda: review.identity(bad))
    reject("file_identity_boolean_size_rejected", lambda: review.identity(dict(pin, size_bytes=True)))

    tolerance = review.compare(dict(value=.5), dict(value=.5 + 5e-13))
    review.require(tolerance["maximum_absolute_float_error"] <= 1e-12, "Float comparison tolerance differs")
    passed("declared_float_arithmetic_tolerance")
    reject("float_arithmetic_over_tolerance_rejected", lambda: review.compare(.5, .5 + 2e-12))
    reject("boolean_count_rejected", lambda: review.compare(1, True))
    reject("undefined_precision_replacement_rejected", lambda: review.compare(None, 0.))
    reject("missing_metric_field_rejected", lambda: review.compare(dict(tp=1, fp=0), dict(tp=1)))

    yy, xx = np.indices((384, 384))
    labels = ((xx + yy) % 2).astype(np.uint8)
    known = np.ones((384, 384), dtype=np.bool_)
    primary = (yy >= 32) & (yy < 352) & (xx >= 32) & (xx < 352)
    arrays = dict(surface=np.full((17, 384, 384), .125, dtype=np.float32), labels=labels,
                  known_mask=known, surface_mask=known.copy(), primary_mask=primary)
    case = dict(primary_positive=51200, primary_negative=51200)
    support = review.validate_arrays(case, arrays)
    review.require(support["pixels"] == 102400 and support["primary_mask_exact"], "Canonical 32px primary fixture differs")
    passed("independent_32_pixel_primary_support_and_class_counts")
    old_margin = (yy >= 64) & (yy < 320) & (xx >= 64) & (xx < 320)
    reject("wrong_64_pixel_primary_margin_rejected", lambda: review.validate_arrays(case, dict(arrays, primary_mask=old_margin)))
    escaped = arrays["surface_mask"].copy()
    escaped[100, 100] = False
    reject("known_labels_outside_surface_rejected", lambda: review.validate_arrays(case, dict(arrays, surface_mask=escaped)))
    wrong_binary = labels.copy()
    wrong_binary[100, 100] = 2
    reject("nonbinary_labels_rejected", lambda: review.validate_arrays(case, dict(arrays, labels=wrong_binary)))
    reject("uint8_quantized_surface_rejected", lambda: review.validate_arrays(case, dict(arrays, surface=arrays["surface"].astype(np.uint8))))
    reject("fp64_surface_rejected", lambda: review.validate_arrays(case, dict(arrays, surface=arrays["surface"].astype(np.float64))))
    reject("wrong_primary_class_counts_rejected", lambda: review.validate_arrays(dict(case, primary_positive=51201), arrays))
    reject("support_uint8_rejected", lambda: review.validate_arrays(case, dict(arrays, known_mask=known.astype(np.uint8))))

    original_argv = sys.argv
    try:
        sys.argv = ["verify_inference", "--plan", "/no-plan.json", "--plan-sha256", "0" * 64,
                    "--results", "/no-results.json", "--results-sha256", "0" * 64,
                    "--metrics", "/no-metrics.json", "--metrics-sha256", "0" * 64,
                    "--source-freeze-sha256", "0" * 64, "--output", "/no-output.json"]
        reject("unauthorized_completed_review_rejected_before_files_or_arrays", review.main)
    finally:
        sys.argv = original_argv

    fragments = {"frag3": "PHercParis1Fr34", "frag1": "PHercParis2Fr47", "frag2": "PHercParis2Fr143"}
    plan = dict(cases=[dict(id=name, fragment=fragments[name.split("-")[0]],
                           parent_papyrus="Paris1" if name.startswith("frag3") else "Paris2",
                           role="calibration" if name.startswith("frag3") else "evaluation") for name in review.CASES])
    thresholds = {role: dict(retained_fourteenth=(.7558947205543518, .7450202703475952, .7422624826431274)[i],
                             frag3_only=float(np.nextafter(np.float32(.6 + i * .02), np.float32(np.inf))))
                  for i, role in enumerate(review.ROLES)}
    contexts = {}
    truth = labels[primary].astype(np.bool_)
    for j, row in enumerate(plan["cases"]):
        models = {}
        for i, role in enumerate(review.ROLES):
            scores = ((xx % 4) * .2 + (yy % 2) * .03 + i * .03 + j * .015).astype(np.float32)[primary]
            rank = arithmetic.ranking(truth, scores)
            models[role] = dict(auc=rank["auc"], ap=rank["ap"], supplied_positive_prevalence=.5,
                operating={rule: arithmetic.derived_operating(truth, scores, value) for rule, value in thresholds[role].items()},
                scores={"synthetic": True}, evaluated={"synthetic": True}, logits={"synthetic": True})
        contexts[row["id"]] = dict(fragment=row["fragment"], parent_papyrus=row["parent_papyrus"], role=row["role"],
            fixed_primary_mask={"synthetic": True}, pixel_count=102400, positive=51200, negative=51200,
            models=models, candidate_minus_baseline={role: arithmetic.delta(models[role], models["frozen_seed42"])
                                                   for role in review.ROLES[1:]})
    metrics = review.reconstruct_metrics(plan, {"synthetic": "plan"}, {"synthetic": "results"},
        {"synthetic": "barrier"}, thresholds, contexts, arithmetic)
    review.require(len(metrics["per_context"]) == 6 and len(metrics["per_fragment"]) == 3 and
                   set(metrics["thresholds"]) == set(review.ROLES), "Complete synthetic report scope differs")
    for role in review.ROLES:
        expected_auc = math.fsum(contexts[name]["models"][role]["auc"] for name in review.EVALUATION) / 4.
        review.require(abs(metrics["paris2_equal_fragment"][role]["auc"] - expected_auc) <= 1e-12,
                       "Equal-fragment versus equal-context synthetic result differs")
        for rule in review.RULES:
            checks95 = [20 * contexts[name]["models"][role]["operating"][rule]["confusion"]["fp"] <=
                        contexts[name]["negative"] for name in review.EVALUATION]
            review.require(metrics["specificity_transfer_checks"][role][rule]["all_four"] == all(checks95),
                           "Exact all-four screen differs")
    passed("six_context_three_state_two_cutoff_metrics_macros_screens")
    review.require(metrics["per_fragment"]["PHercParis1Fr34"]["models"]["frozen_seed42"]
                   ["operating"]["retained_fourteenth"]["precision"] is None,
                   "Undefined precision did not propagate")
    passed("undefined_precision_preserved_in_every_containing_macro")
    changed = copy.deepcopy(metrics)
    changed["per_context"]["frag1-context1"]["models"]["A_preserve"]["operating"]["frag3_only"]["confusion"]["fp"] += 1
    reject("one_pixel_confusion_change_rejected", lambda: review.compare(metrics, changed))
    changed = copy.deepcopy(metrics)
    changed["specificity_transfer_checks"]["A_preserve"]["frag3_only"]["all_four"] = not changed["specificity_transfer_checks"]["A_preserve"]["frag3_only"]["all_four"]
    reject("altered_transfer_conjunction_rejected", lambda: review.compare(metrics, changed))
    changed = copy.deepcopy(metrics)
    changed["per_fragment"]["PHercParis1Fr34"]["models"]["frozen_seed42"]["operating"]["retained_fourteenth"]["precision"] = 0.
    reject("undefined_macro_precision_fabrication_rejected", lambda: review.compare(metrics, changed))
    review.require("torch" not in sys.modules, "Preparation imported Torch")
    return dict(status="PASS", cases=len(checks), checks=checks, actual_scientific_array_reads=0,
                actual_output_reads=0, model_calls=0, Torch_imports=0, checkpoint_reads=0, network_requests=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review.require(not args.output.exists(), "Synthetic receipt must be fresh")
    value = fixtures()
    value["sources"] = [review.record(Path(__file__)), review.record(review.HERE / "verify_inference.py"), review.BLEND, review.MATH]
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status="PASS", cases=value["cases"], receipt=review.record(args.output)), sort_keys=True))


if __name__ == "__main__":
    main()

"""Source-only context tests; no scientific plan/results/metrics/array reads."""

import argparse
import copy
import json
from pathlib import Path
import sys

import verify_inference as review


def fixtures():
    pin = review.EXECUTION_CONTEXT
    review.require(review.record(pin["path"]) == pin, "Authorized retrospective context pin changed")
    context = review.decode_json(Path(pin["path"]).read_text())
    plan = {"code": {"run_comparison.py": {"path": str(review.BASE / "inference/run_comparison.py")}}}
    entries = [context[k] for k in ("plan", "results", "metrics")]
    checks = []

    def validate(item, selected=entries, selected_plan=plan):
        return review.validate_execution_context(item, *selected, selected_plan)

    def reject(name, mutation):
        value = copy.deepcopy(context)
        mutation(value)
        try:
            validate(value)
        except ValueError as error:
            checks.append({"case": name, "status": "PASS", "rejected_as_expected": str(error)})
        else:
            raise AssertionError("Expected context rejection: " + name)

    review.require(validate(context) is context, "Exact root context was not retained")
    checks.append({"case": "exact_pinned_retrospective_context_and_entries", "status": "PASS"})
    for name, value in (("wrong_cwd", "/workspace"), ("relative_cwd", "reports/sixteenth-experiment"),
                        ("noncanonical_cwd", str(review.BASE) + "/."), ("null_cwd", None),
                        ("nul_cwd", str(review.BASE) + "\x00"), ("newline_cwd", str(review.BASE) + "\n")):
        reject(name, lambda x, value=value: x.update(cwd=value))
    reject("nonretrospective_context", lambda x: x.update(retrospective=False))
    reject("integer_retrospective_flag", lambda x: x.update(retrospective=1))
    reject("wrong_context_status", lambda x: x.update(status="PASS"))
    for role in ("plan", "results", "metrics"):
        for field, value in (("path", "/wrong/entry.json"), ("sha256", "0" * 64), ("size_bytes", 1)):
            reject(role + "_" + field + "_substitution", lambda x, role=role, field=field, value=value:
                   x[role].update({field: value}))
    for index, value, name in (
        (0, "/usr/bin/python", "wrong_python"), (1, "inference/other.py", "wrong_controller_source"),
        (1, "inference/./run_comparison.py", "controller_source_equivalent_spelling"),
        (1, str(review.BASE / "inference/run_comparison.py"), "controller_source_absolute_spelling"),
        (2, "--plan-file", "malformed_plan_flag"), (3, "/wrong/plan.json", "wrong_controller_plan"),
        (3, "inference/final-plan/./comparison-plan.json", "controller_plan_equivalent_spelling"),
        (4, "--output", "malformed_output_flag"), (5, "inference/other", "wrong_controller_output"),
        (5, "inference/./run", "controller_output_equivalent_spelling"),
        (5, "inference/run/../other", "controller_output_traversal"), (5, "", "empty_controller_output"),
        (5, None, "null_controller_output"), (5, "inference/run\x00", "nul_controller_output"),
        (5, "inference/run\n", "newline_controller_output")):
        reject(name, lambda x, index=index, value=value: x["controller_argv"].__setitem__(index, value))
    reject("extra_controller_argument", lambda x: x["controller_argv"].append("--extra"))
    reject("missing_controller_argument", lambda x: x["controller_argv"].pop())
    reject("nonlist_controller_argv", lambda x: x.update(controller_argv="command"))
    reject("changed_environment", lambda x: x.update(environment_overrides={}))
    reject("added_environment", lambda x: x["environment_overrides"].update(EXTRA="1"))
    for field, value, suffix in (("external_timeout_seconds", 601, "wrong"),
                                 ("external_timeout_seconds", 600.0, "float"),
                                 ("external_timeout_seconds", True, "bool"),
                                 ("external_timeout_kill_after_seconds", 6, "wrong"),
                                 ("external_timeout_kill_after_seconds", 5.0, "float"),
                                 ("external_timeout_kill_after_seconds", True, "bool")):
        reject(field + "_" + suffix, lambda x, field=field, value=value: x.update({field: value}))
    changed = copy.deepcopy(plan)
    changed["code"]["run_comparison.py"]["path"] = "/wrong/source.py"
    try:
        validate(context, selected_plan=changed)
    except ValueError as error:
        checks.append({"case": "sealed_plan_controller_source_substitution", "status": "PASS",
                       "rejected_as_expected": str(error)})
    else:
        raise AssertionError("Expected sealed controller source rejection")
    review.require(review.record(pin["path"]) == pin, "Context changed during source-only tests")
    review.require("torch" not in sys.modules, "Source-only test imported Torch")
    return {"status": "PASS", "cases": len(checks), "checks": checks, "authorized_context_source": pin,
            "authorized_context_source_reads": 1, "actual_plan_results_metrics_body_reads": 0,
            "actual_scientific_array_reads": 0, "model_calls": 0, "Torch_imports": 0,
            "checkpoint_reads": 0, "network_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review.require(not args.output.exists(), "Context synthetic receipt must be fresh")
    value = fixtures()
    value["sources"] = [review.record(Path(__file__)), review.record(review.HERE / "verify_inference.py")]
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "cases": value["cases"], "receipt": review.record(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

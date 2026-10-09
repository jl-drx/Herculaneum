"""Render numeric-only figures after root confirms the fixed comparison is complete.

Reads only completed metrics JSON; never opens source images, arrays or models.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

ROLES = ["frozen_seed42", "A_smooth", "B_smooth", "A_preserve", "B_preserve"]
LABELS = ["Frozen\nbaseline", "A\nsmoothing", "B\nsmoothing", "A\npreservation", "B\npreservation"]
COLORS = ["#333333", "#D55E00", "#0072B2", "#CC79A7", "#009E73"]
CASES = ["frag4-context1", "frag4-context2"]
DIRECTIONS = ["forward", "reverse"]
METRICS = ["auc", "ap", "primary_sensitivity", "primary_specificity"]
MARKERS = [("Context 1", "o", -0.15, 34), ("Context 2", "s", 0.15, 34),
           ("Equal-context mean", "D", 0.0, 56)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def valid(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1, "Metric must be finite in [0,1]")
    return float(value)


def valid_threshold(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), "Threshold must be finite numeric")
    return float(value)


def compact(metrics):
    require(metrics["status"] == "PASS", "Completed PASS metrics are required")
    require(metrics["verified_run_count"] == 20, "All twenty fixed runs are required")
    require(metrics["direction_roles"] == {"forward": "predeclared primary", "reverse": "predeclared secondary"},
            "Direction roles differ from the fixed comparison")
    require(set(metrics["per_context"]) == set(DIRECTIONS), "Both fixed directions are required")
    require(set(metrics["state_sha256"]) == set(ROLES), "All five states are required")
    result = {"physical_specimen": "PHercParis1Fr39", "reference_domain": "primary",
              "direction_roles": metrics["direction_roles"], "state_sha256": metrics["state_sha256"],
              "per_context": {}, "equal_context_macro": {},
              "interpretation": "Two contexts from one specimen; descriptive supplied-reference agreement; historical exposure and registration uncertainty remain."}
    for direction in DIRECTIONS:
        contexts = metrics["per_context"][direction]
        require(set(contexts) == set(CASES), "The two fixed contexts are required")
        result["per_context"][direction] = {}
        for case_id in CASES:
            row = contexts[case_id]["domains"]["primary"]
            require(set(row["models"]) == set(ROLES), "All five states must be present in each context")
            require(row["fixed_count"] == row["common_count"] == row["positive"] + row["negative"],
                    "Fixed primary support must not be reduced")
            require(row["positive"] >= 512 and row["negative"] >= 8192, "Primary class gate differs")
            target = {"positive": row["positive"], "negative": row["negative"],
                      "prevalence": valid(row["prevalence"]), "models": {}}
            for role in ROLES:
                model = row["models"][role]
                target["models"][role] = {"auc": valid(model["auc"]), "ap": valid(model["ap"]),
                    "primary_sensitivity": valid(model["operating"]["primary"]["sensitivity"]),
                    "primary_specificity": valid(model["operating"]["primary"]["specificity"])}
                threshold = model["operating"]["primary"].get("threshold")
                if threshold is not None:
                    target["models"][role]["primary_threshold"] = valid_threshold(threshold)
            result["per_context"][direction][case_id] = target
        macro = metrics["equal_context_macro"][direction]["primary"]
        require(set(macro) == set(ROLES), "Macro must include all five states")
        result["equal_context_macro"][direction] = {}
        for role in ROLES:
            result["equal_context_macro"][direction][role] = {}
            for key in METRICS:
                value = valid(macro[role][key])
                expected = sum(result["per_context"][direction][case_id]["models"][role][key]
                               for case_id in CASES) / 2
                require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12), "Equal-context macro does not match its two contexts")
                result["equal_context_macro"][direction][role][key] = value
    return result


def figure(data, keys, titles, title, target, plt, Line2D):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), sharey=True, sharex=False)
    for row_idx, direction in enumerate(DIRECTIONS):
        for col_idx, (key, metric_title) in enumerate(zip(keys, titles)):
            axis = axes[row_idx, col_idx]
            for state_idx, (role, color) in enumerate(zip(ROLES, COLORS)):
                values = [data["per_context"][direction][case_id]["models"][role][key] for case_id in CASES]
                values += [data["equal_context_macro"][direction][role][key]]
                for value, (_, marker, offset, size) in zip(values, MARKERS):
                    axis.scatter(state_idx + offset, value, s=size, marker=marker, color=color,
                                 edgecolor="white", linewidth=0.5, zorder=3)
            axis.set_ylim(0, 1)
            axis.set_xlim(-0.5, len(ROLES) - 0.5)
            axis.set_yticks([i / 5 for i in range(6)])
            tick_labels = [label + "\nmean " + f"{data["equal_context_macro"][direction][role][key]:.3f}"
                           for label, role in zip(LABELS, ROLES)]
            axis.set_xticks(range(len(ROLES)), tick_labels)
            axis.tick_params(axis="x", labelbottom=True, length=0, pad=8, labelsize=10)
            axis.grid(axis="y", color="#D9DEE4", linewidth=0.7, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#B0B8C2")
            direction_title = "Forward · primary direction" if direction == "forward" else "Reverse · secondary direction"
            axis.set_title(f"{direction_title}\n{metric_title}", fontsize=12, loc="left", pad=10)
            if key == "auc":
                axis.axhline(0.5, color="#A9AFB7", linestyle=":", linewidth=1)
            if col_idx == 0:
                axis.set_ylabel("Score", fontsize=11)
    legend = [Line2D([], [], linestyle="none", marker=marker, markersize=7,
                     markerfacecolor="#333333", markeredgecolor="white", label=label)
              for label, marker, _, _ in MARKERS]
    fig.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945), frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=16, x=0.06, ha="left", y=0.985)
    footer = ("Two contexts from one specimen. Primary reference support; equal-context means are descriptive, with no confidence intervals.\n"
              "Historical model exposure and registration uncertainty remain. Directions are predeclared and kept separate.\n"
              "The third line of each state label gives its equal-context mean; axes are 0–1 throughout.")
    fig.text(0.06, 0.025, footer, ha="left", va="bottom", fontsize=9, color="#4A515B")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.185, top=0.82, hspace=0.64, wspace=0.15)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(target.with_suffix("." + suffix), dpi=180, facecolor="white", metadata={"Creator": "Herculaneum numeric comparison visualization"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison-complete", action="store_true",
                        help="Execution owner confirms that the actual fixed comparison and analysis are complete")
    args = parser.parse_args()
    require(args.comparison_complete, "Root confirmation of completion is required before rendering")
    metrics_path = args.metrics.resolve()
    metrics_hash = sha256(metrics_path)
    code_hash = sha256(Path(__file__))
    data = compact(json.loads(metrics_path.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "matplotlib-cache"))
    font_cache = args.output_dir / "font-cache"
    font_cache.mkdir()
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    (args.output_dir / "numeric-input.json").write_text(json.dumps(data, indent=2) + "\n")
    with (args.output_dir / "numeric-results.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "direction_role", "reference_domain", "context_or_macro", "state"] + METRICS)
        for direction in DIRECTIONS:
            for context in CASES + ["equal_context_macro"]:
                rows = (data["equal_context_macro"][direction] if context == "equal_context_macro"
                        else data["per_context"][direction][context]["models"])
                for role in ROLES:
                    writer.writerow([direction, data["direction_roles"][direction], "primary", context, role] +
                                    [rows[role][key] for key in METRICS])
    figure(data, ["auc", "ap"], ["ROC AUC", "Average precision"],
           "PHercParis1Fr39: supplied-reference ranking agreement", args.output_dir / "ranking-agreement", plt, Line2D)
    figure(data, ["primary_sensitivity", "primary_specificity"],
           ["Sensitivity at each state's fixed primary threshold", "Specificity at each state's fixed primary threshold"],
           "PHercParis1Fr39: operating points at retained thresholds", args.output_dir / "fixed-thresholds", plt, Line2D)
    require(sha256(metrics_path) == metrics_hash, "Metrics changed during rendering")
    require(sha256(Path(__file__)) == code_hash, "Plot code changed during rendering")
    outputs = [{"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(args.output_dir.iterdir()) if path.is_file()]
    require(sum(item["size_bytes"] for item in outputs) < 8 * 1024 ** 2, "Numeric figure artifact cap exceeded")
    receipt = {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "input": {"path": str(metrics_path), "sha256": metrics_hash, "size_bytes": metrics_path.stat().st_size},
               "code": {"path": str(Path(__file__).resolve()), "sha256": code_hash},
               "matplotlib_version": matplotlib.__version__, "outputs": outputs,
               "directions_separate": True, "all_five_states": ROLES,
               "reference_domain": "primary", "equal_context_macro_not_pixel_pooled": True,
               "axes_limits_all": [0, 1], "numeric_only_no_images_arrays_models": True,
               "no_fitted_thresholds_or_orientation_selection": True,
               "input_and_code_unchanged": True,
               "layout_revision": "v2; separate header legend and mean values in state labels",
               "supersedes_figures_in": str(args.output_dir.parent / "results")}
    (args.output_dir / "OUTPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "receipt": str(args.output_dir / "OUTPUT_RECEIPT.json"), "artifact_files": len(outputs)}))


if __name__ == "__main__":
    main()

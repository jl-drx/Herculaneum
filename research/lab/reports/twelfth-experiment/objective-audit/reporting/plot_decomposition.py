#!/usr/bin/env python3
"""Plot saved objective-audit summaries; never read models or source arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED_AUDIT = "f6116a851a418d2982b025f04b595af919eca037dd1f90001eb3be580e217a80"
WINDOWS = ["1-8", "9-16", "17-24", "25-32"]


def record(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.audit.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_AUDIT:
        raise ValueError("Saved audit hash differs from the root-executed PASS receipt")
    audit = json.loads(source)
    if audit["status"] != "PASS":
        raise ValueError("Only a completed PASS audit may be plotted")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/herculaneum-objective-audit-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1), sharey=True)
    colors = {"D": "#1b8075", "L": "#c87330", "base": "#315caa", "full": "#943e75"}
    for ax, role, title in zip(axes, ["A_preserve", "B_preserve"],
                               ["A: manual + preservation", "B: manual + pseudo + preservation"]):
        means = [audit["arms"][role]["windows"][w]["equal_step_means"] for w in WINDOWS]
        x = np.arange(len(WINDOWS))
        ax.bar(x - .17, [m["weighted_KL"] for m in means], width=.32,
               color=colors["D"], label="Weighted KL penalty (D)")
        ax.bar(x + .17, [m["weighted_linear_drift"] for m in means], width=.32,
               color=colors["L"], label="Signed first-order term (L)")
        ax.plot(x, [m["base_delta"] for m in means], "o-", color=colors["base"],
                linewidth=1.5, label="Smoothed base difference: D + L")
        ax.plot(x, [m["full_delta"] for m in means], "^-", color=colors["full"],
                linewidth=1.5, label="Optimized total difference: 2D + L")
        ax.axhline(0, color="#333333", linewidth=.9)
        ax.set_xticks(x, WINDOWS)
        ax.set_xlabel("Fixed training-step window")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-.066, .109)
        ax.grid(axis="y", alpha=.20)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean loss difference from matched-input initializer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(.5, .065), frameon=False)
    fig.suptitle("Smoothed loss rises in the saved preupdate snapshots", fontsize=14, y=.97)
    fig.text(.5, .018,
             "Equal-step window means; changing models and batches. Not final-model or parameter-step evidence.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.085, right=.98, bottom=.29, top=.83, wspace=.12)
    paths = [args.output_dir / "objective-decomposition.png", args.output_dir / "objective-decomposition.svg"]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    if args.audit.read_bytes() != source:
        raise ValueError("Saved audit changed during reporting")
    receipt = {"status": "saved_output_only", "audit": record(args.audit),
               "plot_script": record(__file__), "figures": [record(p) for p in paths],
               "matplotlib_version": matplotlib.__version__, "windows": WINDOWS,
               "model_calls": 0, "source_array_reads": 0, "network_calls": 0,
               "input_bytes_unchanged_before_after": True}
    target = args.output_dir / "figure-receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": record(target), "figures": receipt["figures"]}))


if __name__ == "__main__":
    main()

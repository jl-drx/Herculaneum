#!/usr/bin/env python3
"""Plot saved first-step probe JSON; no models, checkpoints or source arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED = {"A": "234ee87243e25ee2f78158399b1bff7e3909757bdb9956fd6459dc8d9823a5ce",
            "B": "1ecbfb91d97c0fdc42a50241c2582a707acf7d55fc5514922f8c099f42d14410"}


def record(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    saved = {}
    for arm in ["A", "B"]:
        path = args.probe_run / arm / "result.json"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != EXPECTED[arm]:
            raise ValueError("Saved root probe output hash differs")
        result = json.loads(data)
        if result["status"] != "PASS" or result["objective"] != arm:
            raise ValueError("Expected a completed PASS result for each fixed objective")
        if [t["learning_rate"] for t in result["trials"]] != [1e-6, 1e-5, 1e-4]:
            raise ValueError("Unexpected probe step-size schedule")
        saved[arm] = (path, data, result)
    if saved["A"][2]["input_target_sha256"] != saved["B"][2]["input_target_sha256"]:
        raise ValueError("The probe inputs and targets differ between objectives")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/herculaneum-objective-audit-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/herculaneum-objective-audit-fontcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.9), sharey=True)
    for ax, arm, title in zip(axes, ["A", "B"],
                               ["A: manual + preservation", "B: manual + pseudo + preservation"]):
        trials = saved[arm][2]["trials"]
        for key, label, color, marker in [
                ("base_delta", "Smoothed base loss change", "#315caa", "o"),
                ("full_delta", "Optimized total loss change", "#943e75", "^")]:
            x = [t["learning_rate"] for t in trials]
            y = [t["stable_change"]["weighted"][key] for t in trials]
            ax.scatter(x, y, color=color, marker=marker, s=60, label=label, zorder=3)
        ax.axhline(0, color="#333333", linewidth=.9)
        ax.set_xscale("log")
        ax.set_xticks([1e-6, 1e-5, 1e-4], ["1e-6", "1e-5", "1e-4"])
        ax.minorticks_off()
        ax.set_xlim(6e-7, 1.7e-4)
        ax.set_ylim(-.032, .163)
        ax.set_xlabel("Predeclared AdamW step size (log scale)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.20)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Immediate same-batch loss change after one step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(.5, .085), frameon=False)
    fig.suptitle("First-step probe: base and optimized loss can move differently", fontsize=13, y=.96)
    fig.text(.5, .027,
             "Same first batch; each trial resets initializer and optimizer. No optimum or generalized learning-rate claim.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.81, wspace=.12)
    paths = [args.output_dir / "first-step-loss-change.png", args.output_dir / "first-step-loss-change.svg"]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    for path, data, _ in saved.values():
        if path.read_bytes() != data:
            raise ValueError("Saved probe changed during reporting")
    receipt = {"status": "saved_output_only", "inputs": [record(s[0]) for s in saved.values()],
               "plot_script": record(__file__), "figures": [record(p) for p in paths],
               "matplotlib_version": matplotlib.__version__, "curves_interpolated": False,
               "model_calls": 0, "source_array_reads": 0, "network_calls": 0,
               "input_bytes_unchanged_before_after": True}
    target = args.output_dir / "figure-receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": record(target), "figures": receipt["figures"]}))


if __name__ == "__main__":
    main()

"""Exhaustive tiny tie fixtures; no scientific input files or model imports."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from independent_metrics import compute


def brute(labels, scores):
    pairs = [(float(a > b) + 0.5 * float(a == b))
             for a in scores[labels == 1] for b in scores[labels == 0]]
    previous = 0
    result = 0.0
    positive = int(labels.sum())
    for cutoff in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= cutoff
        tp = int((predicted & (labels == 1)).sum())
        result += (tp - previous) / positive * tp / int(predicted.sum())
        previous = tp
    return sum(pairs) / len(pairs), result


def main():
    checked = 0
    for raw_labels in itertools.product([0, 1], repeat=4):
        if not 0 < sum(raw_labels) < 4:
            continue
        labels = np.asarray(raw_labels, dtype=np.uint8)
        for raw_scores in itertools.product([0.0, 0.5, 1.0], repeat=4):
            scores = np.asarray(raw_scores, dtype=np.float32)
            value = compute(labels, scores, {"equality": 0.5})
            auc, ap = brute(labels, scores.astype(np.float64))
            assert abs(value["auc"] - auc) < 1e-14
            assert abs(value["ap"] - ap) < 1e-14
            pred = scores.astype(np.float64) >= 0.5
            assert value["operating"]["equality"]["tp"] == int((pred & (labels == 1)).sum())
            assert value["operating"]["equality"]["fp"] == int((pred & (labels == 0)).sum())
            checked += 1
    value = compute(np.asarray([1, 0]), np.asarray([0.5, 0.5], dtype=np.float32),
                    {"just_above": np.nextafter(0.5, np.inf)})
    assert value["auc"] == value["ap"] == 0.5
    assert value["operating"]["just_above"]["tp"] == 0
    assert value["operating"]["just_above"]["fp"] == 0
    value = compute(np.asarray([1, 0]), np.asarray([0.0, -0.0]), {"zero": 0.0})
    assert value["auc"] == value["ap"] == 0.5
    for labels, scores in [(np.asarray([0, 1]), np.asarray([0.0, np.nan])),
                           (np.asarray([0, 0]), np.asarray([0.0, 1.0]))]:
        try:
            compute(labels, scores, {})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric domain was accepted")
    root = Path(__file__).resolve().parent
    receipt = {"status": "PASS", "exhaustive_four_pixel_tie_cases": checked,
               "additional_boundary_and_rejection_cases": 4,
               "network_calls": 0, "model_calls": 0, "scientific_arrays_read": 0,
               "code": [{"path": str(root / name),
                         "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                        for name in ["independent_metrics.py", "check_metric_fixtures.py"]]}
    output = root / "metric-fixture-receipt.json"
    if output.exists():
        raise FileExistsError("Fixture receipt already exists; no silent rerun")
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

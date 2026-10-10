"""Literal, synthetic-only fixtures for descriptive endpoints, edges and tied ranks."""
import argparse
import math
from pathlib import Path
import numpy as np
from regional_common import write_json
from regional_stats import plane_stats, rank_metrics, support_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plane = np.array([[0, 1], [3, 65535]], dtype=np.uint16)
    masks = dict(all=np.ones((2, 2), bool), empty=np.zeros((2, 2), bool),
                 singleton=np.array([[1, 0], [0, 0]], bool))
    result = plane_stats(plane, support_geometry(masks), 65535)
    all_stats = result["all"]
    expected = dict(count=4, mean=65539 / 4, q10=0.3, q50=2, q90=45875.4,
                    endpoint_min_count=1, endpoint_max_count=1, endpoint_min_fraction=0.25,
                    endpoint_max_fraction=0.25, horizontal_mean_absolute_difference=32766.5,
                    vertical_mean_absolute_difference=32768.5, horizontal_edge_count=2, vertical_edge_count=2)
    checks = []
    for key, value in expected.items():
        assert math.isclose(all_stats[key], value, rel_tol=1e-12, abs_tol=1e-9), key
        checks.append("all-" + key)
    expected_std = math.sqrt(sum((v - 65539 / 4)**2 for v in (0, 1, 3, 65535)) / 4)
    assert abs(all_stats["std"] - expected_std) < 1e-10
    checks.append("population-std")
    for key in ("mean", "std", "q10", "q50", "q90", "endpoint_min_fraction", "endpoint_max_fraction", "horizontal_mean_absolute_difference", "vertical_mean_absolute_difference"):
        assert result["empty"][key] is None
    assert result["empty"]["count"] == 0
    assert result["singleton"]["count"] == 1 and result["singleton"]["std"] == 0
    assert result["singleton"]["horizontal_edge_count"] == result["singleton"]["vertical_edge_count"] == 0
    checks.extend(["empty-support-null", "single-pixel-support-no-edges"])
    tied = rank_metrics([0, 1, 1, 2], [0, 1, 0, 1])
    assert tied["auroc"] == 0.875 and abs(tied["average_precision"] - 5/6) < 1e-15
    checks.append("tied-ranks-auroc-ap")
    assert rank_metrics([], [])["average_precision"] is None
    assert rank_metrics([1, 2], [0, 0])["average_precision"] is None
    assert rank_metrics([1, 2], [1, 1])["auroc"] is None
    assert rank_metrics([1, 1], [1, 0])["auroc"] == 0.5
    assert rank_metrics([1, 1], [1, 0])["average_precision"] == 0.5
    checks.extend(["empty-ranks-null", "no-positives-ap-null", "single-class-auroc-null", "all-tied-ranks"])
    write_json(args.output_dir / "stats-fixtures.json", dict(status="PASS", synthetic_only=True, checks=checks,
               count=len(checks), actual_input_arrays_decoded=0))
    print(f"PASS: {len(checks)} synthetic statistic fixtures")


if __name__ == "__main__":
    main()

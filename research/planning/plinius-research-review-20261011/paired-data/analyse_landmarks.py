#!/usr/bin/env python3
"""Replay published affine landmark residuals; does not fit a transformation.

Input is a small extract of the retained PHerc0139 public catalog. All
coordinates are interpreted literally in the supplied XYZ order. Residuals
are target-volume voxel vectors, not surface-normal errors or held-out TRE.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def analyse(source):
    rows = []
    for item in source["transforms"]:
        matrix = np.asarray(item["matrix_3x4"], dtype=np.float64)
        before = np.asarray(item["from_landmarks_xyz"], dtype=np.float64)
        after = np.asarray(item["to_landmarks_xyz"], dtype=np.float64)
        assert matrix.shape == (3, 4)
        assert before.shape == after.shape and before.shape[1] == 3
        predicted = before @ matrix[:, :3].T + matrix[:, 3]
        vectors = predicted - after
        distances = np.linalg.norm(vectors, axis=1)
        scalar = [
            [sum(float(matrix[i, j]) * float(point[j]) for j in range(3))
             + float(matrix[i, 3]) - float(target[i]) for i in range(3)]
            for point, target in zip(before, after)
        ]
        scalar_distances = [math.sqrt(sum(v * v for v in vec)) for vec in scalar]
        delta = float(np.max(np.abs(distances - np.asarray(scalar_distances))))
        assert delta < 1e-10, delta
        spacing = float(item["target_spacing_um"])
        summary = {
            "min": float(distances.min()),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p95_linear": float(np.quantile(distances, .95)),
            "max": float(distances.max()),
            "rms": float(np.sqrt(np.mean(distances ** 2))),
        }
        rows.append({
            "from_volume": item["from_volume"], "to_volume": item["to_volume"],
            "landmark_count": len(before), "target_spacing_um": spacing,
            "target_voxel_residuals": summary,
            "residual_um": {k: v * spacing for k, v in summary.items()},
            "scalar_numpy_max_distance_difference": delta,
            "points": [
                {"index": i, "residual_xyz_target_voxels": v.tolist(),
                 "norm_target_voxels": float(d), "norm_um": float(d * spacing)}
                for i, (v, d) in enumerate(zip(vectors, distances))
            ],
        })
    return {
        "status": "PUBLISHED_AFFINE_REPLAY_ONLY",
        "formula": "target_pred_xyz = source_xyz @ M[:,:3].T + M[:,3]",
        "new_transform_fit": False, "independent_test_landmarks": False,
        "coordinate_convention": "literal XYZ sequence in catalog",
        "limitations": [
            "Correspondences accompany the published fit and are not independently held out.",
            "No CT, sheet identity, local deformation or normal-direction measurement is made.",
            "Micrometres use the catalog's target-volume isotropic spacing.",
        ],
        "transforms": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(__file__).with_name("LANDMARK_INPUT.json"))
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("LANDMARK_RESULTS.json"))
    args = p.parse_args()
    payload = args.input.read_bytes()
    result = analyse(json.loads(payload))
    result["input_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for item in result["transforms"]:
        print(item["from_volume"], "->", item["to_volume"], item["landmark_count"],
              item["target_voxel_residuals"])

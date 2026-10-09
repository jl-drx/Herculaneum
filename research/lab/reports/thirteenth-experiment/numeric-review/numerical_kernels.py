"""Independent NumPy kernels; no driver, Torch, model or audit imports."""
import math
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mean(values):
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "Finite nonempty mean required")
    return math.fsum(float(x) for x in array.ravel()) / array.size


def support_terms(student, teacher, hard):
    """Stable log-probability BCE and teacher-to-student KL on one support."""
    z, t, y = [np.asarray(x, dtype=np.float64) for x in (student, teacher, hard)]
    require(z.shape == t.shape == y.shape and z.size > 0, "Aligned nonempty support required")
    require(np.isfinite(z).all() and np.isfinite(t).all(), "Nonfinite logits")
    require(np.isin(y, [0, 1]).all(), "Binary hard labels required")
    lp, lpc = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    lq, lqc = -np.logaddexp(0, -t), -np.logaddexp(0, t)
    q, qc = np.exp(lq), np.exp(lqc)
    soft = 0.25 + 0.5 * y
    bce = -soft * lp - (1 - soft) * lpc
    initial = -soft * lq - (1 - soft) * lqc
    kl = q * (lq - lp) + qc * (lqc - lpc)
    linear = (q - soft) * (z - t)
    u, ut, d, drift = [mean(x) for x in (bce, initial, kl, linear)]
    return {"count": int(y.size), "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()), "student_smoothed_BCE": u,
            "initializer_smoothed_BCE": ut,
            "student_hard_BCE": mean(-y * lp - (1 - y) * lpc),
            "initializer_hard_BCE": mean(-y * lq - (1 - y) * lqc),
            "probability_drift": mean(np.abs(np.exp(lp) - q)),
            "KL": d, "linear_drift": drift, "base_delta": u - ut,
            "full_delta": u + d - ut, "student_full": u + d,
            "initializer_full": ut,
            "base_decomposition_residual": u - ut - d - drift,
            "full_decomposition_residual": u + d - ut - 2 * d - drift}


def weighted_objective(manual, pseudo, coefficient):
    require(coefficient in (0., .25), "Fixed pseudo coefficient required")
    base = manual["student_smoothed_BCE"] + coefficient * pseudo["student_smoothed_BCE"]
    initial = manual["initializer_smoothed_BCE"] + coefficient * pseudo["initializer_smoothed_BCE"]
    kl = manual["KL"] + coefficient * pseudo["KL"]
    return {"base": base, "full": base + kl, "initializer_base": initial,
            "initializer_full": initial, "KL": kl,
            "base_delta_vs_initializer": base - initial,
            "full_delta_vs_initializer": base + kl - initial}


def immediate_change(pre, post):
    """Actual same-batch after-minus-before losses, not paired initializer deltas."""
    require(pre["initializer_base"] == post["initializer_base"], "Same teacher required")
    return {key + "_post_minus_pre": post[key] - pre[key] for key in ("base", "full", "KL")}


def fixed_risk(batch_objectives):
    """Each scheduled two-item batch has equal weight irrespective of support count."""
    require(len(batch_objectives) == 32, "Exactly 32 frozen batches required")
    keys = set(batch_objectives[0])
    require(all(set(b) == keys for b in batch_objectives), "Fixed risk fields differ")
    result = {key: mean([b[key] for b in batch_objectives]) for key in sorted(keys)}
    # Frozen gate arithmetic: difference of the two equal-batch means.
    result["base_delta_vs_initializer"] = result["base"] - result["initializer_base"]
    result["full_delta_vs_initializer"] = result["full"] - result["initializer_full"]
    return result


def loss_record(student, teacher, arrays, coefficient):
    require(student.shape == teacher.shape == (2, 1, 128, 128), "Batch logit shape changed")
    terms = {}
    for name in ("manual", "pseudo"):
        mask, hard = arrays[name + "_mask"], arrays[name + "_hard_target"]
        require(mask.shape == hard.shape == student.shape, "Support shape changed")
        terms[name] = support_terms(student[mask], teacher[mask], hard[mask])
    manual, pseudo = terms["manual"], terms["pseudo"]
    balanced = []
    for region in range(2):
        mask = arrays["manual_mask"][region]
        hard = arrays["manual_hard_target"][region]
        for label in (0, 1):
            selected = mask & (hard == label)
            balanced.append(support_terms(student[region][selected], teacher[region][selected], hard[selected])
                            ["student_smoothed_BCE"])
    objective = weighted_objective(manual, pseudo, coefficient)
    return {"loss_U": manual["student_smoothed_BCE"], "loss_P": pseudo["student_smoothed_BCE"],
            "loss_R": mean(balanced), "loss_D_M": manual["KL"], "loss_D_P": pseudo["KL"],
            "base": objective["base"], "full": objective["full"],
            "hard_base": manual["student_hard_BCE"] + coefficient * pseudo["student_hard_BCE"],
            "probability_drift_manual": manual["probability_drift"],
            "probability_drift_pseudo": pseudo["probability_drift"]}


def retention_gate(final_risk):
    """Both final equal-batch paired deltas must meet the frozen inclusive bound."""
    return (final_risk["base_delta_vs_initializer"] <= -1e-4 and
            final_risk["full_delta_vs_initializer"] <= -1e-4)


def d4(array, code):
    """Destination-to-source indexing independent of rotation/flip helper calls."""
    require(type(code) is int and 0 <= code < 8 and array.shape[-2] == array.shape[-1],
            "Square D4 input and integer code required")
    n = array.shape[-1]
    yy, xx = np.indices((n, n))
    u = xx if code < 4 else n - 1 - xx
    pairs = [(yy, u), (u, n - 1 - yy), (n - 1 - yy, n - 1 - u), (n - 1 - u, yy)]
    sy, sx = pairs[code % 4]
    return np.ascontiguousarray(array[..., sy, sx])


def fixture_checks():
    a = np.arange(1, 10).reshape(3, 3)
    gold = [[[1,2,3],[4,5,6],[7,8,9]], [[3,6,9],[2,5,8],[1,4,7]],
            [[9,8,7],[6,5,4],[3,2,1]], [[7,4,1],[8,5,2],[9,6,3]],
            [[3,2,1],[6,5,4],[9,8,7]], [[9,6,3],[8,5,2],[7,4,1]],
            [[7,8,9],[4,5,6],[1,2,3]], [[1,4,7],[2,5,8],[3,6,9]]]
    for code in range(8):
        require(np.array_equal(d4(a, code), gold[code]), "D4 golden raster differs")
    z, t, y = np.array([-7.,-1.,0.,3.,6.]), np.array([-2.,2.,0.,-1.,4.]), np.array([0.,1.,0.,1.,0.])
    r = support_terms(z, t, y)
    require(abs(r["base_decomposition_residual"]) < 1e-12 and abs(r["full_decomposition_residual"]) < 1e-12,
            "Loss decomposition differs")
    eq = support_terms(t, t, y)
    require(eq["KL"] == eq["base_delta"] == eq["linear_drift"] == 0, "Teacher identity differs")
    require(all(math.isfinite(v) for v in support_terms(np.array([-1000.,1000.]),
            np.array([1000.,-1000.]), np.array([0.,1.])).values()), "Extreme logits fail")
    zero_pos, zero_neg = np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32)
    require(np.array_equal(zero_pos, zero_neg) and not np.array_equal(zero_pos.view(np.uint32),
            zero_neg.view(np.uint32)), "Signed-zero distinction fails")
    for coefficient in (0., .25):
        pre = weighted_objective(r, eq, coefficient)
        post = weighted_objective(eq, eq, coefficient)
        changes = immediate_change(pre, post)
        require(abs(changes["base_post_minus_pre"] + r["base_delta"]) < 1e-12 and
                abs(changes["full_post_minus_pre"] + r["full_delta"]) < 1e-12,
                "Immediate change differs from direct before/after subtraction")
    yy, xx = np.indices((128, 128))
    manual = np.broadcast_to(xx < 64, (2, 1, 128, 128)).copy()
    pseudo = ~manual
    hard = np.broadcast_to((yy % 2).astype(np.float32), (2, 1, 128, 128)).copy()
    arrays = {"manual_mask": manual, "pseudo_mask": pseudo,
              "manual_hard_target": hard, "pseudo_hard_target": hard}
    zero = np.zeros((2, 1, 128, 128), dtype=np.float32)
    for coefficient in (0., .25):
        measured = loss_record(zero, zero, arrays, coefficient)
        require(measured["loss_U"] == measured["loss_P"] == measured["loss_R"] == math.log(2),
                "Full-batch zero-logit BCE fixture fails")
        require(measured["loss_D_M"] == measured["loss_D_P"] == measured["probability_drift_manual"] ==
                measured["probability_drift_pseudo"] == 0., "Zero-logit KL/drift fixture fails")
        require(measured["base"] == measured["full"] == measured["hard_base"] == (1 + coefficient) * math.log(2),
                "Separate-support weighted zero-logit fixture fails")
    unequal = [dict(base=0. if i < 16 else 2., full=0. if i < 16 else 2.,
                    initializer_base=1., initializer_full=1., KL=0.,
                    base_delta_vs_initializer=-1. if i < 16 else 1.,
                    full_delta_vs_initializer=-1. if i < 16 else 1.) for i in range(32)]
    risk = fixed_risk(unequal)
    require(risk["base"] == 1. and risk["base_delta_vs_initializer"] == 0., "Equal-batch risk fails")
    # Hypothetical support sizes 1 versus 100 would give a different pixel-pooled result.
    pixel_pooled = (16 * 1 * 0. + 16 * 100 * 2.) / (16 * 1 + 16 * 100)
    require(pixel_pooled != risk["base"], "Equal-batch fixture cannot distinguish pooling")
    for base, full, expected in [(-1e-4,-1e-4,True),
            (np.nextafter(-1e-4,math.inf),-1e-4,False),
            (-1e-4,np.nextafter(-1e-4,math.inf),False), (-.1,0.,False), (0.,-.1,False)]:
        require(retention_gate(dict(base_delta_vs_initializer=base,full_delta_vs_initializer=full)) == expected,
                "Inclusive joint final-risk gate fails")
    return {"status":"PASS", "golden_D4_cases":8, "decomposition_cases":1,
            "teacher_identity_cases":1, "extreme_logit_cases":1, "signed_zero_cases":1,
            "immediate_change_cases":2, "equal_batch_vs_pixel_pooled_cases":1,
            "joint_retention_gate_cases":5, "complete_loss_record_cases":2,
            "actual_retained_array_reads":0}

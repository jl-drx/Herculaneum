"""Arithmetic and support fixtures; contains no real teacher reads."""
import json
import unittest
from pathlib import Path

import numpy as np

import analysis as a


class OfflineArithmetic(unittest.TestCase):
    def test_asymmetric_linear_gradient_nonzero_origin(self):
        yy, xx = np.mgrid[:64, :64]
        image = (xx + 2 * yy).astype(np.uint8)
        crop, origin = image[13:48, 9:48], (9, 13)
        uv = np.array([[[4.25, 5.5], [5.0, 6.0], [8.125, 7.875]]])
        expected = (4*uv[..., 0]+1.5) + 2*(4*uv[..., 1]+1.5)
        for op in a.OPERATORS:
            value, valid = a.sample_teacher(crop, origin, uv, np.ones((1, 3), bool), op)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)

    def test_mean_preserves_fractional_uint8_block_values(self):
        image = np.zeros((16, 16), np.uint8)
        image[4, 4] = 1
        uv = np.array([[[1.0, 1.0]]])
        value, valid = a.sample_teacher(image, (0, 0), uv, np.ones((1, 1), bool), "mean16_bilinear")
        self.assertTrue(valid.item())
        self.assertEqual(value.item(), 1/16)
        point, _ = a.sample_teacher(image, (0, 0), uv, np.ones((1, 1), bool), "point_bilinear")
        self.assertEqual(point.item(), 0)

    def test_exhaustive_scalar_operator_check(self):
        rng = np.random.default_rng(2432)
        image = rng.integers(0, 256, (80, 90), dtype=np.uint8)
        origin = (7, 11)
        crop = image[11:73, 7:81]
        uv = rng.uniform([3, 4], [17, 16], (4, 5, 2))
        for op in a.OPERATORS:
            actual, valid = a.sample_teacher(crop, origin, uv, np.ones((4, 5), bool), op)
            expected = np.zeros((4, 5))
            for row in range(4):
                for col in range(5):
                    u, v = uv[row, col]
                    if op == "point_bilinear":
                        u, v = 4*u+1.5, 4*v+1.5
                    x, y = int(np.floor(u)), int(np.floor(v))
                    for sy in range(2):
                        for sx in range(2):
                            weight = ((u-x) if sx else (1-u+x)) * ((v-y) if sy else (1-v+y))
                            if op == "point_bilinear":
                                val = int(image[y+sy, x+sx])
                            else:
                                val = sum(int(image[4*(y+sy)+iy, 4*(x+sx)+ix])
                                          for iy in range(4) for ix in range(4))/16
                            expected[row, col] += val*weight
            self.assertTrue(valid.all())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_partial_blocks_and_geometry_do_not_fill_with_zero(self):
        crop = np.full((25, 25), 100, dtype=np.uint8)
        uv = np.array([[[1.2, 2.0], [2.0, 2.0], [3.0, 3.0]]])
        geom = np.array([[True, True, False]])
        values, valid = a.sample_teacher(crop, (5, 7), uv, geom, "mean16_bilinear")
        np.testing.assert_array_equal(valid, [[False, True, False]])
        self.assertTrue(np.isnan(values[0, 0]) and np.isnan(values[0, 2]))
        self.assertEqual(values[0, 1], 100)

    def test_footprints_include_all_bilinear_corners_and_blocks(self):
        uv = np.array([[[1.2, 4.8], [3.9, 2.1]]])
        support = np.ones((1, 2), bool)
        self.assertEqual(a.source_bounds(uv, support, "mean16_bilinear"), [4, 8, 20, 24])
        self.assertEqual(a.source_bounds(uv, support, "point_bilinear"), [6, 9, 19, 22])

    def test_shift_sign_and_dz_reselect_uv(self):
        zz, yy, xx = np.mgrid[:20, :32, :32]
        q = np.stack((xx + 2*zz, yy + 3*zz, zz), axis=-1).astype(float)
        valid = np.ones((20, 32, 32), bool)
        bounds = [2300, 1788, 2302, 1790]
        baseline, _ = a.shifted_queries(q, valid, bounds, 14, (0, -1, 0))
        shifted, _ = a.shifted_queries(q, valid, bounds, 14, (1, 1, -1))
        np.testing.assert_array_equal(shifted-baseline, np.broadcast_to([1, 5], (2, 2, 2)))
        with self.assertRaises(ValueError):
            a.shifted_queries(q, valid, bounds, 25, (0, 0, 0))

    def test_naive_centre_conversion(self):
        uv, valid = a.naive_queries([2416, 1824, 2418, 1826])
        self.assertTrue(valid.all())
        self.assertAlmostEqual(4*uv[0, 0, 0]+1.5, (2416.5)*31040/7980-.5)
        self.assertAlmostEqual(4*uv[0, 0, 1]+1.5, (1824.5)*24880/6400-.5)

    def test_threshold_inclusive_no_rounding_and_zeros_unknown(self):
        base = np.array([[0., 20., 48., 100.]])
        alternative = np.array([[20., 0., 47., 120.]])
        valid = np.ones_like(base, bool)
        m = a.compare_targets(base, valid, alternative, valid)
        self.assertEqual(m["teacher_nonzero_common_pixels"], 2)
        self.assertEqual(m["teacher_nonzero_union_pixels"], 4)
        self.assertEqual(m["teacher_nonzero_support_xor_fraction_of_union"], .5)
        self.assertEqual(m["metric_common_pseudo_supervision_pixels"], 2)
        self.assertEqual(m["threshold_xor_fraction_of_positive_union"], .5)
        self.assertEqual(m["mean_absolute_difference_all_uint8"], 10.5)
        self.assertEqual(m["lost_positive_pixels"], 1)
        self.assertEqual(m["gained_positive_pixels"], 0)

    def test_no_positive_evidence_is_null_not_perfect(self):
        base = np.full((12, 12), 1.)
        valid = np.ones_like(base, bool)
        m = a.compare_targets(base, valid, base, valid)
        self.assertIsNone(m["threshold_iou"])
        self.assertIsNone(m["mean_absolute_difference_positive_union_uint8"])
        self.assertEqual(m["positive_evidence"], "no-positive-evidence")
        self.assertFalse(m["adequate_positive_union_and_negative_common_counts"])

    def test_zero_unknowns_do_not_satisfy_negative_adequacy(self):
        values = np.zeros((2, 20, 20))
        values[:, :10, :] = 100
        result = a.finite_set_summary(values, np.ones_like(values, bool))
        self.assertEqual(result["negative_common_pixels"], 0)
        self.assertEqual(result["positive_union_pixels"], 200)
        self.assertFalse(result["adequate_positive_union_and_negative_common_counts"])

    def test_global_common_support_and_native_eligibility(self):
        values = np.array([[[100., 1., 50., 60.]], [[80., 2., 0., 10.]], [[90., 3., 45., 5.]]])
        eligible = np.array([[True, True, True, False]])
        result = a.finite_set_summary(values, np.ones_like(values, bool), eligible)
        self.assertEqual(result["complete_across_finite_set_pixels"], 3)
        self.assertEqual(result["common_pseudo_supervision_pixels"], 2)
        self.assertEqual(result["teacher_nonzero_union_pixels"], 3)
        self.assertEqual(result["teacher_nonzero_common_pixels"], 2)
        self.assertEqual(result["threshold_variable_envelope_pixels"], 0)
        self.assertEqual(result["score_range_mean_uint8"], 11.)

    def test_variable_envelope_not_mean_pairwise_disagreement(self):
        values = np.array([[[50., 50., 1.]], [[40., 50., 1.]], [[50., 40., 1.]]])
        result = a.finite_set_summary(values, np.ones_like(values, bool))
        self.assertEqual(result["threshold_variable_envelope_pixels"], 2)
        self.assertEqual(result["threshold_variable_envelope_fraction_of_positive_union"], 1.)
        self.assertTrue(result["operational_material_envelope"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineArithmetic)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name("SYNTHETIC_CHECKS.json").write_text(json.dumps(dict(
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        status="PASS" if result.wasSuccessful() else "FAIL",
        analysis_sha256=a.sha256(Path(a.__file__)), tests_sha256=a.sha256(Path(__file__)),
        limitation="Arithmetic and support fixtures only; no real teacher target accessed."), indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)

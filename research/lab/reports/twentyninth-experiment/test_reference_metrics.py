"""Fixed synthetic examples and independent scalar oracles. No model/data reads."""
import itertools
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import zarr
import reference_metrics as m
import upstream_extract as up

FIXTURES = Path(__file__).resolve().parent / 'run-v1/fixtures'
OBSERVATIONS = {}
GROUPS = dict(Frag1='Paris2', Frag2='Paris2', Frag3='Paris1', Frag4='Paris1',
              Frag5='PHerc1667', Frag6='PHerc51')


def oracle(scores, labels):
    positives = [float(s) for s, y in zip(scores, labels) if y == 1]
    negatives = [float(s) for s, y in zip(scores, labels) if y == 0]
    auc = sum(1 if p > n else .5 if p == n else 0 for p in positives for n in negatives)
    auc /= len(positives) * len(negatives)
    ap = 0.
    for threshold in sorted(set(map(float, scores)), reverse=True):
        above = [y for s, y in zip(scores, labels) if s >= threshold]
        new_pos = sum(y == 1 and s == threshold for s, y in zip(scores, labels))
        ap += (new_pos / len(positives)) * (sum(y == 1 for y in above) / len(above))
    return auc, ap


def ranking(scores, labels, valid=None, selected=None):
    p, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=np.int8)
    return m.rank_metrics(p, y, selected=np.ones(p.shape, bool) if selected is None else selected,
                          prediction_valid=np.ones(p.shape, bool) if valid is None else valid)


def depth(p, **overrides):
    p = np.asarray(p, dtype=float)
    args = dict(voxel_valid=np.ones(p.shape, bool), surface_layers=np.full(p.shape[1], 4),
                surface_valid=np.ones(p.shape[1], bool), selected=np.ones(p.shape[1], bool), z_offset=0)
    args.update(overrides)
    return m.depth_diagnostics(p, **args)


class AdapterTests(unittest.TestCase):
    def test_exhaustive_tied_rankings_against_scalar_oracles(self):
        count = 0
        for labels in itertools.product((0, 1), repeat=4):
            if len(set(labels)) != 2:
                continue
            for scores in itertools.product((0., .5, 1.), repeat=4):
                result = ranking(scores, labels)
                auc, ap = oracle(scores, labels)
                self.assertAlmostEqual(result['auc'], auc, delta=1e-12)
                self.assertAlmostEqual(result['ap'], ap, delta=1e-12)
                count += 1
        OBSERVATIONS['exhaustive_rank_cases'] = count

    def test_random_ties_and_permutation_invariance(self):
        rng = np.random.default_rng(29)
        for _ in range(60):
            scores = rng.integers(0, 8, 12) / 7.
            labels = np.array([1, 0] * 6)
            order = rng.permutation(12)
            expected = oracle(scores, labels)
            actual = ranking(scores[order], labels[order])
            np.testing.assert_allclose([actual['auc'], actual['ap']], expected, atol=1e-12, rtol=0)

    def test_close_float_order_not_quantized(self):
        self.assertEqual((ranking([.5002, .5001], [1, 0])['auc'], ranking([.5002, .5001], [1, 0])['ap']), (1., 1.))

    def test_unknown_missing_and_valid_zero_remain_distinct(self):
        r = ranking([.8, .1, 1., .9, 0., .7], [1, 0, 2, 1, 0, 1],
                    valid=np.array([1, 1, 1, 0, 1, 1], bool),
                    selected=np.array([1, 1, 1, 1, 1, 0], bool))
        self.assertEqual(r['counts'], dict(selected=5, unknown=1, known=4, missing_prediction=1,
                                          scored_positive=1, scored_negative=2))
        self.assertEqual((r['auc'], r['ap']), (1., 1.))

    def test_degenerate_classes_are_explicit(self):
        for labels in ([], [0, 0], [1, 1], [2, 2]):
            r = ranking([0.] * len(labels), labels)
            self.assertIsNone(r['auc']); self.assertIsNone(r['ap'])
            self.assertEqual(r['status'], 'undefined_requires_both_classes')
            json.dumps(r, allow_nan=False)

    def test_invalid_payload_only_allowed_outside_validity(self):
        for bad in (np.nan, np.inf, -np.inf, -.01, 1.01):
            with self.assertRaises(ValueError):
                ranking([.7, bad], [1, 0])
            r = ranking([.7, .2, bad], [1, 0, 0], valid=np.array([1, 1, 0], bool))
            self.assertEqual(r['auc'], 1.)

    def test_encoding_and_endianness(self):
        for dtype in ('uint8', '<u2', '>u2'):
            hi = np.iinfo(np.dtype(dtype)).max
            a = np.array([0, hi], dtype=dtype)
            encoding = 'uint8' if hi == 255 else 'uint16'
            values, valid = m.decode_probabilities(a, np.ones(2, bool), encoding=encoding)
            np.testing.assert_array_equal(values, [0., 1.]); self.assertTrue(valid.all())
        values, valid = m.decode_probabilities(np.array([0, 1, 2], np.uint8), np.ones(3, bool), encoding='label012')
        np.testing.assert_array_equal(values, [0., 1., 0.]); np.testing.assert_array_equal(valid, [1, 1, 0])
        scaled, _ = m.decode_probabilities(np.array([0, 1], np.uint8), np.ones(2, bool), encoding='uint8')
        self.assertEqual(scaled[1], 1 / 255.)

    def test_strict_shapes_types_and_bounds(self):
        for action in (
            lambda: m.decode_probabilities(np.array([1], np.uint8), np.ones(1, bool), encoding='float01'),
            lambda: m.decode_probabilities(np.array([1.]), np.ones(1, bool), encoding='guess'),
            lambda: m.decode_probabilities(np.array([3], np.uint8), np.ones(1, bool), encoding='label012'),
            lambda: m.decode_probabilities(np.array([1.]), np.ones(1, int), encoding='float01'),
            lambda: ranking([.1], [3]),
            lambda: ranking([.1], [0, 1]),
            lambda: m.decode_probabilities(np.zeros(m.MAX_ELEMENTS + 1), np.zeros(m.MAX_ELEMENTS + 1, bool), encoding='float01'),
            lambda: depth(np.zeros((2, 2_000_001))),
            lambda: depth(np.zeros((0, 1))),
            lambda: depth(np.zeros((10, 1)), surface_layers=np.array([4.5])),
            lambda: depth(np.zeros((10, 1)), z_offset=.5),
            lambda: depth(np.zeros((10, 1)), band=(2, 1)),
        ):
            with self.assertRaises(ValueError): action()

    def test_explicit_row_coverage(self):
        np.testing.assert_array_equal(m.row_coverage(4, [1, 3]), [0, 1, 1, 0])
        for rows in (None, [-1, 3], [0, 5], [2, 1], [0., 2], [False, 2]):
            with self.assertRaises(ValueError): m.row_coverage(4, rows)

    def test_depth_exact_peak_and_centroid(self):
        p = np.zeros((10, 2)); p[4, 0] = .5; p[6, 1] = 1.
        r = depth(p)
        np.testing.assert_array_equal(r['peak'], [0., 2.])
        np.testing.assert_array_equal(r['centroid'], [0., 2.])
        np.testing.assert_allclose(r['band_mean'], [.05, .1], atol=1e-12, rtol=0)
        self.assertTrue(r['profile_valid'].all() and r['band_valid'].all())

    def test_partial_band_rejected(self):
        r = depth(np.full((6, 1), .5), surface_layers=np.array([0]))
        self.assertFalse(r['band_valid'][0]); self.assertTrue(np.isnan(r['band_mean'][0]))
        self.assertEqual(r['counts']['incomplete_surface_band'], 1)

    def test_band_and_profile_coverage_differ(self):
        p = np.full((12, 2), .25); valid = np.ones(p.shape, bool)
        valid[11, 0] = False; valid[3, 1] = False
        r = depth(p, voxel_valid=valid)
        np.testing.assert_array_equal(r['profile_valid'], [0, 0])
        np.testing.assert_array_equal(r['band_valid'], [1, 0])
        self.assertEqual(r['band_mean'][0], .25)

    def test_zero_flat_and_unknown_surface(self):
        p = np.zeros((10, 3)); p[:, 1:] = .5
        r = depth(p, surface_valid=np.array([1, 1, 0], bool))
        self.assertEqual(r['counts']['zero_mass'], 1)
        self.assertEqual(r['counts']['tied_peak'], 1)
        self.assertEqual(r['counts']['surface_unknown'], 1)
        np.testing.assert_array_equal(r['centroid_valid'], [0, 1, 0])
        np.testing.assert_array_equal(r['peak_valid'], [0, 0, 0])
        self.assertEqual(r['centroid'][1], .5)
        self.assertEqual(r['maximum'][0], 0.)

    def test_depth_z_offset_translation_invariant(self):
        p = np.zeros((10, 1)); p[5] = 1
        a = depth(p); b = depth(p, z_offset=100, surface_layers=np.array([104]))
        for key in ('maximum', 'band_mean', 'centroid', 'peak'):
            np.testing.assert_array_equal(a[key], b[key])

    def test_surface_ratio_is_descriptive_and_total(self):
        def ratio(p, y):
            return m.surface_response_ratio(np.array(p, float), np.array(y, np.int8),
                    selected=np.ones(len(p), bool), prediction_valid=np.ones(len(p), bool))
        self.assertEqual(ratio([.25, .5], [1, 0])['ratio'], 2.)
        self.assertEqual(ratio([.5, 0], [1, 0])['ratio'], 0.)
        self.assertIsNone(ratio([0, .5], [1, 0])['ratio'])
        self.assertIsNone(ratio([.5, .9], [1, 2])['ratio'])
        self.assertIsNone(ratio([1e-320, 1], [1, 0])['ratio'])
        json.dumps(ratio([1e-320, 1], [1, 0]), allow_nan=False)

    def test_parent_group_exclusion(self):
        folds = m.parent_group_folds(GROUPS)
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertFalse(set(GROUPS[f] for f in fold['train']) & set(GROUPS[f] for f in fold['test']))
        shared = [f for f in GROUPS if any(GROUPS[g] == GROUPS[f] for g in GROUPS if g != f)]
        self.assertEqual(shared, ['Frag1', 'Frag2', 'Frag3', 'Frag4'])
        OBSERVATIONS['conservative_group_folds'] = folds
        OBSERVATIONS['ordinary_LOFO_folds_with_same_group_in_training'] = shared


class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = up.scorer_namespace()

    def hist_metrics(self, p, y):
        h = self.ns['tile_hist']; p, y = np.asarray(p), np.asarray(y)
        pos = h(p[y == 1], np.zeros((y == 1).sum(), int), 1)[0]
        neg = h(p[y == 0], np.zeros((y == 0).sum(), int), 1)[0]
        return float(self.ns['auc_from_hist'](pos, neg)), float(self.ns['ap_from_hist'](pos, neg))

    def test_uint8_histograms_retain_order_and_ties(self):
        rng = np.random.default_rng(29)
        for _ in range(100):
            p = rng.integers(0, 256, 12).astype(np.uint8) / 255.
            y = np.array([0, 1] * 6)
            np.testing.assert_allclose(self.hist_metrics(p, y), oracle(p, y), atol=1e-12, rtol=0)
        OBSERVATIONS['upstream_uint8_oracle_cases'] = 100

    def test_float_quantization_merges_distinct_scores(self):
        self.assertEqual(self.hist_metrics([.5002, .5001], [1, 0]), (.5, .5))
        self.assertEqual(oracle([.5002, .5001], [1, 0]), (1., 1.))
        OBSERVATIONS['float_quantization'] = dict(upstream_auc=.5, upstream_ap=.5, exact_auc=1., exact_ap=1.)

    def test_no_positive_AP_zero(self):
        auc, ap = self.hist_metrics([.2, .3], [0, 0])
        self.assertTrue(np.isnan(auc)); self.assertEqual(ap, 0.)
        OBSERVATIONS['upstream_no_positive_AP'] = dict(ap=ap, auc='NaN')

    def test_volume_ignores_coverage_rows(self):
        path = FIXTURES / 'partial.zarr'
        a = zarr.open_array(str(path), mode='w-', shape=(2, 4, 2), chunks=(2, 1, 2), dtype='u1', fill_value=0)
        a[:, 1:3, :] = 255; a.attrs.update(dict(rows=[1, 3], z_offset=12))
        v = self.ns['Volume'](path)
        self.assertEqual(v.z_offset, 12)
        self.assertEqual(v.rows(0, 4).shape, (2, 4, 2))
        self.assertEqual(float(v.rows(0, 1).sum()), 0.)
        self.assertFalse(hasattr(v, 'rows_coverage'))
        self.assertEqual(int(m.row_coverage(4, a.attrs['rows']).sum()), 2)
        OBSERVATIONS['coverage_rows'] = dict(declared=[1, 3], upstream_returns_all_rows=4, explicit_covered_rows=2)

    def test_label_ignore_becomes_negative(self):
        path = FIXTURES / 'labels.npy'; np.save(path, np.array([[[0, 1, 2]]], np.uint8))
        actual = self.ns['Volume'](path, label_value=1).rows(0, 1)
        np.testing.assert_array_equal(actual, [[[0, 1, 0]]])
        OBSERVATIONS['upstream_ignore2'] = 'converted to0 without validity mask'

    def test_opposite_endian_uint16_not_scaled(self):
        dtype = '>u2' if sys.byteorder == 'little' else '<u2'
        path = FIXTURES / 'endian.npy'; np.save(path, np.array([[[65535]]], dtype=dtype))
        value = float(self.ns['Volume'](path).rows(0, 1)[0, 0, 0])
        self.assertEqual(value, 65535.)
        OBSERVATIONS['upstream_opposite_endian_uint16_max'] = value

    def test_partial_band_still_gets_mean(self):
        ns = dict(np=np, rel=np.arange(6)[:, None, None], b_lo=-4, b_hi=5,
                  v=np.full((6, 1, 1), .5), band_mean=np.full((1, 1), np.nan), a=0, b=1)
        up.score_assignments(ns, ('inb', 'nb', 'band_mean'))
        self.assertEqual(int(ns['nb'][0, 0]), 6); self.assertEqual(float(ns['band_mean'][0, 0]), .5)
        OBSERVATIONS['partial_band'] = dict(available_layers=6, required_layers=10, upstream_mean=.5, adapter_status='excluded')

    def test_tile_ids_include_empty_bin(self):
        ns = dict(np=np, H=512, W=512, TILE=256)
        up.score_assignments(ns, ('tile_id',))
        ids = ns['tile_id']
        self.assertEqual(int(ids.max()) + 1, 5); self.assertEqual(len(np.unique(ids)), 4)
        OBSERVATIONS['tile_index_gap'] = dict(max_id_plus_one=5, occupied_tiles=4)

    def test_small_ink_sample_mislabels_clear(self):
        ns = dict(np=np, N_SAMPLE=5, rng=np.random.default_rng(0),
                  ink_core=np.array([1, 1, 0, 0, 0, 0, 0], bool),
                  clear=np.array([0, 0, 1, 1, 1, 1, 1], bool))
        up.sample_assignments(ns)
        mislabels = int((ns['is_ink'] & ns['clear'][ns['idx']]).sum())
        self.assertEqual(mislabels, 3)
        OBSERVATIONS['small_sample_mislabels'] = dict(requested_ink=5, actual_ink=2, clear_mislabeled_ink=3)

    def test_total_sample_too_small_raises(self):
        ns = dict(np=np, N_SAMPLE=5, rng=np.random.default_rng(0),
                  ink_core=np.array([1, 1, 0, 0], bool), clear=np.array([0, 0, 1, 1], bool))
        with self.assertRaisesRegex(ValueError, 'negative dimensions'):
            up.sample_assignments(ns)
        OBSERVATIONS['small_total_sample'] = 'ValueError: negative dimensions are not allowed'

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import unittest

from herculaneum.jobs import estimate_pilot_cost, validate_gpu_job


class PilotBudgetTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "schema_version": 1, "kind": "gpu-pilot", "ready_for_gpu": True, "scientific_blockers": [],
            "identities": {"git_commit": "a" * 40, "image_digest": "example/image@sha256:" + "b" * 64,
                           "model_sha256": ["c" * 64], "input_manifest_sha256": "d" * 64,
                           "experiment_config_sha256": "e" * 64},
            "budget": {"currency": "GBP", "cap_gbp": "10", "prior_spend_gbp": "1", "rate_gbp_per_hour": "12",
                       "billing_quantum_seconds": 60, "billable_seconds": 601,
                       "other_costs_gbp": "0.50", "tax_fraction": "0.20", "margin_gbp": "1",
                       "quote_expires_at": "2026-10-09T00:00:00+00:00"},
            "shutdown": {"independent_verified": True, "compute_timeout_seconds": 540,
                         "hard_timeout_seconds": 600},
        }
        self.now = datetime(2026, 10, 8, tzinfo=timezone.utc)

    def test_aggregate_cost_includes_rounding_tax_prior_attempt_and_margin(self):
        self.assertEqual(estimate_pilot_cost(self.spec["budget"]), Decimal("5.24"))
        self.assertTrue(validate_gpu_job(self.spec, self.now)["ready"])

    def test_retry_cannot_reset_or_exceed_cap(self):
        self.spec["budget"]["prior_spend_gbp"] = "9"
        self.assertFalse(validate_gpu_job(self.spec, self.now)["ready"])
        self.spec["budget"]["cap_gbp"] = "20"
        with self.assertRaises(ValueError):
            estimate_pilot_cost(self.spec["budget"])

    def test_reject_invalid_unbounded_cost_values(self):
        for value in [None, True, "NaN", "Infinity", "-1"]:
            with self.subTest(value=value):
                budget = deepcopy(self.spec["budget"])
                budget["rate_gbp_per_hour"] = value
                with self.assertRaises(ValueError):
                    estimate_pilot_cost(budget)

    def test_unknown_identity_or_shutdown_blocks_launch_readiness(self):
        self.spec["shutdown"]["independent_verified"] = False
        self.spec["identities"]["image_digest"] = "example/image:latest"
        report = validate_gpu_job(self.spec, self.now)
        self.assertFalse(report["ready"])
        self.assertFalse(report["launch_performed"])

    def test_expired_quote_and_no_upload_window_block_readiness(self):
        self.spec["budget"]["quote_expires_at"] = "2026-10-07T00:00:00+00:00"
        self.spec["shutdown"]["compute_timeout_seconds"] = 600
        self.assertFalse(validate_gpu_job(self.spec, self.now)["ready"])

    def test_scientific_plan_cannot_pass_as_launch_job(self):
        self.assertFalse(validate_gpu_job({"kind": "scientific-plan", "ready_for_gpu": False}, self.now)["ready"])

    def test_unsupported_schema_or_non_gbp_currency_rejected(self):
        self.spec["schema_version"] = 999
        self.spec["budget"]["currency"] = "USD"
        self.assertFalse(validate_gpu_job(self.spec, self.now)["ready"])

    def test_malformed_sections_return_errors(self):
        for name in ["identities", "budget", "shutdown"]:
            with self.subTest(name=name):
                spec = deepcopy(self.spec)
                spec[name] = ["invalid"]
                self.assertFalse(validate_gpu_job(spec, self.now)["ready"])

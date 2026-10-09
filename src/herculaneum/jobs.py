"""Validate a proposed GPU pilot; this module never launches cloud resources."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import re

APPROVED_PILOT_CAP_GBP = Decimal("10.00")


def amount(value, name):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{name} must be a finite nonnegative number") from None
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def estimate_pilot_cost(budget):
    """Conservative aggregate cost, with provider billing rounded upward."""
    if not isinstance(budget, dict) or budget.get("currency") != "GBP":
        raise ValueError("The approved pilot budget must be explicitly denominated in GBP")
    cap = amount(budget.get("cap_gbp"), "cap_gbp")
    if cap > APPROVED_PILOT_CAP_GBP:
        raise ValueError("The first pilot's aggregate cap cannot exceed £10")
    quantum = amount(budget.get("billing_quantum_seconds"), "billing_quantum_seconds")
    duration = amount(budget.get("billable_seconds"), "billable_seconds")
    rate = amount(budget.get("rate_gbp_per_hour"), "rate_gbp_per_hour")
    if quantum <= 0 or duration <= 0 or rate <= 0:
        raise ValueError("Billing quantum, billable duration and hourly rate must be positive")
    billed = (duration / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    other = amount(budget.get("other_costs_gbp"), "other_costs_gbp")
    tax = amount(budget.get("tax_fraction"), "tax_fraction")
    margin = amount(budget.get("margin_gbp"), "margin_gbp")
    prior = amount(budget.get("prior_spend_gbp"), "prior_spend_gbp")
    if margin <= 0:
        raise ValueError("A positive currency/cleanup contingency margin is required")
    total = prior + (rate * billed / Decimal(3600) + other) * (1 + tax) + margin
    if total > cap:
        raise ValueError(f"Conservative aggregate estimate £{total:.2f} exceeds the approved cap")
    return total


def validate_gpu_job(spec, now=None):
    """Return readiness errors and an estimate; declarations require external verification."""
    errors = []
    if not isinstance(spec, dict):
        spec = {}
        errors.append("A job specification must be a JSON object")
    if type(spec.get("schema_version")) is not int or spec.get("schema_version") != 1:
        errors.append("Unsupported or missing job schema_version")
    if spec.get("kind") != "gpu-pilot":
        errors.append("This is a scientific plan, not an executable GPU pilot specification")
    if spec.get("ready_for_gpu") is not True or not isinstance(spec.get("scientific_blockers"), list) or spec.get("scientific_blockers"):
        errors.append("Scientific prerequisites are unresolved")
    sections = {}
    for name in ["identities", "shutdown", "budget"]:
        section = spec.get(name)
        if not isinstance(section, dict):
            errors.append(f"{name} must be a JSON object")
            section = {}
        sections[name] = section
    identities = sections["identities"]
    for name in ["git_commit", "input_manifest_sha256", "experiment_config_sha256"]:
        length = 40 if name == "git_commit" else 64
        if not isinstance(identities.get(name), str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", identities[name]):
            errors.append(f"A pinned {name} is required")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", str(identities.get("image_digest", ""))):
        errors.append("A container image digest is required")
    models = identities.get("model_sha256")
    if not isinstance(models, list) or not models or any(not re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in models):
        errors.append("Verified model checkpoint hashes are required")
    shutdown = sections["shutdown"]
    if shutdown.get("independent_verified") is not True:
        errors.append("An independent provider shutdown deadline must be verified")
    budget = sections["budget"]
    estimate = None
    try:
        estimate = estimate_pilot_cost(budget)
        hard = amount(shutdown.get("hard_timeout_seconds"), "hard_timeout_seconds")
        compute = amount(shutdown.get("compute_timeout_seconds"), "compute_timeout_seconds")
        billable = amount(budget.get("billable_seconds"), "billable_seconds")
        if not 0 < compute < hard <= billable:
            errors.append("Compute must stop before the hard deadline within budgeted billable time")
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    try:
        expiry = datetime.fromisoformat(budget.get("quote_expires_at", ""))
        if expiry.tzinfo is None or expiry <= (now or datetime.now(timezone.utc)):
            errors.append("A current timezone-aware provider/GBP cost quote is required")
    except (ValueError, TypeError):
        errors.append("A current timezone-aware provider/GBP cost quote is required")
    return {"ready": not errors, "errors": errors, "estimated_aggregate_gbp": str(estimate) if estimate is not None else None,
            "launch_performed": False}

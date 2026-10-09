"""Offline post-run tensor receipts; import has no model or checkpoint side effects.

Call verify_checkpoints(plan, workers) once in a fresh review process after all
four workers complete. No worker, benchmark metric, forward or optimizer code
is imported. One unchanged CPU factory supplies canonical parameter names;
checkpoint payloads are then inspected and released one at a time.
"""
from __future__ import annotations

from collections.abc import Mapping
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

LAB = Path("/workspace/herculaneum-lab")
AUDIT = LAB / "reports/sixth-experiment/model-contract/audit-v2/model-contract.json"
AUDIT_SHA = "6cbd873f6c41a01414351424d6d3e487b15f934597b838858e965faaab3714b5"
AUDIT_BYTES = 935610
HELPER = LAB / "reports/sixth-experiment/model-contract/model_contract_v2.py"
HELPER_SHA = "f391fde6495b2abdb68a77b6af22766fd3b8cdb2c2c9b597abf5c33cc2b693fd"
REFERENCE = LAB / "run_ink_reference.py"
REFERENCE_SHA = "16bc6e9fb080eb5c2fcc1f20975481881e513b36b7533c1987bdb422340fea9f"
VENDOR = LAB / "vendor/villa-ink-baseline"
VENDOR_SHA = "2165b5def9d0dede136dde8202993be800b121cb2feb13cf98e98cd0629d75df"
ARMS = ("frozen_seed42", "A", "B", "C")
MAX_FILE_BYTES = 200 * 1024**2
MAX_EXPANDED_ZIP_BYTES = 200 * 1024**2
MAX_ZIP_ENTRIES = 8192
MAX_UNIQUE_STATE_BYTES = 160 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record(path):
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), "Expected a regular retained file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}


def _pin(item, receipts):
    expected_size = item.get("size_bytes", item.get("byte_length"))
    _require(isinstance(expected_size, int) and not isinstance(expected_size, bool), "Invalid file byte count")
    _require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "Invalid file SHA")
    actual = _record(item["path"])
    _require(actual["size_bytes"] == expected_size and actual["sha256"] == item["sha256"],
             "Retained file hash/size differs: " + Path(item["path"]).name)
    old = receipts.get(actual["path"])
    _require(old is None or old == actual, "Conflicting retained receipts")
    receipts[actual["path"]] = actual
    return Path(actual["path"])


def _semantic_hash(state):
    """Independent byte-for-byte implementation of experiment_common.state_hash."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        frame = json.dumps([key, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode()
        digest.update(frame + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _partition(state):
    groups = {}
    for key in sorted(state):
        groups.setdefault(state[key].untyped_storage()._cdata, []).append(key)
    return sorted(groups.values(), key=lambda group: group[0])


def _schema(state):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(),
                      bytes=value.numel() * value.element_size(), stride=list(value.stride()),
                      storage_offset=value.storage_offset(), storage_bytes=value.untyped_storage().nbytes())
            for key, value in sorted(state.items())}


def _factory_schema(plan, audited, receipts, torch):
    helper_path = _pin(plan["helper"], receipts)
    _require(helper_path == HELPER and plan["helper"]["sha256"] == HELPER_SHA,
             "Unchanged frozen contract helper required")
    _pin({"path": str(REFERENCE), "size_bytes": REFERENCE.stat().st_size, "sha256": REFERENCE_SHA}, receipts)
    manifest_path = _pin({"path": str(VENDOR / "source-manifest.json"),
                          "size_bytes": (VENDOR / "source-manifest.json").stat().st_size,
                          "sha256": VENDOR_SHA}, receipts)
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["source_commit"] == "e0bbb8b40a2db58b1d71864f286eb85717e59e64", "Factory revision differs")
    for item in manifest["files"]:
        path = (VENDOR / item["path"]).resolve(strict=True)
        _require(path.is_relative_to(VENDOR), "Vendor source escaped its frozen root")
        _pin({"path": str(path), "size_bytes": item["size_bytes"], "sha256": item["sha256"]}, receipts)
    license_path = VENDOR / "UPSTREAM-LICENSE.txt"
    _pin({"path": str(license_path), "size_bytes": license_path.stat().st_size,
          "sha256": manifest["license_sha256"]}, receipts)
    spec = importlib.util.spec_from_file_location("eighth_checkpoint_schema_contract", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runtime = helper.load_runtime()
    _require(runtime["source_manifest_sha256"] == VENDOR_SHA, "Focused source manifest differs")
    authored = audited["embedded_config"]
    helper.validate_authored_config(authored)
    config = runtime["InkConfig"].from_mapping(authored)
    _require(config.model.pretrained_backbone is None and config.model.input_pad_depth_to is None,
             "Factory must not acquire pretrained weights or pad depth")
    with torch.no_grad():
        factory = runtime["make_model"](config)
    state = factory.state_dict()
    _require(all(value.device.type == "cpu" for value in state.values()), "CPU factory required")
    schema, groups = _schema(state), _partition(state)
    parameters = dict(factory.named_parameters())
    parameter_names = tuple(parameters)
    buffer_names = tuple(name for name, _ in factory.named_buffers())
    parameter_elements = sum(value.numel() for value in parameters.values())
    _require(parameter_elements == audited["parameter_count"] == 34546498, "Factory parameter count differs")
    core = [dict(key=key, shape=value["shape"], dtype=value["dtype"], numel=value["numel"], bytes=value["bytes"])
            for key, value in schema.items()]
    _require(core == audited["tensor_schema"], "Factory tensor schema differs from frozen audit")
    grouped = [item["keys"] for item in audited["storage_contract"]["alias_groups"]]
    aliased = {key for group in grouped for key in group}
    audit_partition = sorted(grouped + [[key] for key in schema if key not in aliased], key=lambda group: group[0])
    _require(groups == audit_partition, "Factory alias partition differs from frozen audit")
    _require(all(name in schema for name in parameter_names + buffer_names), "Factory name missing in state")
    del parameters, state, factory, config, runtime, helper
    gc.collect()
    return schema, groups, parameter_names, buffer_names, parameter_elements


def _bounded_load(item, receipts, torch):
    size = item.get("size_bytes", item.get("byte_length"))
    _require(isinstance(size, int) and 0 < size <= MAX_FILE_BYTES, "Serialized checkpoint exceeds fixed file cap")
    path = _pin(item, receipts)
    _require(zipfile.is_zipfile(path), "PyTorch ZIP archive required; no legacy or pickle fallback")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        _require(0 < len(entries) <= MAX_ZIP_ENTRIES, "Checkpoint ZIP-entry bound exceeded")
        _require(len({entry.filename for entry in entries}) == len(entries), "Duplicate ZIP entry")
        _require(all(not entry.flag_bits & 1 and entry.compress_type == zipfile.ZIP_STORED for entry in entries),
                 "Only unencrypted uncompressed PyTorch ZIP members allowed")
        expanded = sum(entry.file_size for entry in entries)
        _require(0 < expanded <= MAX_EXPANDED_ZIP_BYTES, "Expanded checkpoint ZIP bound exceeded")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _pin(item, receipts)
    return payload, {"zip_entry_count": len(entries), "expanded_zip_bytes": expanded,
                     "serialized_bytes": size, "weights_only": True, "map_location": "cpu",
                     "sha256_verified_before_and_after_load": True}


def _check_state(state, schema, expected_groups, audited, torch):
    _require(isinstance(state, Mapping) and set(state) == set(schema), "Checkpoint tensor keys differ")
    _require(all(isinstance(key, str) for key in state), "Nonstring tensor key")
    for key, value in state.items():
        expected = schema[key]
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided and value.device.type == "cpu"
                 and not value.is_quantized and not value.is_complex(), "Ordinary dense CPU tensor required: " + key)
        _require(list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"]
                 and value.numel() == expected["numel"] and value.numel() * value.element_size() == expected["bytes"],
                 "Tensor shape/dtype/size differs: " + key)
        _require(list(value.stride()) == expected["stride"] and value.storage_offset() == expected["storage_offset"]
                 and value.untyped_storage().nbytes() == expected["storage_bytes"], "Tensor view differs: " + key)
        _require(bool(torch.isfinite(value).all()), "Nonfinite tensor: " + key)
    groups = _partition(state)
    _require(groups == expected_groups, "Checkpoint storage alias partition differs from factory")
    unique_bytes = sum(state[group[0]].untyped_storage().nbytes() for group in groups)
    named_bytes = sum(value.numel() * value.element_size() for value in state.values())
    _require(unique_bytes <= MAX_UNIQUE_STATE_BYTES, "Unique checkpoint storage exceeds cap")
    aliases = [group for group in groups if len(group) > 1]
    expected = audited["storage_contract"]
    _require(len(state) == expected["tensor_count"] == 508 and len(groups) == expected["unique_storage_count"] == 264
             and len(aliases) == expected["alias_group_count"] == 244
             and unique_bytes == expected["unique_storage_bytes"] == 138185992
             and named_bytes == expected["named_tensor_bytes"] == 272701704,
             "Fixed retained tensor/storage counts differ")
    for group in aliases:
        first = state[group[0]].detach().contiguous().reshape(-1).view(torch.uint8)
        for key in group[1:]:
            _require(torch.equal(first, state[key].detach().contiguous().reshape(-1).view(torch.uint8)),
                     "Aliased values differ bitwise")
    return dict(tensor_count=len(state), unique_storage_count=len(groups), alias_group_count=len(aliases),
                named_tensor_bytes=named_bytes, unique_storage_bytes=unique_bytes,
                unique_storage_cap_bytes=MAX_UNIQUE_STATE_BYTES, exact_keys_shapes_dtypes=True,
                all_tensors_finite=True, factory_strides_offsets_and_storage_sizes_match=True,
                factory_alias_partition_matches=True, alias_values_bitwise_equal=True,
                strict_schema_compatibility=True, strict_model_load_performed=False)


def _initializer_state(payload, audited):
    _require(isinstance(payload, Mapping) and isinstance(payload.get("step"), int)
             and not isinstance(payload["step"], bool) and payload["step"] == 74999,
             "Initializer container/step differs")
    for name in ("ema_model", "state_dict", "model_state_dict", "model"):
        selected = payload.get(name)
        if isinstance(selected, Mapping):
            _require(name == audited["weight_container"] == "model" and bool(selected), "Initializer selected container differs")
            _require(all(isinstance(key, str) for key in selected), "Invalid initializer keys")
            if all(key.startswith("module.") for key in selected):
                selected = {key[len("module."):]: value for key, value in selected.items()}
            return selected
    raise ValueError("Initializer has no model state")


def verify_checkpoints(plan, workers):
    """Validate completed worker tensors offline; returns JSON-serializable proof.

    workers maps frozen_seed42/A/B/C to their completed result.json objects.
    The caller owns those result-file receipts and the final plan receipt.
    This helper never invokes forward/backward, inference, training or network.
    """
    _require(plan["kind"] == "matched-continuation-v1" and plan["ready_for_execution"] is True,
             "Final ready continuation plan required")
    _require(set(workers) == set(ARMS) and all(workers[arm]["status"] == "PASS" for arm in ARMS),
             "All four workers must be complete before checkpoint review")
    receipts = {}
    common_plan = workers[ARMS[0]]["plan"]
    plan_path = _pin(common_plan, receipts)
    _require(json.loads(plan_path.read_text()) == plan and all(workers[arm]["plan"] == common_plan for arm in ARMS),
             "Workers used a different frozen plan")
    source_contract = _pin(plan["code"]["experiment_common.py"], receipts)
    _require(source_contract == LAB / "reports/eighth-experiment/implementation/experiment_common.py",
             "Unexpected semantic-hash source contract")
    audit_path = _pin({"path": str(AUDIT), "size_bytes": AUDIT_BYTES, "sha256": AUDIT_SHA}, receipts)
    audit = json.loads(audit_path.read_text())
    _require(audit["status"] == "passed" and audit["tensor_schema_identical_across_four_arms"], "Retained model audit failed")
    audited = audit["models"]["seed42"]
    initializer = plan["initializer"]
    _require(initializer["label"] == "seed42" and initializer["sha256"] == audited["sha256"]
             and initializer["size_bytes"] == audited["size_bytes"], "Initializer differs from frozen audit")
    for arm in ARMS:
        _require(workers[arm]["arm"] == arm and workers[arm]["initializer"] == initializer, "Worker initializer role differs")
    _require(workers["frozen_seed42"]["training"] is None, "Frozen control was adapted")
    import torch
    torch.set_num_threads(4)
    _require(torch.get_num_threads() == 4, "Four Torch threads required")
    schema, groups, parameters, buffers, parameter_elements = _factory_schema(plan, audited, receipts, torch)
    payload, initializer_archive = _bounded_load(initializer, receipts, torch)
    state = _initializer_state(payload, audited)
    initial_storage = _check_state(state, schema, groups, audited, torch)
    initial_hash = _semantic_hash(state)
    initial_parameter_hash = _semantic_hash({name: state[name] for name in parameters})
    initial_tensor_hashes = {name: _tensor_hash(state[name]) for name in parameters}
    _require(all(workers[arm]["initial_state_sha256"] == initial_hash for arm in ARMS),
             "Actual initializer semantic hash differs between workers")
    del state, payload
    gc.collect()
    final_results = {}
    for arm in ("A", "B", "C"):
        training = workers[arm]["training"]
        _require(isinstance(training, dict) and training["steps"] == 32,
                 "Completed 32-step training receipt required")
        _require(training["initial_state_sha256"] == initial_hash
                 and training["initial_parameter_sha256"] == initial_parameter_hash,
                 "Training initialization differs from actual seed42 tensors")
        payload, archive = _bounded_load(training["checkpoint"], receipts, torch)
        state = payload
        storage = _check_state(state, schema, groups, audited, torch)
        final_hash = _semantic_hash(state)
        final_parameter_hash = _semantic_hash({name: state[name] for name in parameters})
        _require(final_hash == training["final_state_sha256"] and final_parameter_hash == training["final_parameter_sha256"],
                 "Final tensor hashes differ from worker receipt")
        _require(final_parameter_hash != initial_parameter_hash, "No actual parameter change")
        claimed = training["persisted_validation"]
        _require(claimed["weights_only"] is True and claimed["exact_tensor_values"] is True
                 and claimed["missing_keys"] == [] and claimed["unexpected_keys"] == []
                 and claimed["semantic_state_sha256"] == final_hash, "Persisted worker validation receipt differs")
        for key in ("tensor_count", "unique_storage_count", "alias_group_count", "named_tensor_bytes", "unique_storage_bytes"):
            _require(claimed["storage_contract"][key] == storage[key], "Persisted storage-count receipt differs")
        changed = [name for name in parameters if _tensor_hash(state[name]) != initial_tensor_hashes[name]]
        _require(bool(changed), "No independently measured changed parameter tensor")
        final_results[arm] = dict(checkpoint=dict(training["checkpoint"]), archive=archive, storage_contract=storage,
                                 semantic_state_sha256=final_hash, semantic_parameter_sha256=final_parameter_hash,
                                 initial_state_sha256=initial_hash, initial_parameter_sha256=initial_parameter_hash,
                                 actual_parameters_changed=True, changed_parameter_tensor_count=len(changed),
                                 changed_parameter_names=changed,
                                 elements_in_changed_parameter_tensors=sum(schema[name]["numel"] for name in changed),
                                 changed_element_count_not_measured=True)
        del state, payload
        gc.collect()
    for item in receipts.values():
        _require(_record(item["path"]) == item, "File changed during checkpoint review")
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    names_hash = hashlib.sha256(json.dumps(sorted(parameters), separators=(",", ":")).encode()).hexdigest()
    return dict(status="PASS", deserialization="torch.load(map_location='cpu', weights_only=True)",
                torch_threads=4, checkpoints_loaded=4, peak_concurrent_checkpoint_payloads=1,
                factory_constructions=1, factory_forward_calls=0, strict_model_load_calls=0,
                factory_schema_sha256=schema_hash, canonical_parameter_names_sha256=names_hash,
                unique_parameter_tensor_count=len(parameters), unique_buffer_tensor_count=len(buffers),
                parameter_element_count=parameter_elements,
                initial_state_sha256=initial_hash, initial_parameter_sha256=initial_parameter_hash,
                all_four_initializer_fingerprints_equal=True, initializer_archive=initializer_archive,
                initializer_storage_contract=initial_storage, adapted_checkpoints=final_results,
                verified_source_and_checkpoint_records_before_and_after=list(receipts.values()),
                benchmark_metric_code_imported=False, worker_code_imported=False,
                forward_calls=0, backward_calls=0, inference_calls=0, training_calls=0, network_requests=0,
                limitation="Tensor identity and actual parameter changes only; no causal mechanism, independent holdout, calibrated confidence or ink authenticity established.")

"""Prepared actual-seed42 smoke on synthetic input; root review required to run.

No scientific arrays, labels, metrics, checkpoint output, network or GPU calls.
Controller pins all local helper code before starting one bounded CPU worker.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
LAB = PHASE.parent.parent
LOADER = LAB / "reports/sixth-experiment/model-contract/model_contract_v2.py"
LOADER_SHA = "f391fde6495b2abdb68a77b6af22766fd3b8cdb2c2c9b597abf5c33cc2b693fd"
CHECKPOINT = dict(label="seed42", repository="scrollprize/ink_9um",
    revision="7109667e2607db1b90c37c8b09cb876ea7fe7bb1",
    repository_path="hybrid_3d2d-seed42/step-075000.pth", size_bytes=138360039,
    sha256="e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab",
    embedded_step=74999)
CHECKPOINT_PATH = LAB / "data/models/ink9um-seed42-step075000.pth"
RSS_CAP = 4 * 1024**3
OUTPUT_CAP = 64 * 1024**2
WALL_CAP = 300


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def verify_sources(expected_manifest_sha):
    manifest = HERE / "smoke-plan.json"
    require(sha(manifest) == expected_manifest_sha, "Prototype execution plan changed")
    value = json.loads(manifest.read_text())
    require(value["checkpoint"] == CHECKPOINT and value["checkpoint_path"] == str(CHECKPOINT_PATH),
            "Plan checkpoint identity differs")
    require(value["execution"] == dict(threads=4, interop_threads=1,
        wall_cap_seconds=WALL_CAP, rss_cap_bytes=RSS_CAP, output_cap_bytes=OUTPUT_CAP,
        input_shape=[1,1,17,128,128], input_kind="synthetic Gaussian FP32",
        input_generator_seed=20261009, adapter_seed=20261008,
        target="FP32 linspace(-1,1,16384) reshaped [1,1,128,128]",
        loss="mean squared error", optimizer="SGD", learning_rate=.001,
        momentum=0, weight_decay=0, steps_per_arm=1, backwards_per_arm=2,
        scientific_array_reads=0, scientific_training=False, retries=0),
        "Frozen synthetic execution contract differs")
    for record in value["records"]:
        path = Path(record["path"])
        require(path.is_file() and path.stat().st_size == record["size_bytes"]
                and sha(path) == record["sha256"], "Pinned prototype source changed")
    require(sha(LOADER) == LOADER_SHA, "Strict seed42 loader changed")
    return value


def state_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def worker(output, manifest_sha):
    start = time.monotonic()
    signal.alarm(WALL_CAP - 5)
    resource.setrlimit(resource.RLIMIT_CPU, (WALL_CAP - 5, WALL_CAP - 5))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_CAP, OUTPUT_CAP))
    verify_sources(manifest_sha)
    import torch
    import numpy as np
    from fusion_adapter import ARMS, PARAMETER_COUNT, FrozenFusionModel, matched_adapter
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.__version__ == "2.12.0+cpu", "Unexpected Torch runtime")
    require(sys.version_info[:3] == (3,12,14) and np.__version__ == "2.5.3",
            "Unexpected Python/NumPy runtime")
    require(not torch.cuda.is_available(), "Expected CPU-only runtime")
    spec = importlib.util.spec_from_file_location("phase19_verified_seed42_loader", LOADER)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    runtime = loader.load_runtime()
    model, metadata = loader.build_model(CHECKPOINT_PATH, CHECKPOINT, runtime)
    model.requires_grad_(False)
    model.eval()
    require(sum(p.numel() for p in model.parameters()) == 34546498,
            "Unexpected unique seed42 parameter count")
    before = state_digest(model)
    buffer_names = sorted(dict(model.named_buffers()))
    counters = dict(original_full_model=0, original_depth_fusion=0, original_2d_network=0,
                    adapter_forwards=0, backwards=0, synthetic_optimizer_steps=0)
    handles = []

    def count(name):
        def hook(module, inputs, output):
            counters[name] += 1
        return hook

    handles.append(model.depth_fusion.register_forward_hook(count("original_depth_fusion")))
    handles.append(model.network.register_forward_hook(count("original_2d_network")))
    handles.append(model.register_forward_hook(count("original_full_model")))
    # A fixed local generator creates synthetic standardized intensities; it
    # is not an acquired scan, an empirical normalization fit or a data cache.
    generator = torch.Generator(device="cpu").manual_seed(20261009)
    image = torch.randn((1, 1, 17, 128, 128), generator=generator, dtype=torch.float32)
    input_sha = hashlib.sha256(image.numpy().tobytes()).hexdigest()
    target = torch.linspace(-1, 1, 128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    target_sha = hashlib.sha256(target.numpy().tobytes()).hexdigest()
    with torch.no_grad():
        baseline = model(image)["ink"]
        fused = model.depth_fusion(image).detach()
    require(fused.shape == (1, 32, 128, 128), "Original fusion output differs")
    adapters = {arm: matched_adapter(20261008) for arm in ARMS}
    initial_states = {arm: state_digest(adapter) for arm, adapter in adapters.items()}
    require(len(set(initial_states.values())) == 1, "Arm initialization differs")
    observations = {}
    for arm in ARMS:
        adapter = adapters[arm]
        require(sum(p.numel() for p in adapter.parameters()) == PARAMETER_COUNT,
                "Adapter capacity differs")
        handles.append(adapter.register_forward_hook(count("adapter_forwards")))
        wrapper = FrozenFusionModel(model, adapter, arm).train()
        require(all(not child.training for child in model.modules()),
                "Some backbone module is not in eval mode")
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.001, momentum=0,
                                    weight_decay=0, foreach=False, fused=False)
        require({id(p) for group in optimizer.param_groups for p in group["params"]}
                == {id(p) for p in adapter.parameters()}, "Optimizer membership differs")
        logits, residual, adapted = wrapper.forward_from_fused(image, fused)
        require(torch.equal(fused, adapted) and not bool(residual.any()),
                "Zero initialization changed fused features")
        require(torch.equal(logits, baseline), "Zero initialization changed original logits")
        loss = ((logits - target) ** 2).mean()
        loss.backward()
        counters["backwards"] += 1
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()),
                "Missing/nonfinite adapter gradient")
        projection_grad = float(sum(p.grad.double().square().sum() for p in adapter.projection.parameters()).sqrt())
        first_stem_grad = float(sum(p.grad.double().square().sum() for p in adapter.stem.parameters()).sqrt())
        require(projection_grad > 0 and first_stem_grad == 0,
                "Unexpected first-step gradient path")
        stem_before = state_digest(adapter.stem)
        optimizer.step()
        counters["synthetic_optimizer_steps"] += 1
        require(state_digest(adapter.stem) == stem_before,
                "Zero-gradient hidden layer changed at synthetic step one")
        optimizer.zero_grad(set_to_none=True)
        second, _, _ = wrapper.forward_from_fused(image, fused)
        second_loss = ((second - target) ** 2).mean()
        second_loss.backward()
        counters["backwards"] += 1
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()),
                "Missing/nonfinite second adapter gradient")
        second_stem_grad = float(sum(p.grad.double().square().sum() for p in adapter.stem.parameters()).sqrt())
        require(second_stem_grad > 0, "Gradient did not reach hidden representation")
        require(not torch.equal(second, baseline), "Projection update did not change logits")
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),
                "Backbone gradient or trainability changed")
        require(all(not child.training for child in model.modules()), "Backbone left eval mode")
        require(state_digest(model) == before, "Backbone parameters or buffers changed")
        observations[arm] = dict(parameter_count=PARAMETER_COUNT,
            exact_initial_feature_identity=True, exact_initial_logit_identity=True,
            projection_gradient_norm=projection_grad, first_stem_gradient_norm=first_stem_grad,
            second_stem_gradient_norm=second_stem_grad,
            fixed_synthetic_loss_before=float(loss.detach()),
            fixed_synthetic_loss_after_one_step=float(second_loss.detach()),
            unchanged_backbone_state=True, backbone_gradients_absent=True,
            all_backbone_modules_eval=True, adapter_final_state_sha256=state_digest(adapter))
        del logits, residual, adapted, loss, second, second_loss, optimizer, wrapper
    expected = dict(original_full_model=1, original_depth_fusion=2, original_2d_network=5,
                    adapter_forwards=4, backwards=4, synthetic_optimizer_steps=2)
    require(counters == expected, "Observed model call counts differ from frozen schedule")
    require(state_digest(model) == before, "Final backbone state changed")
    for handle in handles:
        handle.remove()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak_rss < RSS_CAP and time.monotonic() - start < WALL_CAP,
            "Worker resource cap exceeded")
    result = dict(kind="phase19-actual-seed42-synthetic-integration-smoke-v1", status="PASS",
        execution_plan_sha256=manifest_sha, strict_checkpoint=CHECKPOINT,
        initial_arm_state_sha256=initial_states,
        backbone_state_before_sha256=before, backbone_state_after_sha256=state_digest(model),
        backbone_named_buffer_names=buffer_names,
        input=dict(kind="synthetic iid CPU FP32 Gaussian", shape=list(image.shape),
                   generator_seed=20261009, sha256=input_sha),
        objective=dict(kind="synthetic MSE to fixed spatial linspace(-1,1)", target_sha256=target_sha),
        observations=observations, actual_calls=counters,
        resources=dict(wall_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss,
                       threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads()),
        scientific_array_reads=0, annotation_reads=0, network_calls=0, gpu_calls=0,
        saved_model_checkpoints=0, pretrained_checkpoint_loads=1,
        limitations=["Synthetic integration only; no ink-detection efficacy or timing on scans.",
                     "Two synthetic SGD steps validate mechanics, not a scientific training protocol.",
                     "No trained adapter state is saved or eligible for evaluation."])
    write_json(output / "worker-result.json", result)
    print(json.dumps(dict(status="PASS", actual_calls=counters, resources=result["resources"])))


def controller(output, manifest_sha):
    verify_sources(manifest_sha)
    require(output.parent == PHASE and output.name == "actual-backbone-smoke",
            "Output must be the new phase19 actual-backbone-smoke sibling directory")
    require(not output.exists(), "Refuse overwrite or implicit retry")
    output.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--output", str(output), "--expected-plan-sha256", manifest_sha]
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    start = time.monotonic()
    write_json(output / "controller-start.json", dict(command=command,
        execution_plan_sha256=manifest_sha, wall_cap_seconds=WALL_CAP,
        rss_cap_bytes=RSS_CAP, output_cap_bytes=OUTPUT_CAP, retries=0,
        scope="Actual seed42, deterministic synthetic input/objective only"))
    stop_reason = None
    peak_sampled_rss = 0
    with (output / "worker.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        while process.poll() is None:
            if time.monotonic() - start > WALL_CAP:
                stop_reason = "wall cap"
            status = Path(f"/proc/{process.pid}/status")
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        peak_sampled_rss = max(peak_sampled_rss, int(line.split()[1]) * 1024)
            except FileNotFoundError:
                pass
            if peak_sampled_rss > RSS_CAP:
                stop_reason = "RSS cap"
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > OUTPUT_CAP:
                stop_reason = "output cap"
            if stop_reason:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.05)
        code = process.wait()
    result = dict(status="PASS" if code == 0 and stop_reason is None else "FAIL",
        exit_code=code, stop_reason=stop_reason, wall_seconds=time.monotonic()-start,
        peak_sampled_rss_bytes=peak_sampled_rss, retries=0,
        worker_result_sha256=sha(output/"worker-result.json") if (output/"worker-result.json").is_file() else None)
    write_json(output / "controller-result.json", result)
    print(json.dumps(result))
    require(result["status"] == "PASS", "Bounded smoke failed; preserve evidence and do not retry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.expected_plan_sha256)
    else:
        controller(output, args.expected_plan_sha256)


if __name__ == "__main__":
    main()

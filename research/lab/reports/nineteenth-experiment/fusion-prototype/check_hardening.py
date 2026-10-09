"""Only the two new guards and their unaffected valid path, after source review."""
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from fusion_adapter import FrozenFusionModel, depth_features, matched_adapter
from check_synthetic import SyntheticBackbone

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(9182026)
start = time.monotonic()
checks = []


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    checks.append(dict(name=name, passed=True))


image = torch.randn(1,1,17,8,8)
overflow = image.clone()
overflow[:,:,0] = -torch.finfo(torch.float32).max
overflow[:,:,1] = torch.finfo(torch.float32).max
try:
    depth_features(overflow, "explicit_depth")
except ValueError as error:
    check("finite_inputs_with_overflowed_derivative_rejected", str(error) == "Nonfinite derived depth feature")
else:
    raise AssertionError("Overflow guard did not reject")
check("ordinary_feature_path_remains_finite", bool(torch.isfinite(depth_features(image,"explicit_depth")).all()))
base = SyntheticBackbone()
wrapper = FrozenFusionModel(base, matched_adapter(), "explicit_depth")
calls = []
handle = base.depth_fusion.register_forward_hook(lambda *args: calls.append("fusion"))
base.network.train()
try:
    wrapper.frozen_features(image)
except ValueError as error:
    check("nested_train_mode_rejected_before_any_fusion", not calls and str(error) == "Backbone must remain frozen and in eval mode")
else:
    raise AssertionError("Nested eval guard did not reject")
base.eval()
fused = wrapper.frozen_features(image)
check("ordinary_fusion_path_preserved", calls == ["fusion"] and fused.shape == (1,32,8,8))
handle.remove()
result = dict(kind="phase19-targeted-post-review-hardening-v2", status="PASS", checks=checks,
    coverage="Supplement to retained 45-check v1; only two added guards and valid paths rerun",
    source_pins=[dict(path=str(HERE/n),sha256=hashlib.sha256((HERE/n).read_bytes()).hexdigest())
                 for n in ["fusion_adapter.py","check_synthetic.py","check_hardening.py"]],
    wall_seconds=time.monotonic()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    pretrained_checkpoint_loads=0, scientific_array_reads=0, network_calls=0,
    optimizer_steps=0, backwards=0)
with (HERE/"synthetic-results-v2.json").open("x") as stream:
    stream.write(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASS",check_count=len(checks),wall_seconds=result["wall_seconds"],peak_rss_bytes=result["peak_rss_bytes"])))

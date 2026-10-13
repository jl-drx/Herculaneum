"""Qualify a CPU tensor-layout optimization without changing weights or precision."""
from pathlib import Path
import importlib.util,json,time
import numpy as np,torch
P=Path(__file__).resolve().parent;W=P/'work';OLD=P.parent/'fortyfifth-experiment'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
spec=importlib.util.spec_from_file_location('hecate',OLD/'source-evidence/hecate-hecate.py');hec=importlib.util.module_from_spec(spec);spec.loader.exec_module(hec)
model=hec.load_model(W/'hecate_9.6um.pth','cpu');raw=np.random.default_rng(20261009).integers(1,256,size=(16,64,64),dtype=np.uint8);x=torch.from_numpy((raw.astype(np.float32)/255)[None,None]).repeat(4,1,1,1,1)
baseline=np.load(W/'synthetic-hecate-batch.npy')
torch.nn.utils.convert_conv3d_weight_memory_format(model,torch.channels_last_3d)
with torch.inference_mode():
    timings=[]
    for _ in range(2):
        t=time.monotonic();out=model(x).sigmoid().numpy();timings.append(time.monotonic()-t)
err=float(np.max(np.abs(out-baseline)));assert np.isfinite(out).all() and err<1e-4,err
np.save(W/'synthetic-hecate-channels-last.npy',out)
(P/'HECATE_LAYOUT_BENCHMARK.json').write_text(json.dumps(dict(status='PASS',input='same deterministic synthetic tile as runtime_smoke',precision='fp32',change='Conv3d weight memory layout only',batch4_seconds=timings,baseline_batch4_seconds=json.loads((P/'RUNTIME_SMOKE.json').read_text())['models'][-1]['batch4_seconds'],max_probability_difference=err,tile_forwards=8,admission='Requires a second comparison on the fixed actual CT tile before full run.'),indent=2)+'\n')
print('channels last',timings,'max difference',err,flush=True)

"""Compare regenerated CPU model tensors with the sealed phase44 fingerprints."""
import argparse,hashlib,json
from pathlib import Path
import torch

def fingerprint(state):
 return {name:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for name,t in sorted(state.items())}
def verify(checkpoint,manifest,arm):
 actual=fingerprint(torch.load(checkpoint,map_location='cpu',weights_only=True));expected=manifest['checkpoints'][arm]['tensors']
 if actual!=expected:raise ValueError('Model tensors differ from the sealed predecessor')
 return dict(status='PASS',arm=arm,tensors=len(actual),all_tensors_bit_identical=True,source_container_sha256_matches=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==manifest['checkpoints'][arm]['source_checkpoint_sha256'])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();print(json.dumps(verify(a.checkpoint,json.loads(a.manifest.read_text()),a.arm)))
if __name__=='__main__':main()

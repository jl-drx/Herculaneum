"""Verify a freshly fetched private LFS snapshot and replay one completed job.

Fetch the snapshot into an empty dedicated LFS cache first; this script does no
network access and no trained-model forward. It restores immutable files only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import torch
import numpy as np
import runner as r


class NoInference(torch.nn.Module):
    def forward(self, *args):
        raise AssertionError('Remote restoration must not run a model')


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--lfs-cache',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.destination.exists() and not a.output.exists()
    index=json.loads(subprocess.check_output(['git','-C',str(a.repo),'show',a.commit+':INDEX.json']))
    archived=a.destination/'archives';archived.mkdir(parents=True)
    (archived/'INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    for generation in index['generations']:
        sha=generation['sha256'];path=a.lfs_cache/'objects'/sha[:2]/sha[2:4]/sha
        assert path.stat().st_size==generation['bytes'] and r.sha(path)==sha
        shutil.copy2(path,archived/generation['name'])
    restore_file=a.repo/'research/in-progress/pherc0139-native-learning-20261011/restore_backup.py'
    spec=importlib.util.spec_from_file_location('restore_original',restore_file)
    restore=importlib.util.module_from_spec(spec);spec.loader.exec_module(restore)
    run=a.destination/'restored';result=restore.restore(archived,run)
    go=r.read(run/'GO.json');assert r.sha(run/'GO.json')==index['go_sha256']
    for row in go['code_pins']:r.check(run,row)
    for row in go['runtime_pins']:r.check(a.repo,row)
    for row in go['input_pins']:r.check(a.inputs,row)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    infer,hec=r.runtime(a.repo)
    case=r.read(a.inputs/'SELECTION.json')['cases'][0]
    patches=r.read(a.inputs/'PATCHES_COMPLETE.json')['cases'][0]
    folder=run/'jobs'/('reader_v2--'+case['id']);done=r.read(folder/'COMPLETE.json')
    maps,counts=r.calculate(infer,hec,a.inputs,case,patches,'reader_v2',NoInference(),folder,done['binding'],readonly=True)
    for core_id,arrays in maps.items():
        row=next(x for x in done['outputs'] if x['path']==core_id+'.npz')
        path=r.check(folder,row)
        with np.load(path) as saved:
            assert set(saved.files)==set(arrays)
            assert all(np.array_equal(arrays[k],saved[k]) for k in arrays)
    assert counts==dict(model_calls=54,new_forwards=0,reused_forwards=54)
    result.update(remote_commit=a.commit,archive_bytes=sum(g['bytes'] for g in index['generations']),
        go_sha256=index['go_sha256'],restored_job='reader_v2--'+case['id'],restored_core_maps=6,
        exact_replay=True,replay_counts=counts,trained_model_forwards=0,
        scope='Whole snapshot file/hash restoration plus first completed job numeric replay; not process restart or all-job numeric replay',
        implementation=r.pin(Path(__file__)),restore_implementation=r.pin(restore_file))
    r.atomic(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

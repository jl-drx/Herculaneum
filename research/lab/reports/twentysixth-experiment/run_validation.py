"""Frozen offline validation and geometry-only planning; never opens a network socket."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import unittest
import numpy as np


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def no_network(*args,**kwargs):
    raise AssertionError('Offline study: network prohibited')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();freeze=json.loads(Path(args.freeze).read_text())
    for record in freeze['files']:assert pin(record['path'])==record
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
    os.environ['XDG_CACHE_HOME']=str(out/'xdg-cache')
    socket.socket.connect=no_network;socket.socket.connect_ex=no_network;socket.create_connection=no_network
    started=time.monotonic()
    import test_reference_mechanics as test
    from reference_mechanics import VolumeSpec,ray_points,plan_samples,sample_trilinear,affine_landmark_check
    suite=unittest.defaultTestLoader.loadTestsFromModule(test)
    text=io.StringIO();result=unittest.TextTestRunner(stream=text,verbosity=2).run(suite)
    (out/'tests.log').write_text(text.getvalue())
    save(out/'TESTS.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        log=pin(out/'tests.log'),network_calls=0,model_calls=0))
    print(text.getvalue())
    if not result.wasSuccessful():raise SystemExit(1)
    volume=test.source_volume();spec=VolumeSpec(volume.shape,(4,3,5),'<f8')
    points=np.random.default_rng(829).uniform(1,7,(256,3))
    expected=test.polynomial(*points.T)
    measurements={}
    for name,locations in [('correct',points),('swapped_xz',points[:,::-1]),('half_voxel_shift',points+.5)]:
        load,_=test.loader_for(volume,spec)
        values,valid,stats=sample_trilinear(locations,spec,load)
        assert valid.all()
        error=np.abs(values-expected)
        measurements[name]=dict(mean_absolute_error=float(error.mean()),maximum_absolute_error=float(error.max()))
    assert measurements['correct']['maximum_absolute_error']<=1e-10
    assert measurements['swapped_xz']['mean_absolute_error']>1
    assert measurements['half_voxel_shift']['mean_absolute_error']>1
    src=np.array([[0,0],[10,0],[0,10],[10,10],[2,3],[7,8]],dtype=float)
    matrix=np.array([[1.1,.2,3],[-.1,.9,-2.]])
    true=src@matrix[:,:2].T+matrix[:,2];targets=true.copy();targets[4]+=[3,4]
    ids=list('abcdef')
    registration=affine_landmark_check(src,targets,ids,fit_ids=ids[:4],check_ids=ids[4:])
    assert registration['fit']['maximum_pixels']<1e-12
    assert abs(registration['check']['maximum_pixels']-5)<1e-12
    save(out/'SYNTHETIC_DIAGNOSTICS.json',dict(status='PASS',sampling=measurements,registration=registration,
        scope='Synthetic mechanics and negative controls; not candidate accuracy.'))
    ppm_record=freeze['sparse_geometry_records']
    assert pin(ppm_record['path'])==ppm_record
    records=np.frombuffer(Path(ppm_record['path']).read_bytes(),dtype='<f8').reshape(-1,6)
    assert len(records)==729
    mapped=~np.all(records==0,axis=1);assert int(mapped.sum())==558
    rays,ray_valid=ray_points(records[:,:3],records[:,3:],[-3.24,0,3.24],[3.24]*3,
        direction_space='voxel_unit_vector',mapped=mapped)
    hypothetical=VolumeSpec((8855,2260,6300),(64,64,64),'<u2')
    plan=plan_samples(rays.reshape(-1,3),hypothetical,mapped=ray_valid.reshape(-1))
    assert plan.geometric_valid.sum()==1674
    dry=dict(status='PASS',retained_geometry=ppm_record,total_records=729,mapped_records=558,zero_sentinels=171,
        diagnostic_offsets_um=[-3.24,0,3.24],assumed_spacing_xyz_um=[3.24]*3,
        hypothetical_shape_zyx=list(hypothetical.shape_zyx),hypothetical_chunks_zyx=[64]*3,
        hypothetical_dtype='<u2',source_volume_bound=False,actual_storage_layout_verified=False,
        requests_made=0,source_voxels_loaded=0,stage_C_authorized=False,
        limitation='Hypothetical source layout and vector interpretation; not an authenticated Frag6 render recipe or compressed/wire-byte estimate.',
        **plan.summary())
    save(out/'SPARSE_GEOMETRY_DRY_PLAN.json',dry)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    depths=np.linspace(0,10,41);line=np.column_stack((np.full(41,2.3),np.full(41,3.4),depths))
    load,_=test.loader_for(volume,spec);samples,valid,_=sample_trilinear(line,spec,load);assert valid.all()
    axes[0].plot(depths,test.polynomial(*line.T),label='Analytic truth',lw=2)
    axes[0].scatter(depths[::2],samples[::2],label='Chunked interpolation',s=15,color='black')
    axes[0].set(xlabel='Z voxel-center coordinate',ylabel='Synthetic intensity',title='Known coordinate phantom');axes[0].legend(fontsize=8)
    names=list(measurements);means=[measurements[n]['mean_absolute_error'] for n in names]
    axes[1].bar(['Correct','X/Z swapped','Half-voxel\nshift'],means,color=['#287d5b','#bd493b','#d79835'])
    axes[1].set(ylabel='Mean absolute synthetic error',title='Deliberate coordinate mistakes')
    for i,value in enumerate(means):axes[1].text(i,value+3,f'{value:.3g}',ha='center',fontsize=9)
    axes[1].set_ylim(0,max(means)*1.18)
    axes[2].scatter(true[:4,0],true[:4,1],marker='s',label='Fit landmarks')
    axes[2].scatter(true[4:,0],true[4:,1],marker='o',facecolors='none',edgecolors='black',label='Held-out truth')
    axes[2].scatter(targets[4:,0],targets[4:,1],marker='x',color='#bd493b',label='Held-out observations')
    axes[2].annotate('',xy=targets[4],xytext=true[4],arrowprops=dict(arrowstyle='->',color='#bd493b'))
    axes[2].set(xlabel='Target X pixels',ylabel='Target Y pixels',title='5-pixel held-out error remains visible')
    axes[2].set_aspect('equal');axes[2].legend(fontsize=8)
    fig.suptitle('Synthetic reference-workflow controls — no scroll images or predictions',fontsize=13)
    fig.savefig(out/'synthetic-controls.png',dpi=150);plt.close(fig)
    for record in freeze['files']:assert pin(record['path'])==record
    assert pin(ppm_record['path'])==ppm_record
    phase=Path(__file__).resolve().parent
    artifact_bytes=sum(f.stat().st_size for f in phase.rglob('*') if f.is_file())
    assert artifact_bytes<16*1024**2
    receipt=dict(status='PASS',elapsed_seconds=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        tests=result.testsRun,network_calls=0,model_calls=0,scientific_optimizer_updates=0,new_candidate_image_pixels=0,
        retained_ppm_records=729,GPU_spend_GBP=0,source_unchanged=True,phase_bytes_before_receipt=artifact_bytes,
        freeze=pin(args.freeze),outputs=[pin(out/f) for f in ['TESTS.json','SYNTHETIC_DIAGNOSTICS.json','SPARSE_GEOMETRY_DRY_PLAN.json','synthetic-controls.png']])
    save(out/'RUN_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

"""Build a small model-comparison intake from verified public metadata only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    pins=[];failures=[]
    def read(name):
        path=a.metadata/(name+'.raw')
        r=json.loads(path.with_suffix('.receipt.json').read_text())
        assert r['status']==200 and sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
        return json.loads(path.read_text())
    requests=[]
    for plan in ['REQUEST_PLAN.json','REQUEST_PLAN_2.json','REQUEST_PLAN_3.json']:
        requests.extend(json.loads((a.packet/plan).read_text()))
    for req in requests:
        r=json.loads((a.metadata/(req['id']+'.receipt.json')).read_text())
        assert r['url']==req['url']
        if 'error_type' in r:
            assert r['http_status']==404
            failures.append({'id':req['id'],'http_status':404})
        else:
            f=a.metadata/r['file'];assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
            pins.append({'file':f.name,'bytes':r['bytes'],'sha256':r['sha256']})
    admission=json.loads((a.packet/'CATALOG_ADMISSION.json').read_text())
    rows=[]
    for case in admission['rows']:
        sample,sid=case['sample'],case['segment'];prefix=sample.lower()+'-'+sid
        renders=[]
        for d in case['data']:
            if d['type']!='layers-zarr':continue
            vid=d['parameters']['target_volume'];ident=prefix+'-'+vid
            attrs=read(ident+'-attrs');arr=read(ident+'-array')
            scale=next(t['scale'] for t in attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'] if t['type']=='scale')
            assert attrs['canvas_size']==[arr['shape'][2],arr['shape'][1]]
            renders.append({'volume':vid,'shape_zyx':arr['shape'],'spacing_zyx_um':scale,'chunks':arr['chunks'],
                            'dtype':arr['dtype'],'compressor':arr['compressor'],'source_path':d['origins'][0]['path']})
        renders.sort(key=lambda r:r['spacing_zyx_um'][0],reverse=True)
        assert len(renders)==2
        native,fine=renders
        roles=[]
        for role in ['inklabels','supervision','validation']:
            group=read(prefix+'-'+role+'-group-v3')['attributes'];array=read(prefix+'-'+role+'-array-v3')
            rp=group['ink_republish']
            assert array['shape']==fine['shape_zyx'][1:] and array['data_type']=='uint8'
            assert array['dimension_names']==['y','x'] and rp['atlas_sample']==sample and rp['atlas_segment']==case['long_id']
            roles.append({'role':role,'shape_yx':array['shape'],'zarr_format':array['zarr_format'],
                          'dtype':array['data_type'],'chunk_grid':array['chunk_grid'],'codecs':array['codecs'],
                          'republish_metadata':rp})
        transforms=[]
        for origin in case['volume_transforms']:
            for target in origin['transforms']:
                if {origin['from_volume_id'],target['to_volume_id']}=={native['volume'],fine['volume']}:
                    m=np.eye(4);m[:3]=target['matrix'];transforms.append(m)
        assert len(transforms)==2
        inverse_error=float(np.max(np.abs(transforms[0]@transforms[1]-np.eye(4))))
        assert inverse_error<1e-7
        label=next(d for d in case['data'] if d['type']=='ink-labels')
        rows.append({'sample':sample,'segment':sid,'native':native,'fine':fine,'labels':roles,
                     'label_catalog_properties':label['properties'],'affine_inverse_max_error':inverse_error,
                     'affine_check_scope':'Algebra only, not local physical registration.',
                     'reference_role':'Published transferred/model-reviewed annotation agreement on development material; not authenticated physical ink truth.',
                     'prior_audit':'Study 22 for PHerc0841; study 11 source-transfer contract for PHerc0009B. Prior unresolved provenance/validity limits retained.'})
    out={'status':'METADATA_QUALIFIED_NOT_PIXEL_EVALUATION_READY','requests':len(requests),
         'successful_metadata_responses':len(pins),'successful_response_bytes':sum(p['bytes'] for p in pins),
         'initial_v2_lookup_404s':failures,'resolved_format':'Label packages are sharded Zarr v3; render volumes are Zarr v2.',
         'rows':rows,'source_pins':pins,'array_payloads_read':0,'new_inference':0,
         'next':'Reuse prior provenance conclusions. Prepare one mask-selected fixed common-support comparison with supplied-reference agreement explicitly separated from independent accuracy; no further provenance search in this batch.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','source_pins','initial_v2_lookup_404s']}))


if __name__=='__main__':main()

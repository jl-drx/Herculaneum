"""Replay source contract checks using saved small public metadata responses."""
from pathlib import Path
import hashlib,json,math
BASE=Path(__file__).resolve().parent

def main():
    extract=json.loads((BASE/'CATALOG_EXTRACT.json').read_text())
    arrays=json.loads((BASE/'RETAINED_ARRAYS.json').read_text())
    claims=[];requests=[]
    for folder in [BASE,BASE/'sampling-metadata']:
        plan=json.loads((folder/'REQUEST_PLAN.json').read_text());ledger=json.loads((folder/'REQUEST_LEDGER.json').read_text());assert len(plan['requests'])==len(ledger)
        for req,item in zip(plan['requests'],ledger):
            assert req['url']==item['url'] and item['status']==200
            body=(folder/'responses'/item['name']).read_bytes();assert len(body)==item['bytes'] and hashlib.sha256(body).hexdigest()==item['sha256']
        requests.extend(ledger)
    volumes={'2.399':'20260102150214','9.362':'20250728140407','1.129':'20260413113053'}
    rows=[]
    for a in arrays:
        sid,spacing=a['tag'].split('-');vid=volumes[spacing];attrs=json.loads((BASE/'responses'/f'{sid}-{vid}-layers-zarr.json').read_text());arr=a['metadata'];multi=attrs['multiscales'][0]
        assert [k['name'] for k in multi['axes']]==['z','y','x']
        assert attrs['canvas_size']==arr['shape'][::-1][:2]
        assert attrs['num_slices']==arr['shape'][0] and attrs['chunk_size']==arr['chunks']
        transforms=multi['datasets'][0]['coordinateTransformations'];sc=next(t['scale'] for t in transforms if t['type']=='scale');tr=next(t['translation'] for t in transforms if t['type']=='translation')
        expected=2.258 if spacing=='1.129' else float(spacing)
        assert sc==[expected]*3 and tr==[0.0]*3
        rows.append({'segment':sid,'volume':vid,'shape_zyx':arr['shape'],'chunks_zyx':arr['chunks'],'nominal_surface_scale_um_zyx':sc,'source_group':attrs['source_group'],'raw_chunk_bytes':math.prod(arr['chunks']),'centered_depth_extent_um_if_renderer_centered':[-(arr['shape'][0]-1)*expected/2,(arr['shape'][0]-1)*expected/2]})
    fine_pyramids=[]
    for sid in extract['segments']:
        ar=json.loads((BASE/'sampling-metadata/responses'/f'{sid}-fine78-level2-zarray.json').read_text());attrs=json.loads((BASE/'responses'/f'{sid}-20260102150214-layers-zarr.json').read_text());sc=next(t['scale'] for t in attrs['multiscales'][0]['datasets'][2]['coordinateTransformations'] if t['type']=='scale')
        assert sc==[2.399,9.596,9.596] and ar['shape'][0]==109 and ar['chunks']==[109,128,128] and ar['compressor'] is None
        fine_pyramids.append({'segment':sid,'level':'2','shape_zyx':ar['shape'],'scale_um_zyx':sc,'raw_chunk_bytes':math.prod(ar['chunks'])})
    recons=[]
    for vid in ['20250728140407','20251107132835','20251107135911']:
        ar=json.loads((BASE/'sampling-metadata/responses'/f'{vid}-0-zarray.json').read_text());v=extract['volumes'][vid]
        assert ar['shape']==v['properties']['shape'] and ar['chunks']==[128]*3 and ar['compressor'] is None and ar['dtype']=='|u1'
        recons.append({'volume':vid,'scan':v['scan_id'],'shape_zyx':ar['shape'],'chunk_bytes':128**3,'physical_spacing_um_from_catalog':v['properties']['pixel_size_um'],'note':'Volume OME scales are relative, unitless; do not read level0 scale=1 as1um.'})
    report={'request_count':len(requests),'response_body_bytes':sum(i['bytes'] for i in requests),'surface_contracts_verified':len(rows),'surfaces':rows,'fine78_level2':fine_pyramids,'same_scan_reconstructions':recons,'scope':'Metadata consistency, not local registration, validity or paired-tissue admission. Depth extents conditional on stated centered-render convention. No array pixels read by this script.','future_pairing_budget_note':'Fine level2 changes XY only:109planes at2.399um, XY9.596um.16chunksfine+16chunksnative per bounded block =35913728bytes (34.25MiB); no block assigned or fetched by this metadata plan.'}
    assert len(set(r['scan'] for r in recons))==1
    (BASE/'METADATA_FINDINGS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['request_count','response_body_bytes','surface_contracts_verified']}))
if __name__=='__main__':main()

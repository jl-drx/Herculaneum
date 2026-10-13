"""Resume the unchanged CT plan using independently parsed complete inventories."""
import argparse, json, shutil, time, xml.etree.ElementTree as ET
from pathlib import Path
import intake


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args();root=a.work
    intake.verify_freeze(root);plan=intake.load(root/'CT_PLAN.json');selection=intake.load(root/'SELECTION.json')
    assert plan['selection_sha256']==intake.sha(root/'SELECTION.json')
    copies=['acquire_sparse_ct.py','CT_FILL_AMENDMENT.md']
    amendment=root/'CT_FILL_EXECUTION.json'
    pins={name:intake.pin(Path(__file__).parent/name) for name in copies}
    if amendment.exists():assert intake.load(amendment)['implementation']==pins
    else:
        for name in copies:shutil.copyfile(Path(__file__).parent/name,root/'implementation'/name)
        intake.write(amendment,{'utc':intake.stamp(),'implementation':pins,
             'selection_sha256':intake.sha(root/'SELECTION.json'),'original_ct_plan_sha256':intake.sha(root/'CT_PLAN.json'),
             'cohort_changed':False,'inference_started':False})
    ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'};objects={};invpins={}
    for c in selection['cases']:
        ip=root/'ct-inventory'/(c['id']+'.json');inventory=intake.load(ip);invpins[ip.name]=intake.pin(ip)
        assert inventory['complete'] and inventory['prefix']==c['native']['source_path']+'0/0/'
        parsed=[]
        for n,r in enumerate(inventory['receipts']):
            p=ip.parent/r['file'];assert intake.pin(p)=={k:r[k] for k in ['bytes','sha256']}
            doc=ET.fromstring(p.read_bytes());assert doc.findtext('s:Prefix',namespaces=ns)==inventory['prefix']
            assert doc.findtext('s:IsTruncated',namespaces=ns)==('false' if n==len(inventory['receipts'])-1 else 'true')
            for e in doc.findall('s:Contents',ns):
                row={'key':e.findtext('s:Key',namespaces=ns),'size':int(e.findtext('s:Size',namespaces=ns)),
                     'etag':e.findtext('s:ETag',namespaces=ns)}
                assert row['key'].startswith(inventory['prefix']) and row['key'] not in objects
                parsed.append(row);objects[row['key']]=row
        assert parsed==inventory['objects']
        meta=intake.load(root/'metadata'/f'{c["id"]}-{c["native"]["volume"]}-array.raw')
        assert meta['fill_value']==0 and meta['dtype']=='|u1' and meta['compressor'] is None
    execution={'implementation':pins,'inventory_pins':invpins,'selection_sha256':plan['selection_sha256']}
    intake.write(root/'CT_SPARSE_GO.json',execution)
    start=time.monotonic();present=fill=body=0
    for n,r in enumerate(plan['requests']):
        key=r['url'].removeprefix(intake.BASE)
        if key in objects:
            obj=objects[key];assert obj['size']==r['expected_bytes']
            intake.fetch(root,dict(r,etag=obj['etag']),'ct',plan['body_cap_bytes']);present+=1;body+=r['expected_bytes']
        else:
            dest=root/r['relative_path'];receipt=root/'receipts'/(r['id']+'.json')
            zeros=bytes(r['expected_bytes'])
            expected=dict(id=r['id'],url=r['url'],bytes=len(zeros),sha256=__import__('hashlib').sha256(zeros).hexdigest(),
                          relative_path=r['relative_path'],stage='canonical_fill',source_body_bytes=0,
                          inventory_pin=invpins[r['case']+'.json'],
                          identity_scope='Absent from complete native-array inventory; declared Zarr fill=0; not measured CT')
            if receipt.exists():assert intake.load(receipt)==expected and intake.pin(dest)=={k:expected[k] for k in ['bytes','sha256']}
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(zeros);intake.write(receipt,expected)
            fill+=1
        if (n+1)%25==0:print(json.dumps({'processed':n+1,'total':len(plan['requests']),'retrieved':present,'canonical_fill':fill}),flush=True)
    result=dict(status='PASS',plan_sha256=intake.sha(root/'CT_PLAN.json'),sparse_go_sha256=intake.sha(root/'CT_SPARSE_GO.json'),
                objects=present+fill,retrieved_objects=present,canonical_fill_objects=fill,downloaded_body_bytes=body,
                materialized_array_bytes=plan['expected_bytes'],seconds=time.monotonic()-start,
                first_attempt_404_preserved=True,cohort_changed=False)
    intake.write(root/'CT_COMPLETE.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

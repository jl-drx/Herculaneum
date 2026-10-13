"""Offline integrity, decoder and coverage replay; never invokes acquisition."""
import argparse,importlib.metadata,json,pathlib,sys
import numpy as np
import intake as original

parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
R=pathlib.Path(__file__).resolve().parent
original.verify_freeze()
ledger=original.load(R/'REQUEST_LEDGER.json')
assert ledger['complete'] and len(ledger['requests'])==6 and ledger['body_bytes']==65648
for row in ledger['requests']:
    b=(R/row['path']).read_bytes();assert len(b)==row['bytes'] and original.sha(b)==row['sha256']
mask_receipt=original.load(R/'MASK_STAGE_COMPLETE.json')
for path,h in mask_receipt['files'].items():assert original.sha((R/path).read_bytes())==h
plan=original.load(R/'SOURCE_PLAN.json');old_cov={s['winding']:s for s in original.load(R/'MASK_COVERAGE.json')['sites']};old_results={s['winding']:s for s in original.load(R/'RESULTS.json')['sites']}
checks=[]
for w in ['w035','w039','w044']:
    m_item=next(r for r in plan['arrays'] if r['winding']==w and r['role']=='supervision')
    i_item=next(r for r in plan['arrays'] if r['winding']==w and r['role']=='inklabels')
    m,mi=original.decode_crosscheck(R/'arrays'/w/'supervision',original.metadata(m_item))
    i,ii=original.decode_crosscheck(R/'arrays'/w/'inklabels',original.metadata(i_item))
    c=original.enumerate_coverage(m,m_item);assert c==old_cov[w]
    v,p=m>0,i>0;expected=old_results[w]
    actual={'total_ink_pixels':int(np.count_nonzero(p)),'supervised_pixels':int(np.count_nonzero(v)),'ink_supervised':int(np.count_nonzero(p&v)),'nonink_supervised':int(np.count_nonzero(~p&v)),'ink_outside_supervision':int(np.count_nonzero(p&~v))}
    assert all(expected[k]==val for k,val in actual.items())
    for row in expected['rows']:
        x,y=row['x'],row['y'];vv=v[y:y+128,x:x+128];pp=p[y:y+128,x:x+128]
        assert row['ink_supervised']==int(np.count_nonzero(vv&pp))
        assert row['nonink_supervised']==int(np.count_nonzero(vv&~pp))
        assert row['ink_outside_supervision']==int(np.count_nonzero(~vv&pp))
    checks.append({'winding':w,'all_full_array_counts_match':True,'every_eligible_core_count_matches':True,'all_grid_rows_and_ct_chunks_match':True,'mask_decoded_sha256':mi['decoded_sha256'],'label_decoded_sha256':ii['decoded_sha256']})
original.write(args.output/'REPLAY.json',{'checks':checks,'all_passed':True,'network_access':False,'source_packet':str(R)})
print(json.dumps(checks,indent=2))

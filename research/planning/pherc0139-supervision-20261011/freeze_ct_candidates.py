"""Freeze all previously described CT alternatives; never opens target arrays."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--correspondence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    scorepath=a.correspondence/'method-review/score-check/site1-independent-landscapes.npy'
    resultpath=a.correspondence/'method-review/score-check/REAL_DATA_REPLAY.json'
    scores=np.load(scorepath,allow_pickle=False);result=json.loads(resultpath.read_text())['sites'][0]
    fit=result['fit_translation_zyx'];assert fit==[0,-1,0]
    entries=[]
    for i,e in enumerate(result['supports'][4:],4):
        indices=np.argwhere(scores[i]>=e['at_fit_score']-.05)
        shifts=indices-6;assert len(shifts)==[11,6,22,4][i-4]
        assert any(np.array_equal(s,fit) for s in shifts)
        x,y=e['centre_core_xy']; x+=2304;y+=1792
        entries.append(dict(support=i,centre_native_xy=[x,y],native_xy_half_open=[x-16,y-16,x+16,y+16],at_fit_score=e['at_fit_score'],score_threshold=e['at_fit_score']-.05,candidates=[dict(shift_zyx=s.tolist(),ncc=float(scores[(i,*idx)])) for s,idx in zip(shifts,indices)],nonzero_dz_candidates=sum(int(s[0]!=0) for s in shifts)))
    out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='finite CT-based sensitivity sets fixed before target pixels; not calibrated confidence regions or probabilities',site='site1',fit_translation_zyx=fit,prior_structural_screen='FAIL; unchanged',input_files=[dict(relative=str(p.relative_to(a.correspondence)),sha256=sha(p)) for p in [scorepath,resultpath]],supports=entries)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(support_counts=[len(e['candidates']) for e in entries],nonzero_dz_counts=[e['nonzero_dz_candidates'] for e in entries],sha256=sha(a.output))))
if __name__=='__main__':main()

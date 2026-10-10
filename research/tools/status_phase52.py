"""Read process identity and retained phase52 progress; never starts inference."""
import argparse,json,datetime,os
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=Path('/workspace/herculaneum-lab/reports/fiftysecond-experiment'));args=parser.parse_args();p=args.study;w=p/'work';state=json.loads((w/'RUN_STATE.json').read_text());pid=state['pid'];alive=False
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();fields=raw.rsplit(')',1)[1].split();cmd=Path(f'/proc/{pid}/cmdline').read_bytes();alive=fields[0]!='Z' and fields[19]==state['start_ticks'] and str(p/'worker.py').encode() in cmd
 except FileNotFoundError:pass
 progress=json.loads((w/'inference/PROGRESS.json').read_text()) if (w/'inference/PROGRESS.json').exists() else None
 maps=[]
 for f in sorted((w/'inference').glob('*/COMPLETE.json')):
  r=json.loads(f.read_text());assert r['status']=='PASS' and r['go_sha256']==state['go_sha256'];maps.append(dict(model=r['model'],phase=r['phase'],direction=r['direction'],depth=r['depth'],seconds=r['seconds'],tiles=r['counts']['model_tiles']))
 total=(p/'INFERENCE_RESULTS.json').exists();status='COMPLETE_RECEIPT_PRESENT' if total and len(maps)==32 else 'RUNNING' if alive else 'STOPPED_INCOMPLETE'
 backup=Path('/workspace/scratch/phase52-backup/STATUS.json');saved=json.loads(backup.read_text()) if backup.exists() else None
 print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,worker_identity_live=alive,pid=pid,completed_maps=len(maps),expected_maps=32,committed_batch_receipts=len(list((w/'inference').glob('*/batch-*.json'))),progress=progress,completed=maps,last_recorded_backup=saved,scope='Live process identity and receipt inventory; not a full result/hash verification or a prediction of continued platform availability.'),indent=2))
if __name__=='__main__':main()

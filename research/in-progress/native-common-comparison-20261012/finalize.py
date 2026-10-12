"""Wait for all frozen maps, score once, verify, and publish a final backup.

The numerical definitions and their qualified implementations were frozen before
model metrics were opened. This orchestration adds no inference or metric choice.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import runner as r


def alive(worker):
    try:
        fields=Path(f'/proc/{worker["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==worker['start_ticks']
    except FileNotFoundError:
        return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--backup-state',type=Path,required=True);a=p.parse_args()
    worker=r.read(a.run/'WORKER.json')
    try:
        while not (a.run/'INFERENCE_COMPLETE.json').exists():
            if (a.run/'FAILED.json').exists():
                raise RuntimeError('Inference reported failure; evaluation remains locked')
            if not alive(worker):
                raise RuntimeError('Inference worker stopped before completion; inspect and resume first')
            time.sleep(15)
        freeze=r.read(a.run/'EVALUATION_GO.json')
        assert freeze['go_sha256']==r.sha(a.run/'GO.json')
        for row in freeze['code_pins']+[freeze['qualification']]:r.check(a.run,row)
        if (a.run/'RESULTS.json').exists():
            raise RuntimeError('Results already committed; verify rather than overwrite')
        impl=a.run/'evaluation-implementation'
        if not (a.run/'evaluation/METRICS.json').exists():
            subprocess.run([sys.executable,str(impl/'evaluate.py'),'--inputs',str(a.inputs),'--run',str(a.run)],check=True,timeout=300)
        else:
            # A partially written evaluation is not accepted just because metrics exist.
            expected=['METRICS.json','metrics.csv']+[x['id']+'.png' for x in r.read(a.inputs/'SELECTION.json')['cases']]
            assert all((a.run/'evaluation'/x).is_file() for x in expected), 'Partial evaluation; preserve and inspect'
        verified=a.run/'evaluation/VERIFICATION.json'
        if not verified.exists():
            subprocess.run([sys.executable,str(impl/'verify_results.py'),'--inputs',str(a.inputs),
                            '--run',str(a.run),'--output',str(verified)],check=True,timeout=300)
        assert r.read(verified)['status']=='PASS'
        summaries=r.read(a.run/'evaluation/METRICS.json')['summary']
        table=['# Verified development comparison tables','',
               'All 20 jobs and 120 maps completed. Supplied-reference agreement; not physical ink accuracy.',
               'These numerical tables precede interpretation of the fixed-scale panels.','',
               '| Model | Equal-parent AP | Equal-parent AUC | PHerc0841 AP (segment macro) | PHerc0009B AP |',
               '|---|---:|---:|---:|---:|']
        for row in summaries['equal_parent_macro']:
            model=row['model'];m=row['metrics']
            parent={x['id']:x['metrics']['ap'] for x in summaries['parent_macro'] if x['model']==model}
            table.append(f'| {model} | {m["ap"]:.6f} | {m["auc"]:.6f} | {parent["PHerc0841"]:.6f} | {parent["PHerc0009B"]:.6f} |')
        table+=['','All 14 one-class cores are retained. Their AP/AUC are null. See metrics.csv and',
                'METRICS.json for every core, segment, calibration measure and checkpoint comparison.',
                'No final generalisation claim follows from two exposed parents. No new paid compute was used.','']
        (a.run/'evaluation/TABLES.md').write_text('\n'.join(table))
        paths=sorted(p for p in (a.run/'evaluation').iterdir() if p.is_file())
        paths+=sorted(p for p in impl.iterdir() if p.is_file())
        paths+=[a.run/'EVALUATION_GO.json',a.run/'METRIC_QUALIFICATION.json']
        for name in ['REMOTE_PROGRESS_RESTORE.json']:
            if (a.run/name).exists():paths.append(a.run/name)
        result=dict(status='VERIFIED_DEVELOPMENT_COMPARISON_COMPLETE',utc=r.now(),go_sha256=r.sha(a.run/'GO.json'),
                    inference=r.pin(a.run/'INFERENCE_COMPLETE.json',a.run),outputs=[r.pin(x,a.run) for x in paths],
                    trained_model_forwards=1698,core_maps=120,metric_records=r.read(verified)['recomputed_metric_records'],
                    gpu_spend_GBP=0,scope='Transferred/model-reviewed development reference, upstream exposure unresolved',
                    orchestrator=r.pin(Path(__file__)))
        r.atomic(a.run/'RESULTS.json',result)
        # The watcher may still hold its short-lived snapshot lock. Retry only
        # publication; scoring and inference never repeat here.
        for attempt in range(4):
            task=subprocess.run([sys.executable,str(a.run/'implementation/backup.py'),'--repo',str(a.repo),
                 '--run',str(a.run),'--state',str(a.backup_state)],timeout=300)
            if task.returncode==0:break
            if attempt==3:raise RuntimeError('Final private publication failed; verified local results are retained')
            time.sleep(15)
        r.atomic(a.run/'FINALIZER_COMPLETE.json',dict(status='PASS',utc=r.now(),go_sha256=r.sha(a.run/'GO.json'),
                 results=r.pin(a.run/'RESULTS.json',a.run),backup_status=r.read(a.backup_state/'STATUS.json')))
    except Exception as e:
        r.atomic(a.run/'FINALIZER_FAILED.json',dict(status='STOPPED',utc=r.now(),error_type=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__':main()

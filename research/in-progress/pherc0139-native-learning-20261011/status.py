"""Read-only process identity, local committed progress and backup freshness."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from backup import alive

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
p.add_argument('--backup-state',type=Path);a=p.parse_args()
def optional(path):return json.loads(path.read_text()) if path.exists() else None
worker=optional(a.run/'WORKER.json')
arms=[]
for directory in sorted((a.run/'work').glob('*-seed*')):
    receipts=sorted(directory.glob('step-*.json'))
    if receipts:
        r=optional(receipts[-1]);arms.append(dict(arm=directory.name,step=r['step'],best_step=r['best_step'],
                last_committed_utc=r['committed_utc'],termination=r['termination'],complete=(directory/'COMPLETE.json').exists()))
result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),worker_alive=bool(worker and alive(worker)),
            worker=worker,status=optional(a.run/'STATUS.json'),arms=arms,error=optional(a.run/'ERROR.json'))
if a.backup_state:
    result['backup']=optional(a.backup_state/'STATUS.json')
    result['backup_last_error']=optional(a.backup_state/'ERROR.json')
print(json.dumps(result,indent=2))

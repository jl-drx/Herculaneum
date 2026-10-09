"""Avoid simultaneous multithreaded Torch workloads under the four-core quota."""
import os,signal,time,json
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
while alive(51527):time.sleep(2)
if alive(52792):
    os.kill(52792,signal.SIGCONT)
    (P/'PANEL_RESUME.json').write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),signal='SIGCONT',pid=52792,reason='Native worker exited; panel resumes from unchanged process state.'),indent=2)+'\n')
    print('Panel resumed',flush=True)

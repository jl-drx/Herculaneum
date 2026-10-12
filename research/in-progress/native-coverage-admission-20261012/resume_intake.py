"""Explicitly lower transport concurrency without changing frozen data requests."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import intake


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    record=a.work/'TRANSPORT_PLAN.json'
    if record.exists():raise FileExistsError('Transport attempt already declared')
    history=a.work/'transport-first-attempt';history.mkdir()
    for name in ['ACQUISITION_FAILURE.json','acquisition.log','ACQUISITION_STATUS.json']:
        if (a.work/name).exists():shutil.copyfile(a.work/name,history/name)
    intake.write(record,dict(utc=intake.stamp(),request_plan_sha256=intake.sha(a.work/'REQUEST_PLAN.json'),
        acquisition_implementation_sha256=intake.sha(Path(intake.__file__)),
        transport_implementation_sha256=intake.sha(Path(__file__)),workers=4,
        saved_receipts=len(list((a.work/'receipts').glob('*.json'))),
        reason='Use four concurrent reads. Added-stream initial attempt stopped after repeated HTTP 503; a later 16-byte range on its first unaccounted source succeeded (206, expected full length). Original-stream first transport also uses this lower concurrency.',
        recovery_scope='One explicitly recorded recovery invocation; at most three transient attempts per remaining object in this invocation, same byte budget and URLs, completed receipts reused unchanged',
        scientific_inputs_or_eligibility_changed=False,diagnostic_added_stream_body_bytes=16 if (history/'ACQUISITION_FAILURE.json').exists() else 0))
    original_executor=concurrent.futures.ThreadPoolExecutor
    def bounded_executor(*,max_workers):
        assert max_workers==16
        return original_executor(max_workers=4)
    concurrent.futures.ThreadPoolExecutor=bounded_executor
    intake.acquire(a)
    intake.write(a.work/'TRANSPORT_COMPLETE.json',dict(status='PASS',utc=intake.stamp(),
        transport_plan_sha256=intake.sha(record),acquisition_complete_sha256=intake.sha(a.work/'ACQUISITION_COMPLETE.json')))


if __name__=='__main__':main()

"""External wall/RSS/artifact guard around the frozen, GO-gated Stage B worker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import LIMITS, ROOT, directory_bytes, now, pin, require, save
from prepare_plan import load_go, load_plan


def child_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])*1024
    except FileNotFoundError:
        return 0
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--plan-sha256",required=True)
    p.add_argument("--go",required=True)
    p.add_argument("--output-dir",required=True)
    args = p.parse_args()
    load_plan(args.plan,args.plan_sha256)
    go = load_go(args.go,args.plan_sha256)
    output = Path(args.output_dir).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh Stage B-owned controller output required")
    output.mkdir(parents=True)
    command = [sys.executable,str(ROOT/"worker.py"),"--plan",str(Path(args.plan).resolve()),
               "--plan-sha256",args.plan_sha256,"--go",str(Path(args.go).resolve()),
               "--output-dir",str(output/"acquisition")]
    start = dict(status="RUNNING",started_utc=now(),plan=pin(args.plan),root_go=go,
                 command=command,limits=LIMITS,guardian="Separate process polling wall, worker RSS and artifact size every 0.1 seconds; kill process group on breach.")
    save(output/"controller-start.json",start)
    begun, peak, stop, proc = time.monotonic(), 0, None, None
    try:
        with (output/"worker.log").open("xb") as log:
            env = dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while proc.poll() is None:
                peak = max(peak,child_rss(proc.pid))
                if time.monotonic()-begun > LIMITS["controller_wall_seconds"]:
                    stop = "External wall deadline exceeded"
                elif peak > LIMITS["maximum_rss_bytes"]:
                    stop = "External observed worker RSS exceeded"
                elif directory_bytes(output) > LIMITS["maximum_artifact_bytes"]:
                    stop = "External artifact size exceeded"
                if stop:
                    os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(0.1)
            proc.wait(timeout=5)
            log.flush();os.fsync(log.fileno())
        require(stop is None,stop or "External stop")
        require(proc.returncode == 0,"Worker exited without success")
        require(directory_bytes(output) <= LIMITS["maximum_artifact_bytes"],"Final controller artifact bound exceeded")
        worker = json.loads((output/"acquisition/worker-result.json").read_text())
        require(worker["status"] == "PASS","Worker did not retain a success receipt")
        load_plan(args.plan,args.plan_sha256)
        require(load_go(args.go,args.plan_sha256) == go,"GO changed during controller execution")
        result = dict(status="PASS",worker_result=pin(output/"acquisition/worker-result.json"))
    except Exception as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
        result = dict(status="STOP",failure_type=type(error).__name__,stop_reason=str(error))
    result.update(completed_utc=now(),elapsed_seconds=time.monotonic()-begun,
                  maximum_observed_worker_rss_bytes=peak,artifact_bytes_before_final_receipt=directory_bytes(output),
                  worker_returncode=None if proc is None else proc.returncode)
    save(output/"controller-result.json",result)
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

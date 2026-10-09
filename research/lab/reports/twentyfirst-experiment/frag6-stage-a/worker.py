"""Live Stage A worker, callable only with matching frozen plan and root GO."""
import argparse
import json
from pathlib import Path
import resource
from acquire import run
from common import LIMITS, ROOT, now, pin, require, save
from prepare_plan import load_go, load_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--plan-sha256",required=True)
    p.add_argument("--go",required=True)
    p.add_argument("--output-dir",required=True)
    args = p.parse_args()
    plan = load_plan(args.plan,args.plan_sha256)
    go_pin = load_go(args.go,args.plan_sha256)
    output = Path(args.output_dir).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh Stage A-owned output required")
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS["address_space_limit_bytes"],LIMITS["address_space_limit_bytes"]))
    result = run(output,plan["objects"],pin(args.plan))
    try:
        load_plan(args.plan,args.plan_sha256)
        require(load_go(args.go,args.plan_sha256) == go_pin,"Root GO changed during execution")
    except Exception as error:
        result.update(status="STOP",stop_reason="Post-execution pin verification failed: "+str(error))
        save(output/"STRUCTURAL_CONTRACT.json",result)
        raise
    save(output/"worker-result.json",dict(status="PASS",completed_utc=now(),root_go=go_pin,
         result=pin(output/"STRUCTURAL_CONTRACT.json"),limits=LIMITS,source_pins_reverified=True))
    print(json.dumps(dict(status=result["status"],requests=result["http_attempts"],reserved_bytes=result["reserved_response_body_bytes"])))


if __name__ == "__main__":
    main()

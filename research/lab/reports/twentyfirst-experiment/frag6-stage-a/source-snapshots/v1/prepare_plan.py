"""Freeze Stage A source/fixture pins and validate a separately written root GO."""
import argparse
import json
from pathlib import Path
from common import LIMITS, PROPOSAL, PROPOSAL_SHA, ROOT, now, pin, require, save, sha, verify

CODE_NAMES = ("common.py","parsers.py","transport.py","acquire.py","prepare_plan.py",
              "worker.py","controller.py","check_fixtures.py")


def source_records():
    return [pin(ROOT/name) for name in CODE_NAMES]


def proposal():
    require(sha(PROPOSAL) == PROPOSAL_SHA, "Sealed phase20 proposal changed")
    return json.loads(PROPOSAL.read_text())


def load_plan(path, expected_sha):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected_sha, "Execution plan hash mismatch")
    plan = json.loads(path.read_text())
    require(plan["schema"] == "herculaneum.frag6-stage-a-execution.v1" and plan["stage"] == "A_headers", "Only Stage A execution supported")
    p = proposal()
    require(plan["proposal"] == pin(PROPOSAL) and plan["objects"] == p["present_evidence"]["objects"], "Frozen object/proposal pin mismatch")
    require(plan["limits"] == LIMITS and plan["stage_rules"] == p["stages"][0], "Frozen Stage A rule/limit mismatch")
    require(plan["code"] == source_records(), "Frozen implementation changed")
    receipt = json.loads(verify(plan["synthetic_fixtures"]).read_text())
    require(receipt["status"] == "PASS" and receipt["code"] == plan["code"] and receipt["network_calls"] == 0, "Passing exact-source synthetic fixtures required")
    for record in plan["code"]:
        verify(record)
    return plan


def load_go(path, plan_sha):
    go = json.loads(Path(path).read_text())
    require(go.get("status") == "GO" and go.get("stage") == "A_headers" and
            go.get("plan_sha256") == plan_sha and go.get("allow_live_requests") is True,
            "Matching explicit root Stage A live-request GO required")
    return pin(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures",required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Fresh plan inside Stage A ownership required")
    p0 = proposal()
    receipt_pin = pin(args.fixtures)
    receipt = json.loads(verify(receipt_pin).read_text())
    require(receipt["status"] == "PASS" and receipt["code"] == source_records() and receipt["network_calls"] == 0, "Current-source passing fixtures required")
    plan = dict(schema="herculaneum.frag6-stage-a-execution.v1",stage="A_headers",created_utc=now(),
                proposal=pin(PROPOSAL),objects=p0["present_evidence"]["objects"],stage_rules=p0["stages"][0],
                limits=LIMITS,code=source_records(),synthetic_fixtures=receipt_pin,
                root_live_execution_go_required=True,scientific_payload_decoding_authorized=False,
                interpretation="Literal structures/declarations only; all historical render/physical geometry fields remain unresolved.")
    output.parent.mkdir(parents=True,exist_ok=True)
    save(output,plan)
    load_plan(output,sha(output))
    print(json.dumps(pin(output)))


if __name__ == "__main__":
    main()

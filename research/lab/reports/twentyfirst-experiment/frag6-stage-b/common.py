"""Small standard-library primitives shared by the bounded Stage B reader."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal

ROOT = Path(__file__).resolve().parent
PROPOSAL = ROOT.parent.parent / "twentieth-experiment/frag6-plan/GEOMETRY_ACQUISITION_PLAN.json"
PROPOSAL_SHA = "940973d6fccd51dd9d534341e1329a754f7e85b96bd233438b187b036b4619dd"
STAGE_A = ROOT.parent / "frag6-stage-a/run-v1/acquisition/STRUCTURAL_CONTRACT.json"
STAGE_A_SHA = "73ac68ce326c4a0435564bf7afbc6d24a390ae9e5ae648188b376a1ba0401a62"
STAGE_A_REVIEW = ROOT.parent / "frag6-runtime-review/verification.json"
STAGE_A_REVIEW_SHA = "9c895b7260e47e6057438d44f8a256b02e1adb1ed7aba18d836daf4316ce6dfa"
LIMITS = dict(maximum_attempts=1, maximum_reserved_response_body_bytes=222640,
              expected_parts=243, useful_bytes=34992, stream_chunk_bytes=65536,
              maximum_part_header_bytes=512, maximum_part_framing_bytes=768,
              maximum_response_framing_bytes=1024, connect_timeout_seconds=20,
              per_request_wall_seconds=60, maximum_rss_bytes=268435456,
              address_space_limit_bytes=1073741824, maximum_artifact_bytes=33554432,
              controller_wall_seconds=1800)



class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(record):
    path = Path(record["path"]).resolve(strict=True)
    require(path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "Source pin changed")
    return path


def save(path, value):
    """Atomic JSON replacement with file and directory fsync before returning."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def resources(root, limits=LIMITS):
    rss, disk = rss_bytes(), directory_bytes(root)
    require(rss <= limits["maximum_rss_bytes"], "Measured RSS acceptance bound exceeded")
    require(disk <= limits["maximum_artifact_bytes"], "Artifact bound exceeded")
    return dict(peak_rss_bytes=rss, artifact_bytes=disk)


@contextlib.contextmanager
def deadline(seconds):
    """Absolute main-thread Unix timer includes connection and every read."""
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Nested absolute deadline unsupported")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise ContractError("Absolute per-request deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

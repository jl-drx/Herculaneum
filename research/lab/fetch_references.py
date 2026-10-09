"""Fetch only pinned, checksum-verified public presentation images."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

lab = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination-dir", type=Path, default=lab / "data")
args = parser.parse_args()
args.destination_dir.mkdir(parents=True, exist_ok=True)
manifest = json.loads((lab / "data" / "manifest.json").read_text())
for asset in manifest["assets"]:
    target = args.destination_dir / Path(asset["local_path"]).name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != asset["sha256"]:
            raise SystemExit(f"Existing file checksum mismatch; preserved for review: {target}")
        print(f"Verified retained reference: {target.name}")
        continue
    data = bytearray()
    with urlopen(asset["source_url"], timeout=30) as response:
        if not response.url.startswith("https://"):
            raise SystemExit("Reference download redirected outside HTTPS")
        while chunk := response.read(64 * 1024):
            data.extend(chunk)
            if len(data) > 20 * 1024 * 1024:
                raise SystemExit("Reference download exceeds 20 MiB limit")
    if hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise SystemExit(f"Downloaded checksum mismatch: {target.name}")
    with tempfile.NamedTemporaryFile(dir=args.destination_dir, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(target)
    print(f"Downloaded and verified reference: {target.name}")

# Frag6 structural Stage A

This standard-library implementation follows the sealed phase 20 proposal. It reads only the PPM prefix, classic TIFF headers/IFDs and optional bounded ImageDescription text for the exact retained `result.ppm`, `00.tif`, `32.tif` and `64.tif` objects. It has no image decoder, numerical PPM decoder, Stage B/C runner, model call or paid-resource operation.

The maximum is **13 HTTP attempts and 202,301 reserved response-body bytes**. Every request permanently reserves its exact useful byte count plus one excess-byte allowance, and fsyncs the reservation before connection. Failures retain their reservation and all bytes returned to the reader; partial files are hashed. This is not total wire accounting or a claim that bytes buffered inside a failed HTTP library call can be recovered.

HTTP requests use verified TLS, the exact retained strong ETag in `If-Match`, identity encoding and exact ranges. Responses require HTTP 206 and matching Content-Range, Content-Length, ETag and total size. Duplicate identity/framing headers, redirects, unexpected MIME/encoding or transfer encoding stop before body reads. There are no retries, URL substitutions or full-object fallbacks. A 60-second absolute process timer covers each connection, header exchange and body read; the socket timeout is separately 20 seconds.

The PPM parser recognizes UTF-8 header lines separated by literal LF, versions `1`/`1.0`, ordered six-double records, and the exact extent equation. It preserves comments and stops interpreting bytes at `<>` plus LF. Remaining prefix bytes are retained without numerical interpretation. The TIFF parser handles each file's own II/MM declaration, checks all metadata extents, requires one uncompressed uint16 grayscale strip, and records explicit TIFF defaults. The convention tags require SHORT; dimensions and strip quantities permit their specified SHORT/LONG alternatives. External metadata cannot overlap the header, IFD, pixel strip or another external field. Only a bounded ASCII ImageDescription can trigger an additional request; other external metadata remains unacquired.

ImageDescription text is retained literally. Unknown render fields stay null even if that text names a volume or parameter: root must assess what the declaration establishes. Matching canvases do not prove physical registration. No center plane, depth step, normal direction, render recipe or ink depth is inferred.

`check_fixtures.py` uses synthetic bytes and injected responses with socket-connect tripwires. It covers normal/big-endian/default TIFFs, unsupported metadata, PPM poison tails, pre-body HTTP failures, exact maximum budgets, partial/excess/truncated transfers, absolute deadlines, resource failures and missing/mismatched GO. The original first fixture run and exact source snapshots remain additive records; the final execution plan must pin the latest passing receipt and all eight executable sources.

Prepare a plan after the final fixture run:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python prepare_plan.py \
  --fixtures /ABSOLUTE/final-fixtures/fixtures.json \
  --output /ABSOLUTE/frag6-stage-a/execution-plan.json
```

**No live request may run before independent review and root's written GO.** The GO must contain these exact fields, with the actual frozen plan hash:

```json
{"status":"GO","stage":"A_headers","plan_sha256":"ACTUAL_SHA256","allow_live_requests":true}
```

Root can add its scientific scope and independent-review evidence pins. Both controller and worker recheck the plan, implementation, fixture receipt and GO before acquisition; they recheck after successful acquisition too.

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python controller.py \
  --plan /ABSOLUTE/frag6-stage-a/execution-plan.json \
  --plan-sha256 ACTUAL_SHA256 --go /ABSOLUTE/ROOT_GO.json \
  --output-dir /ABSOLUTE/frag6-stage-a/FRESH_RUN_DIRECTORY
```

The worker enforces 1 GiB address space and checks measured RSS against 256 MiB. The external controller polls worker RSS, total run artifacts and wall time every 0.1 seconds, killing the worker's process group on breach. Limits are 32 MiB artifacts and 1,800 seconds external wall time. Measured RSS and disk checks are acceptance limits, not kernel allocation quotas; the maximum bounded payloads themselves are far smaller. The controller records its own terminal result independently if the worker is killed.

A successful `STRUCTURAL_CONTRACT.json` establishes only the inspected storage structures. A STOP preserves partial observations and never advances to another stage. All writes stay in this new phase 21 directory; prior reports and the tracked repository remain inputs.

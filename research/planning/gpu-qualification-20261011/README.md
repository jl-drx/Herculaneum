# Prospective GPU timing and restart qualification

11 October 2026. **Prepared; no image build/pull or GPU execution has occurred.**
This is a separate engineering benchmark. It cannot resume or replace an arm of
the frozen CPU learning study, and a timing PASS is not a scientific result.

## Fixed first workload

The exact122,513-parameter model and masked BCE function are extracted from the
frozen CPU implementation; `MODEL_RECEIPT.json` binds the source and symbols.
Batch4, all28 input planes, full256 context, central128 scored pixels, Adam1e-3,
gradient clipping5, count-proportional sampling and D4 XY augmentation. Float32,
TF32 disabled, deterministic algorithms required. A missing deterministic CUDA
operation is a failed qualification, not permission to silently relax the rule.
No mixed precision, batch sweep, pretrained weights or held-out inputs.

`benchmark.py` uses eight fixed synthetic examples. CUDA mode runs8 warm-up and
32 timed updates, separates host preparation/transfer/device compute, reports
peak allocated/reserved VRAM, writes a checksummed model/Adam/CPU-and-CUDA-RNG
checkpoint and verifies two resumed steps against uninterrupted continuation.
It also times139 synthetic validation forwards. Component timings synchronize
the device and include that overhead; report complete wall time too. These
measurements do **not** time real chunk assembly, CT I/O, ranking metrics, image
startup, installation or export. Synthetic inputs do not test recoverable ink.

The CPU self-test exercises the same full spatial/depth shapes with a shorter
schedule and exact checkpoint restart. CPU throughput is incidental, measured
while the real CPU study shares the machine; do not use it as a clean hardware
speed comparison. Same-backend restart is bitwise; different CPU/GPU training
trajectories are not expected to be bitwise equal.

## Pinned runtime and invocation

`Dockerfile` pins the linux/amd64 PyTorch2.12.0/CUDA12.6/cuDNN9 runtime by provider
digest. The retained Docker Hub response and its SHA-256 receipt identify the
image; no multi-GB pull is needed here. Pinning is not qualification. Before a
paid allocation, build/resolve it and run the CPU harness in that image, record
installed versions, and verify the provider supports its CUDA driver/runtime.
The final built image and harness file hashes must be recorded. The GPU program
checks its torch/CUDA versions and refuses a silent CPU fallback.

Local harness check, inside this directory:

```bash
python benchmark.py --mode cpu-self-test --output /tmp/plinius-qualification-check
```

On a provisioned, qualified worker, the following command is the fixed benchmark
entry point; it **does not provision a worker or authorize spending**:

```bash
python benchmark.py --mode cuda --output /results/plinius-gpu-qualification
```

Freeze a second short real-training-only I/O pass before execution: use the139
admitted training cores for w035-seed17, exact source hashes/fold/normalization,
same shapes and schedule. Time loading, augmentation, complete steps, actual
training-panel metrics, checkpoint/export, and startup to final verified export.
The synthetic harness is implemented; this real-I/O adapter is pending. Exclude
all current held-out cores/outputs from that worker bundle. Do not treat139
synthetic forwards as that real-data pass.

## Advance, stop and cost decision

First candidate: RTX A5000 24GB at the user's **unverified**$0.27/hour quote.
RTX3090 24GB at$0.50/hour must exceed **1.85185×** A5000 throughput to be cheaper
for a fixed amount of compute with equal non-compute overhead. These are not
current account-specific availability/pricing checks. Four compute hours would
be$1.08 or$2.00; the all-in cost is higher and must be quoted in GBP.

For a first allocation, propose a maximum30-minute billable benchmark envelope,
including setup/export, with an independently enforced deadline and a shorter
process timeout. At the supplied A5000 quote this is$0.135 compute only. Do not
start that clock without enough time for private export and teardown. If setup
cannot fit, revise the estimate before launch. Limit paid attempts rather than
retrying until the£10 total budget disappears.

Advance only if the harness and actual training-only I/O pass succeed, memory
fits with20% headroom, restart passes, complete-job timing materially saves time,
and the all-in remaining allocation fits the user's **£10 TOTAL first-pilot
cap**. No positive AP threshold is required merely to justify timing hardware.
Do not infer sufficient training power from GPU speed or a20,000-step slogan;
the research comparison still needs a frozen objective, exposure, seeds and
reference. A larger model gets a separate memory/timing qualification.

Stop on numerical errors, unsupported deterministic operations, wrong runtime,
OOM/headroom failure, restart mismatch, export failure or watchdog/deadline
failure. Preserve failure artifacts and charge that attempt to the same cap.
Never automatically increase GPU size, reduce depth/context, change precision
or launch a replacement worker.

## Launch prerequisites still pending

- Current account-specific GPU/host/storage charges, billing granularity, tax
  and GBP conversion. Compute+attributable storage/transfer+tax+FX margin+
  failed attempts/retries must remain within the same£10 total. If the remaining
  headroom cannot be bounded, do not launch.
- Pinned-image execution and the real-training-only input/adapter qualification.
- Private worker artifact delivery, with readback SHA/length verification and
  an interrupted-upload/retry check. R2 worker secrets were not imported in the
  last verified handoff; their temporary expiry is15 October09:58:59UTC. Recheck
  status/expiry when needed; never give a worker the administrator credential.
  A separately verified private export path is acceptable.
- An independent external termination controller, tested against the intended
  provider/account and resource ID, plus an in-worker shorter timeout. A local
  shell timeout or the conversational cloud worker alone is not independent.
  Verify termination stops billing and remove any separately billed storage.

No pod, Runpod secret, endpoint, volume, Actions job or paid resource was created.

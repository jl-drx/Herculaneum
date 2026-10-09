# GPU pilot policy

The user approved **£10 total for the first GPU pilot**, including failed attempts, compute, idle time, storage and attributable transfers. This is one aggregate allowance, not £10 for every retry.

`experiments/gpu-pilot.template.json` deliberately has unresolved identities, prices and shutdown details. The validator reports these blockers and never provisions compute:

```bash
herculaneum job validate experiments/gpu-pilot.template.json
```

The required cost estimate includes provider billing granularity, prior spend, noncompute charges, applicable tax, GBP conversion and a positive contingency. Set a current quote expiry and derive runtime from the remaining allowance. Record actual charges and conservative accrued estimates before another attempt.

Before launch, verify a provider-side deadline or an independent control-plane watchdog. Use an earlier compute deadline for a bounded export window, and stop compute at the hard deadline even if uploads fail. Preserve an explicit incomplete state. Verify provider teardown and temporary-storage cleanup; a successful Python exit does not stop Pod billing. Retained storage needs an explicit size and retention limit.

The configuration validator checks declarations; provider access, price accuracy, independent shutdown and input/image checksums still need operational verification. No cloud launcher or verified CUDA image is included yet. The current CPU lock installs CPU PyTorch. Prepare a separate pinned CUDA environment and verify a fixed CPU/GPU comparison before declaring GPU readiness.

Benchmark a useful frozen workload, recording host/VRAM use, network/decode time, throughput and total cost. Select hardware from measurements. H100 is an option when memory, throughput or total cost justify it, rather than a default requirement.

The native PHerc1667 reference plan is distinct from target adaptation. Do not train on today's diffuse target responses. A documented null result remains useful.

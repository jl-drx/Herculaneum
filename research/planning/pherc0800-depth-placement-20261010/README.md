# PHerc0800 bounded depth-placement design

The study52 design and input feasibility are complete; **shifted inference has
not started**. Read the [report](source/RESULTS.md), [frozen protocol](source/PROTOCOL.md)
and [design review](source/DESIGN_REVIEW.md).

- Fixed middle strip: x512:2048/y1024:1536 on segment20251028222030.
- Compare d−15/0/+15, all four models, both padding phases and both orders.
- 786,240 supported pixels; retained inputs suffice with no extra CT download.
- All16 d0 baseline maps reconstruct exactly from1,548 saved batches, without
  model forwards. The earlier checker dtype failure and correction are retained.
- Production design:32 shifted maps /3,096 batches, roughly3h09m CPU plus overhead.
- Stop after the fixed comparison. No brightness-based depth/crop expansion;
  repeatable structured candidates need independent image/geometry review.

[Remaining worker qualification](source/EXECUTION_QUALIFICATION_DESIGN.md) defines
bounded tests before execution. No worker, GPU job, Actions run or spare-VM task
was launched for this design. GPU spend remains£0.

[MANIFEST.json](MANIFEST.json) identifies exact source copies and exclusions.
The local study is `/workspace/herculaneum-lab/reports/fiftysecond-experiment`.
Scripts preserve original absolute prior-study paths and require their pinned
scientific environments and retained inputs; they are write-once preparation
and replay helpers, not standalone installation instructions. Do not run them
against an existing completed design directory. Restoring historical data uses
the separate [study51 checkpoint instructions](../../in-progress/pherc0800-development-20261010/README.md).
The cumulative formal seal remains at study50; this design is not a newly sealed
research result or an ink-accuracy benchmark.

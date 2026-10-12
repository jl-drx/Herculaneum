# Native common-support comparison

Started 12 October 2026, 15:43 UTC. This page records the frozen launch; see the
completion report when available for outcomes. No accuracy result is claimed here.

The [protocol](PROTOCOL.md) compares Reader v2, dense-native, seed42, seed43 and
Hecate on the same 24 native cores from PHerc0841 and PHerc0009B. The admitted
[input panel](../native-common-support-20261012/SUPERVISION_PANEL.md) contains
232,250 score pixels and 20,137 supplied positives. These are exposed development
references with transfer and provenance limitations. All 14 one-class cores stay
in the analysis; their AP and AUC are undefined. Confirmation families stay closed.

[Qualification](QUALIFICATION.json) passed synthetic full/sparse original-pipeline
parity at both spacings, exact normalization of all 216 real hybrid patches, and
cache interruption/rejection tests. There were no trained-model forwards in those
checks. [GO.json](GO.json) pins weights, runtime, input receipts and code. Its SHA256
is `2dcc6c78d6ccb2f129ccb70b01cfa29a069664a8576d1eafed666c8cef452899`.

The run uses CPU fp32, batch one and four threads. It has 1,698 unique forwards,
20 model/segment jobs and 120 core maps. Each logit tile is committed with hashes
before reuse. Every completed job must replay exactly without another forward.
An injected exception after the first two Reader forwards checks actual resumption;
this is not a process-crash or lost-success-response test. Metrics remain unopened
until all jobs finish. Initial runtime estimate: 40–60 minutes, subject to measured
throughput. No paid compute, Actions job or PR is involved.

## Execute or resume

The frozen input archive is
`research/workspace/bundles/native-common-support-inputs-20261012.tar.gz`, SHA256
`b6632e6798d808af7252cc372f9e567d3338adba947e64e54755c895af236968`.
Its own README explains restoration. Model weights are separate; use GO identities.
The exact original runtime source is retained in this repository. Runtime versions
and packages are in GO. The worker checks inputs, checkpoints and source hashes.

```bash
/workspace/.venvs/pherc343/bin/python \
  /workspace/herculaneum-lab/reports/native-common-comparison-20261012/implementation/runner.py \
  --repo /workspace/Herculaneum \
  --inputs /workspace/herculaneum-lab/reports/native-common-support-20261012/supervision-pilot \
  --run /workspace/herculaneum-lab/reports/native-common-comparison-20261012
```

`STATUS.json` is a timestamped progress observation, not proof of a live process.
Check `WORKER.json` against PID and `/proc` start ticks before claiming the worker
is alive. `INFERENCE_COMPLETE.json` is written only after all 20 jobs pass. A
`FAILED.json` requires inspection before restarting; never treat incomplete output
as a finished comparison. The runner holds an exclusive lock against duplicate jobs.

## Private restart backup

`backup.py` publishes append-only verified generations to the private branch
`codex/plinius-native-common-comparison-checkpoints`, using Git/LFS directly and
leaving the research checkout untouched. It checks private repository metadata
before every upload. The five-minute interval preserves committed tile progress;
it cannot preserve a live process or ensure the cloud environment remains running.

Use the generic checksum-checking
[restore script](../pherc0139-native-learning-20261011/restore_backup.py) on a checkout
of that backup branch after fetching only its listed LFS objects. Restore to a new
directory, restore the input bundle separately and provide the pinned model/runtime
prerequisites. GO uses the original absolute checkpoint paths; reproduce those paths
or document and qualify a portability amendment before resuming elsewhere. The
backup branch's INDEX lists every member hash and archive hash. Restore rejects
changed existing files. Credentials are never included.

```bash
python research/in-progress/pherc0139-native-learning-20261011/restore_backup.py \
  --backup /path/to/backup-branch --destination /path/to/restored-run
```

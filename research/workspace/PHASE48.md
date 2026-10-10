# PHerc0800 support check and spare-VM handoff

[Experiment48](../lab/reports/fortyeighth-experiment/RESULTS.md) completed the
full-resolution support calculation on all five retained larger meshes. No
128×128 core passed the frozen conservative criterion; the largest was 69×69.
This is not an absence-of-ink result or proof that exact scheduled context is
infeasible. The next primary task is development-transfer preparation on an
explicitly reused target field.

The compact [independent VM assignment](../handoffs/pherc0800-support-20261010/START_HERE.md)
contains raw coordinate TIFFs, metadata, archived point checks, contracts,
hash-locked dependencies and verification instructions. The private manual
workflow `package-pherc343-review.yml`, with input `packet=pherc0800-support`,
packages that task using `contents: read`, no persisted checkout credentials,
zero LFS fetches and one-day artifact retention. The operational run receipt is
under [handoff operations](../handoffs/pherc0800-support-20261010-operations/).
No research or model execution happens in Actions.

## Dense research supplement

The independent VM task does **not** need this larger supplement. It preserves
the sealed dense masks/distances and original source/report evidence for audit:

```bash
git lfs pull --include="research/workspace/bundles/pherc0800-support-20261010-001.tar.gz" --exclude=""
```

Compressed size: **74,849,090 bytes** (71.4 MiB). Expanded size: **86,775,481 bytes**
across 26 files. Archive SHA256:
`711ff9ccf4e5b304dd2bd8eb3342aeec2143fce710edb0b917480881dc8cb7b5`.
[PHASE48_BUNDLE.json](PHASE48_BUNDLE.json) lists every member and checksum;
[PHASE48_VALIDATION.json](PHASE48_VALIDATION.json) records local verified readback.
The old workspace snapshot and earlier supplements are unchanged.

The current cumulative seal is
`d81d1eb93b9566f0c2f2a1b6475090faad3493c1da2c1cd7f1fb477680ba50e5`,
with 25,886 records / 14,488,524,801 retained bytes. Its full lab readback passed.
Register [v44](../lab/reports/research-register-v44.json) covers 48 studies.

```bash
/workspace/.venvs/herculaneum/bin/python -B \
  /workspace/herculaneum-lab/reports/fortyeighth-experiment/verify_retained_state.py \
  --expected-manifest-sha256 d81d1eb93b9566f0c2f2a1b6475090faad3493c1da2c1cd7f1fb477680ba50e5
```

That check is for the retained complete lab, not just this supplement. Do not
run write-once historical producers in sealed directories. The separate VM task
requires an independent implementation and reports failures as well as passes;
no physical second-device computation is claimed by the packaging receipt.

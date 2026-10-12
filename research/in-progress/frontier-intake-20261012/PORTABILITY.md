# Selective private restoration

The input bundle is about **135.5 MiB compressed / 145.7 MiB expanded**. It contains
the twelve assembled native CT contexts and two adjacent legacy reference cubes
with CT, instance labels, receipts and a checksum manifest. It omits the redundant
648 MiB raw CT chunk cache and public model weights. Code, reports and panels are
ordinary Git files; detailed public-source/CT receipts are in `source_receipts.zip`.

Use the existing research branch and authenticated private GitHub access. Clone
with `GIT_LFS_SKIP_SMUDGE=1` or fetch the branch without an unrestricted LFS pull.
From the repository root:

```bash
git lfs pull --include='research/workspace/bundles/frontier-intake-contexts-20261012.tar.gz' --exclude=''
python3 research/planning/pherc0139-paired-packet-20261011/restore_packet.py \
  --bundle research/workspace/bundles/frontier-intake-contexts-20261012.tar.gz \
  --manifest research/in-progress/frontier-intake-20261012/BUNDLE.json \
  --destination "$PWD/.local/frontier-intake"
```

The standard-library restorer checks the complete archive and all 22 files,
rejects unexpected paths, links and altered bytes, and requires a new destination.
It starts no worker and reads no current native0139 held-out result. A local
restoration PASS and any later real remote-download verification are recorded
separately; consult their receipts rather than assuming an upload worked.

For CPU analysis, the existing tested inference environment can be recreated with
`bash research/tools/setup_portable_cpu.sh pherc343`. The package versions used
here are recorded in `ENVIRONMENT.json`; no new package was installed in the
cloud task. With that environment:

```bash
.venvs/pherc343/bin/python research/in-progress/frontier-intake-20261012/geometry_reference.py \
  --inputs "$PWD/.local/frontier-intake/instance-reference" \
  --output "$PWD/.local/frontier-reference-replay"
.venvs/pherc343/bin/python research/in-progress/frontier-intake-20261012/verify_reference.py \
  --inputs "$PWD/.local/frontier-intake/instance-reference" \
  --primary "$PWD/.local/frontier-reference-replay/RESULTS.json" \
  --output "$PWD/.local/frontier-boundary-replay.json"
```

These commands are reference-only CPU diagnostics, not model training or a new
native surface benchmark. Outputs require unused paths. The full input/array
hashes are checked before the primary computation; the follow-up rechecks its
arithmetic using the same declared reference.

Reader v2 remains available from its immutable Hugging Face revision and published
SHA. Fetch only that checkpoint, then optionally repeat the synthetic CPU check:

```bash
python3 research/in-progress/frontier-intake-20261012/acquire_sources.py \
  --plan research/in-progress/frontier-intake-20261012/source-plans/reader-checkpoint.json \
  --output "$PWD/.local/reader-v2"
.venvs/pherc343/bin/python research/in-progress/frontier-intake-20261012/qualify_reader.py \
  --checkpoint "$PWD/.local/reader-v2/reader-v2-step040000.raw" \
  --runtime research/lab/reports/fortysixth-experiment/runtime-source \
  --output "$PWD/.local/reader-v2-qualification.json"
```

The `.raw` suffix preserves downloaded bytes; the file is the original PyTorch
checkpoint, not a conversion. Deserialization uses `weights_only=True`, and
strict loading must succeed. This command uses synthetic input only; do not use
the reserve or locked evaluation for an improvised smoke test.

`source-plans/restore-original-payloads.json` is an optional hash-bound way to
retrieve all 328 original CT/reference payloads. It requires about 713 MiB of
network transfer and is unnecessary when using the small context bundle. Mutable
upstream bytes that change fail verification rather than being silently accepted.
For offline metadata replay, verify/extract `source_receipts.zip` against its
manifest, then invoke `build_intake.py` with its `sources/` and `prior-sources/`
directories and a fresh output directory. The context assembler additionally
needs the optional original raw chunks; ordinary scientific review does not.

No GitHub Actions job or spare-VM assignment is required for this packet.
Do not put source data, derived label images or newly revealed text in a public
repository; preserve the source-specific notices and research scope.

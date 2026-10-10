# PHerc0800 independent audit and fixed regional comparison

[Study50](../lab/reports/fiftieth-experiment/RESULTS.md) reproduces the returned
independent audit and admits a **128×128 core, x1024:1152/y280:408**, on segment
20251028225813 under every required model/phase footprint. The original192-square
passes its declared single grid; padded Villa context requires the smaller core.
Both arithmetic implementations and the updated exposure inventory agree.

Actual66-plane rendering and all16 fixed-centre maps pass execution checks.
The maps took6.46min on CPU and replayed exactly from288 committed forward logits.
Padding and order sensitivity remain substantial; no defensible lettering or ink
accuracy is established. The region is now review-exposed, not a pristine test.
The [larger development field](../lab/reports/fiftieth-experiment/NEXT_STUDY_DRAFT.md)
is next; no worker is currently active.

## Download and verify saved outputs

From this private research branch, fetch only the new165.4MiB supplement:

```bash
git lfs pull --include="research/workspace/bundles/pherc0800-independent-candidate-20261010-001.tar.gz" --exclude=""
python3 research/tools/replay_phase50.py --destination /tmp/plinius-phase50-replay
```

Use Python with NumPy and a fresh destination. This verifies the archive and all
1,024 members, the study seal,107 original audit payload checksums,
33 raw-source chunks,6,745,728 saved render-sample comparisons,16 completed maps,
and288 committed-logit receipts. It checks the recorded framework replay receipts;
it does not rerun model inference or independently regenerate those logits.
No Torch, network access, credentials or paid job is needed for this saved check.
A fresh-directory replay passed on the cloud host in the minimal NumPy/SciPy
environment. This is not a separate physical-device inference test.

[PHASE50_BUNDLE.json](PHASE50_BUNDLE.json) lists 173,422,658 compressed bytes,
269,353,173 expanded bytes and every member hash. Archive SHA256:
`d837c483f97f9f3fd40b86b3221126b9949dea79989b9588dd2cc53d08e91b23`.

The bundle preserves the uploaded ZIP, extracted audit, complete audit replay,
source chunks, rendered arrays, model inputs, logits, maps, protocols and receipts.
Prior checkpoint weights and upstream software remain in earlier groups. Partial
virtual canvases explicitly contain unknown regions and are valid only for their
admitted patch footprints. Never use them as full target CT. Do not rerun original
write-once producers inside this sealed evidence; see the study PORTABILITY.md.

Current whole-lab seal:
`a0c640dbebe0ce48af00edb2aaaeeab2e21c16376afac65e43fd65fe2376a5ca`,
27,035 records /14,815,313,737 bytes; register v46 covers50 studies. Full retained
readback passed. [PHASE50_VALIDATION.json](PHASE50_VALIDATION.json) distinguishes
local restore and actual remote-download checks. Historical bundles are unchanged.

Short checks stay in this chat. Spare-VM work is useful when a local hour-plus run
or similar occupation creates worthwhile parallel work; study50 needed no new
Actions dispatch or VM assignment. GPU spend remains£0.

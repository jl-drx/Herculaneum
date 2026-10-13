# Portable saved-map replay proof

Final result: **PASS**, 11 October 2026, 09:04 BST (08:04 UTC).

The final read-only `../../verify_saved_maps.py` accepts `--packet PATH`, defaulting to its own directory. It reads only saved packet data and writes JSON to stdout. All 55 frozen input pins must bind uniquely to manifest-relative paths, with exact lengths and SHA-256 hashes. It rejects path traversal, absolute or symlink input paths, duplicate bindings and unexpected panel filenames. It never falls back to original absolute paths.

Both the normal and relocated runs verified all 48 maps, 168 distinct comparisons, 672 support-scope scores, 64 exact panels and 48 contact tiles. The unchanged maximum errors were 3.33e−16 for Pearson and 1.42e−14 uint8 levels for MAD.

The relocated test copied only the files needed for report replay: 128 files, 189,319,787 bytes, including 48 aligned-map NPY files and one support NPZ. It included no model weights, CT or 878 MB checkpoint. A Python audit hook allowed reads only below the relocated packet, the virtual environment and base Python runtime. It rejected writable opens and listed filesystem mutation events. The observed accesses were 362 packet opens and 19 runtime opens, with no denied access or filesystem mutation. Stdout was captured by the harness through a file opened before the guard was enabled.

**Scope:** the hook tests Python audit events; it is not OS filesystem isolation and does not intercept arbitrary direct C syscalls. Its precise allowlist and policy are retained in `RELOCATION_GUARD_PROOF.json`; `guarded_replay.py` records the test harness. No claim of ink accuracy or authenticated text follows from correct reporting.

Final verifier SHA-256: `89defa798b34c9d3d3a766d529eb63a8ad057e4b475976554984aa2635f033b4`.

`RESULT.json` and `RELOCATED_RESULT.json` are the final proofs. `attempt-1/` preserves an earlier successful numerical replay whose panel path handling was tightened during review before final acceptance. Those earlier receipts are historical attempts, not the final accepted verifier. Original generation and v1/v2 independent-reporting evidence were not changed.

`PROOF_MANIFEST.json` covers this proof directory; the parent portable manifest covers the whole packet and the final verifier.

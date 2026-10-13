# Portable native-reference packet

The small analysis packet contains source metadata, six annotation shards, protocols, receipts, cohort lists, code, checks and validity masks. Native CT source chunks are a separate archive to avoid duplicating approximately 235 MiB in every analysis download. Restore both into one packet tree so a CT request's `relative_path` resolves beneath `ct/`, for example `ct/chunks/w035/0/<chunk_y>/<chunk_x>.bin`. Keep the original file paths and frozen files unchanged.

An environment with Python 3.12 and the exact packages in `requirements.txt` ran the saved checks; `ENVIRONMENT.json` records versions. CPU execution is sufficient. Neither PyTorch nor model weights nor GPU access are required. The largest annotation array is66,083,200 uint8 pixels; decoders and histograms use additional temporary memory. A9 GiB CPU VM is sufficient for this bounded replay. CT storage is245,891,072 bytes plus the small reports/masks; optional extracted-array copies are unnecessary.

From the packet directory:

```bash
python -m venv /workspace/native-reference-venv
/workspace/native-reference-venv/bin/python -m pip install -r requirements.txt
/workspace/native-reference-venv/bin/python replay.py --output /workspace/native-annotation-replay
/workspace/native-reference-venv/bin/python ct/replay_ct.py --output /workspace/native-ct-replay
```

Choose new output directories. The replay scripts validate frozen hashes and local receipts, decode the six annotation arrays with both official Zarr and an explicit index/numcodecs assembler, reproduce every frozen grid/core/class count, and verify all CT objects and per-core support masks. They do not fetch inputs. `intake.py` and `ct/intake_ct.py` retain original acquisition implementations for audit, but ordinary validation should use the offline replay entrypoints above.

The geometry audit is separate under `geometry/`; follow its own replay instructions. `aggregate_geometry_classes.py` joins geometry-selected core IDs with already decoded annotation counts and CT support. It cannot choose new cores from class counts. Its saved output pins the geometry result hash and must be regenerated explicitly if that upstream geometry result changes.

Publication/archive checksums and any archive restorer supplied by the root research workflow are additional to the scientific receipts. Successful replay demonstrates saved-byte and calculation reproducibility; it does not establish physical ink accuracy, private historical annotation-transfer correctness or upstream model independence.

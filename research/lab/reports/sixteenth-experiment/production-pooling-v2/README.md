# Offline production CT pooling

Root executes only after all 1,530 pixel requests pass and the source-bound pooling GO is saved. The preparer reads/hash-verifies metadata and source records; it never decodes CT. The worker validates all 153 TIFF layouts, full reviewed metadata/pixel plan lineage, global success inventory and every terminal/config/HTTP/body/range/hash chain before its first numeric sample decode.

Every source row uses that TIFF object's explicit <u2 or >u2 dtype. Pooling is exactly native uint32 sum of 27 values, float64 division by 6939, then one native float32 cast. The four literal fifteenth label/mask records per context are referenced unchanged. No model, network, checkpoint or threshold work occurs.

The entire producer package and this pooling package/output count toward 2 GiB. The worker enforces 120 s stage wall, 360 s CPU, 3 GiB RSS, 12 GiB address-space and the shared M/P/C 1500 s epoch window. It stages final descriptor bytes and publishes the canonical descriptor last, after accounting actual written outputs. Failures leave no canonical descriptor. Independent byte-exact CT verification remains required before inference.

Root commands (replace the GO path with the reviewer's exact receipt):

```sh
/workspace/.venvs/herculaneum/bin/python prepare_pooling_plan.py --raw-provenance ../production-acquisition/pixel-run/raw-provenance.json --source-go ../review/production-pooling-v2/code-go.json --output-dir final-pooling-plan --ready
/workspace/.venvs/herculaneum/bin/python run_pooling.py --plan final-pooling-plan/pooling-plan.json --output-dir pooling-run
```

No real CT/sample arrays were read during preparation. The earlier provenance fixture setup failure was a synthetic contract missing height/strip_bytes; final corrected fixtures pass and do not conceal any scientific execution. Kernel/provenance fixture receipts retain earlier versions.


This additive V2 preserves the original stopped package. Its sole execution-method correction projects reconstructed integer TIFF tag dictionary keys to JSON string keys for exact layout comparison. Tag values, byte order, strip bounds and pooling are unchanged. The original STOP occurred before any numeric sample decoding. V2 remains subject to the original production epoch/deadline.

Correction addendum: ../POOLING_JSON_CORRECTION.md, SHA256 fab8bc5d5ec205b97d47f8c05c78e05dd31006744c8119c52e9e52501cc23625. Original stopped pooling history is also included in combined production storage accounting.

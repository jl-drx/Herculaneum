# Portable independent review replay

`recheck_portable.py` duplicates the immutable `recheck.py` calculations and only adds explicit input/output handling, a source-hash check and a separate provenance receipt. It imports no intake implementation and performs no acquisition.

```bash
python recheck_portable.py --packet /path/to/restored/native-reference-packet --output /path/to/new-review-output
```

The output directory must not exist and must be outside the input packet. The packet remains read-only. The original checker, original results and original checksum manifest are unchanged; this new file is covered by the parent's later archive seal.

Original checker SHA256: `6799ebe7e6089abe96c6a9d8f063be143c1886aeecfb0e4fab647653b7415660`.

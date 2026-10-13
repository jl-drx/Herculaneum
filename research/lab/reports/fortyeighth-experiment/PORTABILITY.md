# Phase48 portable record

The selected source/report export preserves the frozen protocol, GO input pins,
geometry code, results, numerical verification and figure. A separate private
Git LFS bundle retains dense masks/distances and cumulative seals for readback.
Historical source coordinate arrays remain in the original workspace snapshot.
Do not run the write-once producers inside this sealed directory.

The compact spare-VM handoff is
research/handoffs/pherc0800-support-20261010/ in the private research branch. It
contains raw source coordinate TIFFs and metadata, archived point receipts,
pre-existing component labels, contracts, an independent assignment, expected
summaries, hash-locked dependencies and manifests. It needs no model or intensity
volume. Its Actions job packages only these small ordinary Git files, so it
fetches zero LFS objects, uses read-only repository permissions and expires the
artifact after one day. The retained Git packet supports later re-packaging.

The VM assignment is intentionally an independent implementation, not an
unqualified promise that the historical absolute-path producer is portable.
Package checksum checks passed locally; no separate physical-VM execution is
claimed. Actual Actions download verification is an operational repository
receipt and cannot be inferred merely from a local packaging test.

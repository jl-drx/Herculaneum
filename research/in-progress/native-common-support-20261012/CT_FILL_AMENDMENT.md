# Sparse native-store handling — before CT payload acquisition

The first planned native CT GET returned HTTP 404; no body was consumed or
accepted. The original strict acquisition stopped. Complete, bounded S3 prefix
inventories then identified absent chunks in these Zarr v2 stores. Their pinned
array metadata declares fill_value=0. The inventories and metadata, not a 404 by
itself, establish canonical array fill. They cannot establish why CT is absent or
that a physical papyrus location is blank.

Keep FREEZE.json, SELECTION.json and CT_PLAN.json unchanged. No core, threshold,
depth, grid or scoring rule changes. For every selected key, parse and hash-check
the complete listing pages: acquire listed chunks with their exact size and
If-Match ETag; materialize listed-absent chunks as declared fill, with a separate
canonical-fill receipt. Never silently turn a denied, failed or missing listed
object into zeros. The original failed-attempt reservation remains charged.

Array-context completeness now explicitly includes canonical fill. Record its
fraction separately from retrieved CT, and retain the fixed centre-plane >0
score mask. A core with no scored pixels or one reference class remains in the
cohort with undefined metrics; no replacement. This is a storage correction,
not a geometry repair or evidence that a region has no ink.

The amended acquisition records a new execution-code hash before continuing.
Only this fill-aware wrapper resumes CT; the original command remains preserved
as the failed implementation. No inference has run.

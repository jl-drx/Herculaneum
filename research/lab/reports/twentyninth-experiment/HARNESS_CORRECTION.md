# Qualification harness correction, before attempt2

Attempt1 preserved:26tests,25passed,1error; all16adaptertests passed. The remaining test failed at construction of Zarr's local asyncio event loop, before any fixture was written. The audit hook denied socket.__new__ for its AF_UNIX socketpair. This was local inter-thread plumbing, not an internet request. The captured traceback and denied event are in run-v1.

Attempt2 changes only the driver: allow AF_UNIX socket construction; continue to reject all other socket audit events (including connect/bind/DNS), IP socket construction and subprocess/system launch. Route the same tests to a new run-v2/fixtures directory. No change to reference_metrics.py, test_reference_metrics.py, upstream_extract.py, synthetic contract, acceptance tolerances or source files. The existing failure is not overwritten. Scope remains offline fixed synthetic tests; do not claim OS-level isolation or that audit hooks intercept arbitrary native code.

# Preparation failure retained

The first panel worker stopped in the existing checkpoint helper's `verify_artifact` before model construction or a model forward: `ContractError: Unknown checkpoint role`. It supplied the analysis label `seed42-75000`, while the frozen historical helper accepts `seed42` or `seed43` for those exact final artifacts.

The additive v2 worker passes the canonical historical role to that unchanged helper, retaining the descriptive analysis label, exact artifact and all comparison settings. GO_V2 pins the corrected code and states this amendment. The original script, GO and runner log are preserved. No scientific output was used to make the correction; no checkpoint was replaced and no numerical contract was weakened.

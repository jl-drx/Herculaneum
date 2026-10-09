# Independent reviewer schema clarification v2

The preserved training-v1 review stopped on an exact metadata-schema comparison before reading new phase18 logit outputs. The final frozen producer includes `all32_cached_add_zero_bitwise_identity: true` in `initialization_proof`, in addition to the fields expected by the first checker. This is a stronger identity receipt, not a scientific or numerical failure.

`verify_outputs_v2.py` preserves the original checker and adds that exact required Boolean plus an independent bitwise check of all32 cached teacher batches after adding float32 zero. Protocol, input plan, run, objective, tolerances, optimizer, gates and model outputs remain unchanged. No model is rerun. The failed attempt and initial source remain retained.

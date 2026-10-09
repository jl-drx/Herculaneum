# Addendum to the earlier methods audit

The previous methods audit is preserved unchanged. Two observations from this experiment affect its follow-up decisions.

**Checkpoint size is not an architecture diagnosis.** The four actual PyTorch state dictionaries have 68,175,426 named scalar entries (272,701,704 bytes) but only 34,546,498 unique parameters (138,185,992 storage bytes), because 244 groups duplicate tensor views under different names. All exactly match the unchanged model factory and load strictly. The named-entry count equals the 68,175,426 figure noted in the earlier Nieuwlaar metadata audit. This makes alias/serialization accounting a concrete possible explanation of that apparent discrepancy. Nieuwlaar's native checkpoint header, tensor schema, configuration and strict loading remain unverified; this result does not establish compatibility. A later audit should check them directly instead of treating total named bytes as proof of a different architecture.

**The new dense checkpoint is a useful negative control here.** On this one fixed corrected w029 crop it underperforms its matched control in both threshold-independent ranking and the prespecified midpoint diagnostic. More supervision is not an automatic upgrade. The next scientifically useful question concerns when coverage, teacher error and acquisition domain affect generalization, with independent physical regions and controlled training comparisons. This does not establish a general failure of dense pseudo-labeling.

These findings do not alter source permissions or establish independent labels, a new method, a manuscript reading or prize eligibility.

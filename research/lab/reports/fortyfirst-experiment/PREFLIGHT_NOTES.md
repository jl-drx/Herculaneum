# Preflight correction before GO

The first eight-fixture run failed two strict bitwise assertions before any model call or GO. Sum-reduction then division changed original-slot BCE gradients by an FP32 ULP (maximum observed 1.82e-12). The implementation now preserves the historical joint-mean backward exactly on original-slot logits, detaches new-slot logits, and substitutes the directly computed retained sum/N as the scalar value. This is algebraically the same prescribed objective and unchanged original denominator.

The second fixture compared the objective at teacher=student to BCE alone with exact-zero tolerance. The unchanged FP64 KL expression has cancellation roundoff (maximum observed1.36e-20). The analytical comparison now allows1e-18 absolute tolerance and2e-7 relative tolerance for FP32 mean-versus-sum arithmetic; the separate original-slot gradient comparison against full phase40 remains bitwise exact. The independent full-gradient decomposition remains1e-6 relative/2e-11 absolute. Original failed code and receipt are retained under preflight-v1. No scientific training attempt or retry occurred.

Inspection also found that the historical control receipt contains extra diagnostics absent from the six-key historical objective interface. The verifier compares the six shared original-support terms explicitly. This correction also precedes GO and execution.

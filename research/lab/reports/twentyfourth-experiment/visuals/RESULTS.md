# Phase24 fixed depth-permutation panels

The frozen v2 plotter completed one rendering run after root released both actual reverse-result artifacts. It produced exactly four figures:

- [Seed42 / Paris1](render-v1/seed42-paris1-depth-reversal.png)
- [Seed42 / Paris2](render-v1/seed42-paris2-depth-reversal.png)
- [Native / Paris1](render-v1/native-paris1-depth-reversal.png)
- [Native / Paris2](render-v1/native-paris2-depth-reversal.png)

Each model includes all six complete 384² contexts. Columns show supplied labels, forward-order raw scores, reversed-order raw scores and reverse-minus-forward differences. The original primary mask is identical across each row, with gray outside it. Scales remain 0..1 and −1..1 across all cases and both models. The captions identify the intervention as a normalized depth-order permutation and mark the figures as reused development.

All 60 consumed arrays passed exact file/raw-array hashes, shape/dtype, original support-count, full evaluated-coverage and finite score-range checks. Reverse metadata explicitly retained `adapter_read_direction="forward"` and permutation `[16,...,0]`. The exact result/input/protocol/code and image identities are recorded in [PLOT_RECEIPT.json](render-v1/PLOT_RECEIPT.json); all bound source hashes were checked again after rendering.

The four PNGs and receipt total **3,929,726 bytes**. The command exited zero without a rerender or source change. Layout inspection covered all four figures and found complete panels, readable case/model/parent labels, fixed colorbars and the gray-support legend. No figure, crop, scale or region was selected based on outcomes. Root retains scientific interpretation.

No model, checkpoint, CT surface, network request, fitted threshold or new scientific metric was involved in plotting. Earlier preparation versions and both freezes remain unchanged, documenting the prospective replacement of the ambiguous `direction` check with the explicit normalized-input intervention fields. Prior phase23 evidence remains untouched.

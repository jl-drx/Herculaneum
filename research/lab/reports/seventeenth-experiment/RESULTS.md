# Plinius regional alignment and depth audit

The audit found no clear coordinate, decoding or depth-range fault that explains the weak Paris2 regions. Exact CT reconstruction, strong infrared–label agreement and consistent sampled surface geometry all passed their respective checks. Small CT-to-infrared registration errors and the precise ink-bearing depth remain unmeasured. The original model comparison and its failures remain unchanged; this is an evaluation-informed diagnostic, not a new generalization result.

![Six-region comparison and complete acquired depth profiles](overview-v2/regional-overview.png)

All six preselected regions were retained. No model was rerun, trained or promoted; no flip, shift, crop, depth window or threshold was optimized. The published infrared images and fixed sparse coordinate-map samples required 22 public GETs including the historical reader source, totaling 41,827,766 response-body bytes. No full coordinate map or new CT scan was downloaded. GPU spend remains £0, with the £10 total first-pilot cap unchanged.

The table compares the saved seed42 CT scores with two predeclared optical diagnostics on the same primary support. Raw infrared darkness is 255 minus the exact 3×3 mean gray image. Local darkness subtracts the image from a fixed Gaussian-smoothed image, sigma 16 native pixels. Labels were traced from registered infrared, so high optical agreement is expected and is not independent evidence of CT registration accuracy.

| Fixed region | Saved CT AUROC | IR darkness AUROC / AP | Local IR darkness AUROC / AP | Fixed panels |
|---|---:|---:|---:|---|
| Frag1 context1 | 0.446052 | 0.977097 / 0.937864 | 0.554040 / 0.232015 | [CT](implementation/ct-run/frag1-context1-ct-panel.png), [IR](implementation/ir-run/frag1-context1/ir-panel.png) |
| Frag1 context2 | 0.903678 | 0.997953 / 0.992570 | 0.584576 / 0.165217 | [CT](implementation/ct-run/frag1-context2-ct-panel.png), [IR](implementation/ir-run/frag1-context2/ir-panel.png) |
| Frag2 context1 | 0.501298 | 0.959325 / 0.923443 | 0.593617 / 0.301261 | [CT](implementation/ct-run/frag2-context1-ct-panel.png), [IR](implementation/ir-run/frag2-context1/ir-panel.png) |
| Frag2 context2 | 0.779541 | 0.989758 / 0.980628 | 0.610593 / 0.279464 | [CT](implementation/ct-run/frag2-context2-ct-panel.png), [IR](implementation/ir-run/frag2-context2/ir-panel.png) |
| Frag3 context1 | 0.870064 | 0.969733 / 0.839913 | 0.530114 / 0.098466 | [CT](implementation/ct-run/frag3-context1-ct-panel.png), [IR](implementation/ir-run/frag3-context1/ir-panel.png) |
| Frag3 context2 | 0.699512 | 0.984355 / 0.878978 | 0.565757 / 0.066987 | [CT](implementation/ct-run/frag3-context2-ct-panel.png), [IR](implementation/ir-run/frag3-context2/ir-panel.png) |

Frag1/2 belong to one evaluation parent, Paris2; Frag3 is Paris1 calibration material. The two weak CT contexts still have strong raw infrared association. Root reviewed all six CT and six infrared panels: the released label shapes visibly follow the registered optical strokes at the fixed coordinates, with no gross raster swap apparent. This weakens an obvious label-file/crop mismatch explanation. It does not measure the original deformable CT-to-optical warp, because the registered infrared and its labels share that processing history. Independent non-ink landmarks or a documented warp/control-point export are still needed for a residual in pixels or micrometers.

The fixed local-darkness diagnostic was weak in every region and is retained without retuning. Its filter removes much of the broad optical stroke interior as well as slow background variation. That is a warning against assuming local background subtraction preserves the target signal; it is not a demonstration that the same filter would fail or succeed on CT.

The CT check independently reconstructed 406,093,824 native uint16 samples from all 1,530 retained range batches. All six 17×384×384 pooled inputs, comprising 15,040,512 float32 values, were bitwise identical. Coarse centers remain exactly native origin+1+3×index; source planes 7–57 pool into centers 8, 11, …, 56. TIFF Orientation 1 is the format default because the tag is absent. These checks exclude the tested byte-order, row-addressing, coordinate and pooling errors in our local pipeline, while leaving historical registration provenance separate.

Every native depth and every pooled depth was summarized under four fixed supports, including both label classes. At the fixed display depths, texture variation decreases strongly beyond the declared center in all six regions. The following values are native surface-support population standard deviations divided by257 for readability; they are not a new input normalization.

| Region | Plane8 | Plane20 | Plane32 | Plane44 | Plane56 |
|---|---:|---:|---:|---:|---:|
| Frag1 context1 | 38.57 | 39.56 | 36.71 | 6.01 | 4.32 |
| Frag1 context2 | 32.06 | 36.02 | 32.18 | 6.68 | 4.84 |
| Frag2 context1 | 32.57 | 34.83 | 34.16 | 12.66 | 6.08 |
| Frag2 context2 | 24.24 | 25.65 | 29.39 | 10.15 | 5.48 |
| Frag3 context1 | 27.99 | 29.99 | 30.86 | 5.91 | 4.40 |
| Frag3 context2 | 27.59 | 29.52 | 33.27 | 6.28 | 4.32 |

The fixed cross-sections show structured material near the nominal center and a lower-variation side at positive offsets. Frag2 has a broader transition, but both its good and weak CT regions share that pattern. This does not reveal a gross isolated depth-range mistake or justify replacing the frozen depth window. Native plane32 positive-minus-negative mean intensity is negative in five regions and positive in Frag2 context1, further limiting a simple global intensity explanation. Intensity and roughness remain confounded by tissue, fibers, interpolation and flattening. Only planes 7–57 were observed: the diagnostic neither locates the microscopic ink layer nor establishes inward/outward anatomy or tissue extent outside this range.

Both published PPM map variants were sampled at the same fixed 81 sites per case, with 3×3 native neighborhoods. The historical source uses native-host double packing without an endian marker. Before sparse acquisition, a source-driven amendment fixed tests of both endian candidates using normalized normals and the documented flat-map constants, never labels or model scores. All six objects uniquely satisfied the little-endian interpretation under those invariants. This is conditional serialization consistency, not authentication of the historical producer hardware or renderer command.

The two map variants describe different coordinate systems. Every sampled surface_volume.ppm tuple equals (x,y,32,0,0,+1), including a flat unit metric. result.ppm instead contains curved surface coordinates and normals. Eight of 486 result-map centers are unmapped, all outside literal surface, known and primary support; all 409 sampled primary-support centers are mapped. The table includes all fixed sites, with invalid centers counted rather than discarded silently.

| Region | Valid result-map centers /81 | Median area scale | Area-scale range | Median signed normal agreement |
|---|---:|---:|---:|---:|
| Frag1 context1 | 81 | 1.0050 | 0.9431–1.5720 | −0.9990 |
| Frag1 context2 | 76 | 0.9927 | 0.9479–1.4489 | −0.9993 |
| Frag2 context1 | 81 | 0.9769 | 0.9507–1.1197 | −0.9996 |
| Frag2 context2 | 79 | 1.1039 | 0.9559–1.2786 | −0.9996 |
| Frag3 context1 | 81 | 0.9626 | 0.9289–1.1205 | −0.9993 |
| Frag3 context2 | 80 | 0.9747 | 0.9288–1.3865 | −0.9993 |

Area scale is norm(du×dv) in the supplied coordinate units per flattened pixel squared. All 478 valid result-map neighborhoods have nondegenerate derivatives and the same signed orientation relative to their stored normals; agreement ranges from −0.999999 to −0.989235. There is no sampled regional orientation reversal. The opposite sign in the flat map is a coordinate convention, not proof of an anatomical flip. Local stretch and shear vary, so a constant flattened pixel pitch is not a measured constant surface-distance metric. Neither sparse samples nor a flat midpoint map certify the entire surface, its native-volume binding or zero physical sampling offset. The larger distortions also occur in regions with better CT ranking, so geometry alone has not explained the failures.

Independent numerical review covered all 1,632 CT plane/support records, all 18 infrared maps, all 12 optical AUROC/AP pairs, all 2,916 sparse byte ranges, all 8,748 PPM records, all 972 map/site geometries, both endian candidates and support coverage. CT summaries matched exactly across producer NumPy 2.3.5 and reviewer NumPy 2.5.3. Independent Gaussian filtering differed by at most 1.58×10⁻¹³; optical metrics met 10⁻¹² tolerance. PPM comparison differed by at most 1.82×10⁻¹² within the frozen limits. Full environment identities are retained: numerical producers used the platform Python 3.12.14 interpreter with NumPy 2.3.5/SciPy 1.17.0, while independent review used the Herculaneum venv with NumPy 2.5.3. This runtime distinction was recorded without altering the original outputs.

Two independent PPM checker attempts stopped on output-schema expectations (a mapped-field expectation and zero-sentinel null presentation). Separate corrected checkers retained the originals and changed no numerical method, tolerance or scientific output. The original header plan was unexecuted and superseded only to bind the final pre-decode source clarification. The overview's second rendering moves the legend; its numerical summary is unchanged. All these records are retained.

The next research step is a controlled test of depth-aware CT representation and robustness to tissue texture, with raw intensity preserved as a control and explicit local geometry assumptions. There is no evidence here to justify silently correcting an alignment or discarding a hard region. [The next-study draft](NEXT_STUDY_DRAFT.md) sets out that hypothesis and the fresh evaluation requirement. Paris2 and its aliases remain excluded from gradient, teacher and cutoff fitting; any method informed by this diagnostic needs fresh material for a confirmatory claim.

The exact historical warp, native CT frame binding and rendering invocation remain unavailable. No transcription, translation, frontier-performance claim, publication, upload or prize submission results from this audit. The retained EduceLab-Scrolls data attribution and legacy use restrictions continue to apply; source citations are in [SOURCE_AUDIT.md](source-review/SOURCE_AUDIT.md) and its pinned JSON. Research code, reports and artifacts remain outside the unchanged application checkout. [COMPLETION_CONTRACT.json](COMPLETION_CONTRACT.json), the additive register and final retention manifest bind the completed evidence; saved startup text is not an artifact backup or proof of fresh-task restoration.

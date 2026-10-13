# Review of all eight fixed CT sites

11 October 2026. Principal-agent review of all 16 panels produced by the
pre-acquisition script. No model predictions, target maps or annotations were
opened. Every site was viewed at the fixed display scales. These observations
are descriptive and do not change the numerical screens.

The XY panels average native layers 10:18 and the corresponding geometry-only
or fitted fine queries. Intensities use 0–255; signed differences use -64..64.
The depth panels retain all native28/fine109 source layers and show the fixed
center row/column. Fine depth sections follow the unshifted UV trace at the
midpoint of the native j13/j14 geometry queries; they do not apply the fitted
CT shift or claim identical scanner PSFs. Vertical axes are nominal source
normal offsets, not a calibrated common world-coordinate section. Magenta in
the XY panels marks unsupported sampling or an unperformed fit.

| Site | Fixed observations | Interpretation and limitation |
|---|---|---|
| w040-x2368-y5440 | XY dark channels and cross-direction texture correspond. The center-row section has the same broad bent band, a deep trough near core x100 and a localized bright feature near x25/offset -115 µm. The fine section resolves internal interfaces blurred in the native scan. | Visible material correspondence despite the conservative source-support failure. No fitted translation was computed; the magenta fit panels are intentional. The zero samples are not evidence of absent material or ink. |
| w040-x6720-y4672 | Both observations show a broad dark central channel and matching breaks in its boundary. The central band has a similar rise/fall through the row section. Fitting reduces positional difference but leaves a broad intensity offset. | Compatible local material; narrower internal bands are unresolved in the native view. Neither the fit nor visual agreement fixes a particular ink-bearing face. |
| w041-x7488-y3136 | A bent oblique feature in the left-center XY image and bright oblique inclusion near row-coordinate40/offset -25 µm recur. The two acquisitions share the main dark channels and undulating interfaces. The one-pixel fitted shift makes little visible change. | This is the closest numerical case: only the median lateral-consistency screen fails. The shared anatomy is stronger evidence than correlation alone, but does not demonstrate subpixel transfer precision. |
| w041-x4672-y576 | The broad upper dark area, localized bright specks and curved lower tissue boundary recur. Both row sections show a concave boundary with a right-side rise; both column sections contain the same lower oblique boundary. | Clear bulk-structure correspondence. The fit is zero, while separate patches prefer about one voxel of displacement; no local corrections were applied. Fine internal interfaces are not independently labeled wraps. |
| w041-x320-y576 | The bright dot below center and nearby elongated feature on the right recur in XY. The two sections contain matching branching/curving boundaries, with a visible depth offset. The fixed fit reduces broad signed differences. | The fitted +2 depth steps are a CT correspondence adjustment, not an estimate of ink depth or permission to recenter a model slab. |
| w043-x7488-y2880 | Matching XY crossings and a thin bent central band. The row section shows the same left-hand bulge, abrupt descent near x50 and broad dark space on the right. | Material correspondence is visually plausible, although one checking search touches zero-valued fine support. No fit/score was substituted for the frozen failure. |
| w043-x4928-y5440 | The prominent lower-center bright inclusion and horizontal channel recur; their XY displacement visibly decreases after fitting. Full-depth views preserve matching large voids and curved boundaries. | Source geometry alone leaves a local shift; a single fitted shift does not align every checking patch. Multiple fine interfaces remain unresolved within the native bands. |
| w043-x3648-y2624 | Several bright points, the lower horizontal channel and right-edge boundary recur. The row section has the same left-hand arch and sharp right-side rise. Fitting reduces broad XY differences; unsupported pixels near the top-right remain magenta. | Strong checked correspondence after +2 depth steps, but distinct lateral alternatives remain too competitive. Unsupported display pixels lie outside the fixed scored supports; they are not repaired. |

The images support corresponding local material structures on all eight
viewed locations, with quantitative support on the six scorable ones. Fine
observations often resolve several interfaces inside a broad native band.
These could include papyrus internal structure and/or nearby sheets; the
panels do not independently identify an ink-bearing face or establish which
fine interface should supply a target. No letters, ink accuracy, pristine
test set or unique pointwise registration are claimed.

All 16 panels are retained in `panels/`, with hashes in `panels/FILES.json`.
The initial plotting process used Matplotlib's temporary writable cache after
the home cache was unavailable; all figures and labels rendered successfully.
Set MPLCONFIGDIR and XDG_CACHE_HOME to writable temporary directories for replay.

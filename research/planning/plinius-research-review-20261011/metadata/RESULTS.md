# PHerc0139 coordinate metadata: verified contracts

All32 bounded public metadata requests returned200, consuming53,332 response
bytes. [The replay](analyse_metadata.py) verifies every response hash/length,
11 surface-volume contracts, four fine-CT pyramid contracts, and three native
reconstruction-volume contracts. These metadata checks consumed no CT pixels.
The separately frozen reconstruction control is documented in the parent packet.

- Four benchmark surfaces have fine78-keV109-plane and native113-keV28-plane
  arrays. w024/w047/w053 also have59-keV116-plane arrays. Canvas sizes, chunk
  sizes, ZYX axes and depth lengths agree between retained array metadata and
  fresh root metadata.
- The59-keV path names the original1.129µm acquisition, but its public `-L1`
  renders are2.258µm in every axis. The catalogue renderer records group1 and
  voxel-size2.258. Do not supply them to a1.129µm model without resampling.
- Fine78-keV pyramid level2 is **2.399µm in Z and9.596µm in XY**. Its109 depth
  planes are unchanged. It is an inexpensive structural-registration input,
  not an isotropic9.6µm model input. Downsampling in Z and frame conventions
  remain explicit work.
- Root translations of zero describe the abstract surface array coordinates;
  they do not establish fine/native tissue alignment or an anatomical normal.
  Neither physical sample identity nor a shared segment name closes that gap.
- Native raw-volume OME level0 scales are unitless1,1,1; physical9.362µm comes
  from the catalogue. All three raw-volume contracts use uncompressed128³
  uint8 chunks (2MiB), permitting the declared18MiB three-site control.
- Fine78 volume20260102150214 has a newer remasked successor. Preserve the
  older release when reproducing its released teacher benchmark. An identity
  voxel frame does not imply identical mask or CT values.

A prospective pair of at most16 fine-level2 chunks plus16 native-surface
chunks costs at most35,913,728 bytes (34.25MiB) decoded/on-wire under these
uncompressed contracts. This is a conditional block bound, not an acquired
crop or a guarantee of sufficient geometric context. Actual requests and
coordinates must be frozen using the eligible training surface's own metadata.
No benchmark-region crop or teacher-label threshold was selected here.

The [machine-readable findings](METADATA_FINDINGS.json), catalogue extract,
plans, response bodies and ledgers are included. This closes source-format
questions; it does not admit pixelwise paired supervision.

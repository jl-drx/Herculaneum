# Structural review packet: four Frag6 regions

The packet preserves exact crop coordinates and fixed display scales. It contains no model predictions or annotation overlays. All displayed CT values, optical crop values and cross-sections passed independent reconstruction. These are previously inspected development regions; root is not blinded.

Start with [optical crops and fixed CT cross-sections](packet-v1/optical-and-cross-sections.png). Each row shows the exact optical crop and CT sections through raw row193 and column193. CT horizontal axes are local raster pixels; vertical axes are historical layers1..63. The aspect ratio is for display, not physical geometry. All optical pixels are shown on the fixed0..255scale and CT on0..65535.

Each complete montage contains every retained layer, displayed as a3×3 spatial mean with a fixed0..65535scale:

| Region | Raw origin (row,column) | Complete CT sequence |
|---|---|---|
| r544-c4128 | (608,4192) | [All63layers](packet-v1/r544-c4128-all-layers.png) |
| r2080-c3616 | (2144,3680) | [All63layers](packet-v1/r2080-c3616-all-layers.png) |
| r4640-c1056 | (4704,1120) | [All63layers](packet-v1/r4640-c1056-all-layers.png) |
| r6688-c1568 | (6752,1632) | [All63layers](packet-v1/r6688-c1568-all-layers.png) |

Local raw coordinates run0..383. A displayed montage pixel(y,x) averages raw rows3y..3y+2 and columns3x..3x+2. Global raw coordinates equal crop origin plus local coordinates. Layer indices are not a proven anatomical ink depth.

Root's initial review: CT shows substantial fibrous structure, with much less visible structure at later layers. The optical crops are smooth and dominated by dark/light shapes; I cannot confidently identify a shared non-ink landmark in these views. No correspondence has been accepted and no transform has been fitted. A changing CT surface boundary does not itself locate the ink-bearing side.

The specific review question is whether a tear, edge, fold or other non-ink structure can be located confidently in both the optical image and a CT layer. A proposed match should include region, optical/raw coordinates, CT/raw coordinates, CT layer, uncertainty and why the feature is structurally the same. “No clear match” is a valid outcome. Similar dark ink shapes or improved model scores are not registration evidence. The [blank landmark form](LANDMARK_REVIEW.csv) contains no fabricated points. Any subsequent transform must use separately frozen fit and held-out check landmarks.

The new PPM sample covers36fixed locations across all four crops; all324records are mapped with unit normals and all declared source rays remain in bounds. This supports local geometry consistency, not optical registration. Native-source correspondence tests are being planned separately. No optical-ground-truth or benchmark admission follows from the packet.

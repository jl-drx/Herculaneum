# Additional preflight for the next full-field render

The sealed study49 renderer logs report a default output voxel size of1.0 because
VC3D did not derive physical units from the served source metadata. This does not
change the observed intensity comparisons: the mesh is in native CT index units,
the render used scale1 and unit native layer steps, and Hecate preparation received
explicit8.64um spacing in all axes. The fixed31/66-plane numerical gates passed.
Nevertheless, these output Zarr attributes must not be treated as authoritative
physical spacing by a generic downstream reader.

For the next full-field run, freeze `--voxel-size 8.64 --voxel-unit micrometer`
and verify that output scale metadata agrees with the known acquisition. At the
fixed study49 patch, confirm that adding these explicit output-unit arguments
preserves intensity bytes before scaling up. Keep the original study49 archive
and logs unchanged. Future consumers must bind both array identity and physical
spacing; the model argument alone is insufficient provenance for general reuse.

Also retain the writable VC3D settings fix: VC3D_CONFIG_DIR selects the directory
containing VC3D.ini; set viewer/remote_cache_dir there to an explicit study path.
Setting the environment variable alone does not move the default remote cache.

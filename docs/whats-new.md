# What's new

## Unreleased

- New `finalize_healpix_dataset` conversion stage, the last one, run once every
  group of the output dataset has been written: it consolidates the Zarr
  metadata of the output dataset. No other stage writes consolidated metadata
  anymore (`prepare_healpix_dataset` and `generate_pyramids` did): such a copy
  of the metadata of a whole hierarchy is stale as soon as a group or an array
  is written next to it. A conversion run in one call with
  `create_healpix_dataset` runs that stage as well.
- New `consolidate` metadata settings, which select the groups the consolidated
  Zarr metadata is written for: the root group (`root_group`), each group
  converted to HEALPix (`healpix_groups`), each multiscale group
  (`multiscale_groups`), each group holding a derived level of a pyramid
  (`derived_level_groups`) and any other group of the output dataset
  (`groups`). All four flags are set by default; before, only the root group
  was consolidated. A `groups` path that names no group of the output dataset
  is reported with a warning when the staging cache is created, and skipped
  when the metadata is consolidated.
- Minimum versions raised to healpix-resample 2026.10.2 (its `batch_size`
  option, used by the `"dem-glo30"` settings), healpix-geo 0.4.1 and, for
  development, healpix-plot 2026.7.0.
- New `output_dtype` group setting, the floating-point data type of the resampled
  data variables (e.g. `"float16"`): before, the `"bilinear"` and `"psf"`
  resamplers always wrote `float64`. The derived levels of a pyramid follow the
  data type of the converted level.
- Projected input datasets whose x or y coordinates run the other way than
  usual (increasing y, i.e. a bottom-up raster, or decreasing x) are now
  converted: the input points of a chunk are selected with slice bounds that
  follow the signs of the transform. Before, the selection was empty for every
  chunk of such a dataset.
- A chunk whose (buffered) polygon only grazes the extent of a projected input
  dataset, with no pixel centre in the overlap, no longer gives the resampler
  zero points: that dataset is skipped for the chunk, and a chunk left without
  input is not written ({issue}`23`). The conversion of a tile stopped there
  with `nearest` and with healpix-resample < 2026.10.1 for `psf`.
- New `"mode"` reduction for the pyramids (the `reduce` pyramid setting): the
  most frequent value of the cells of a block, for categorical variables such
  as masks and classifications.
- New `codecs` pyramid setting: the Zarr codecs of the aggregated variables of
  the derived levels (Zarr's defaults if not set), e.g. a serializer of
  `healpix-compress` coding one HEALPix cell per chunk. It is checked when the
  settings are created, like the `codecs` group setting.
- The ERA5 converter writes several analysis times in one store: `date` and
  `time` take MARS lists and ranges (`date="20260801/to/20260831"`,
  `time="00:00:00/06:00:00/12:00:00/18:00:00"`), each group gets a `time`
  coordinate, and the new `time_chunk` option sets how many times a chunk
  holds (`"all"`: the time series of 1024 cells in one chunk). The enda group
  gets the 3-hourly times only, the times of the ERA5 EDA. New public
  `download()`, `dataset()` and `grib_paths()` methods give the GRIB files
  without writing a store; a single date and time give the same store and
  cache file names as before.
- The chunks of a derived level always hold all the children of one HEALPix
  cell: on a partial spatial extent, their cells are no longer coarser than the
  coarsest level the converted cells cover entirely. Before, the chunks of the
  derived levels whose chunk cells were partly covered were cut along the array
  and mixed the cells of several HEALPix cells.
- New `extra_dimensions` group setting, to align several input datasets along
  a non-spatial dimension whose labels differ between them (e.g. the
  `detector` dimension of the Sentinel-2 `conditions/geometry` group): union,
  intersection, same labels, or fixed labels for every conversion
  ({issue}`17`, {issue}`54`). Input datasets that differ along a non-spatial
  dimension without such a setting are now refused with a message pointing to
  it; before, inputs of the same size were concatenated position by position
  even when their labels differed.
- New `"dem-glo30"` settings and example notebook for the Copernicus DEM
  GLO-30: bilinear resampling to level 18 in float32, sharded (one shard per level-6 cell,
  inner chunks of one level-8 cell), a pyramid of block means down to level 8,
  and the licence of the data in the output STAC item. The notebook reads the
  heights from Earth Data Hub or from the CDSE COG tiles (`SOURCE`). Based on
  the GLO-30 notebook by Pablo Richard.
- The multiscales metadata describes each derived level of a pyramid: the
  converted level it is aggregated from (`derived_from`), the transformation
  from it (`transform`, whose `scale` is how much longer a cell is) and the
  method its cells are aggregated with
  (`resampling_method`, also given for the whole group).
- A derived level whose variables are aggregated from different native levels
  is described by one layout item per array, whose assets are the paths of the
  arrays (e.g. `"1/a"` aggregated from `"3/a"` and `"1/b"` from `"2/b"`)
  instead of the paths of the level groups. Before, such a level was described
  by a single item naming one of those native levels, which misdescribed the
  variables aggregated from the others (`derived_from`) and how much their
  cells were aggregated (`transform`). The Multiscales Zarr convention asks
  for the same referencing style throughout a layout, so the whole multiscale
  group is then described array by array, the native levels included.
- The HEALPix grid of a level is no longer added to its multiscales layout
  items by default (`add_healpix_positioning`): it repeats the whole grid in
  every item of that level, which one item per array makes particularly
  verbose, while the attributes of a level group already describe its grid and
  the `asset` of an item names its refinement level.
- New `asset_type` multiscale setting: what a layout item refers to, a level
  group (`"group"`) or one array of a level (`"array"`). With `"auto"` (the
  default), the level groups are the assets where they describe every level
  and the arrays otherwise; `"group"` rejects a pyramid whose derived level
  would aggregate its arrays from different native levels.
- New `generate_pyramids` conversion stage, which aggregates the converted
  data onto the coarser HEALPix refinement levels (the derived levels) given by
  the new `pyramid_groups` multiscale settings. One group per derived level,
  and the multiscales metadata describing the pyramid, are created with the
  rest of the structure of the output dataset by `prepare_healpix_dataset`; the
  arrays of those groups are written with the aggregated data, and the
  converted levels are never rewritten. The groups must be converted with the
  `healpix_cell_dense` chunk method: the cells of a coarser cell are then
  consecutive cells of the converted arrays, so the levels are aggregated by
  position without reading any cell id.
- The `asset` of a multiscales layout item is now a path relative to the group
  holding the metadata (e.g. `"7"` instead of `"measurements/7"`), as the
  Multiscales Zarr convention asks.
- The attributes of the output variables can be set from the group settings
  (`metadata.override_attrs`), and not only excluded ({issue}`36`).
- The STAC properties, extensions, links and assets propagated to the output
  dataset can be excluded and set from the conversion settings
  (`metadata.stac`). The STAC metadata written by `healpix-convert` itself
  keeps precedence, except for the recorded conversion settings, which
  `metadata.stac.include_conversion_settings` leaves out of the output. The
  input's assets are no longer propagated by default: they point at the input
  files, which the output does not hold.
- With several input datasets, the STAC properties, extensions, links and
  assets of the output can be merged as the intersection of the input ones,
  and the extensions and links also as their union
  (`metadata.stac.merge_properties`, `merge_extensions`, `merge_links` and
  `merge_assets`). They were only ever taken from the first input dataset,
  which stays the default -- except for the extensions, which used to fail the
  conversion when they differed across the inputs.
- The JSON schema of the conversion settings is generated from
  {py:class}`~healpix_convert.ConvertSettings` and published as
  `convert-settings.schema.json` in the root of the repository. The STAC
  `processing:expression` property of the output records it as the `format` of
  the settings, pinned to the tag or the commit of the `healpix-convert`
  version that produced the dataset ({issue}`19`).
- The CAMS converter's nearest-neighbour method now assigns the input points
  to cells with healpix-geo on the WGS84 ellipsoid that the output declares,
  instead of healpy on the sphere. Some points move to a neighbouring cell,
  and healpy is no longer needed ({pull}`50`).
- The dependencies of the ERA5, CAMS and ClimateDT converters are the `era5`,
  `cams` and `climatedt` extras, and the README (the PyPI page) shows how to
  install from PyPI ({pull}`48`).
- The variables of a chunk are resampled in one batched call ({pull}`31`) and
  the output arrays of a chunk are written concurrently ({pull}`39`).
- The group codecs are checked when the settings are created ({pull}`27`).
- The cell ids can be delta-coded before compression ({pull}`42`, {pull}`47`).
- Object string arrays are written as variable-length strings ({pull}`40`).
- The affine transform derived from x / y coordinates points at the top-left
  corner of the pixels ({pull}`24`).
- A warning is emitted when a variable with valid input values comes back
  all-NaN ({pull}`41`).
- Sparse chunks are written with the data type of the input variable
  ({pull}`46`).
- Documentation site, built with Sphinx and published on GitHub Pages.

## v2026.10.0 (2026-10-08)

- The spatial coordinate is promoted to an xarray coordinate ({pull}`1`).
- The input's description of its variables is kept ({pull}`2`).
- The conventions followed by the output dataset can be selected ({pull}`3`),
  also by the ERA5 / CAMS / ClimateDT converters ({pull}`6`).
- The output records how it was produced, as STAC processing metadata
  ({pull}`5`).
- `CITATION.cff` ({pull}`7`).
- Example notebook converting the ESA CCI land cover map ({pull}`10`).
- Input attributes can be excluded from the output ({pull}`14`).

See the
[release notes](https://github.com/GRID4EARTH/healpix-convert/releases/tag/v2026.10.0).

## v2026.09.0 (2026-09-08)

First release on PyPI.

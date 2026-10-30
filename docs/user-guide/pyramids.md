# HEALPix pyramids

{py:func}`~healpix_convert.generate_pyramids` is the stage of a conversion
that aggregates the converted data onto the coarser HEALPix refinement levels
given by the pyramid settings of each multiscale group, and writes the arrays
of their groups. Only finalizing the output dataset comes after it.

Those groups, one per derived level and named after it, and the
`multiscales` metadata describing the whole pyramid, are part of the structure
of the output dataset: {py:func}`~healpix_convert.prepare_healpix_dataset`
creates them, like it creates the converted groups. Without pyramid settings,
both stages leave the dataset alone.

```python
import healpix_convert as hc

settings = hc.ConvertSettings.from_dict(
    {
        "multiscale_settings": {
            "groups": ["measurements"],
            "pyramid_groups": {"measurements": {"derived_levels": [5, 4]}},
        },
        "group_settings": {
            "measurements/fine": {
                "healpix": {"refinement_level": 7},
                "chunk": {
                    "method": "healpix_cell_dense",
                    "healpix": {"refinement_level": 4},
                },
                ...,
            },
            "measurements/coarse": {"healpix": {"refinement_level": 6}, ...},
        },
    }
)

cache = hc.create_staging_cache(["input.zarr"], settings)
hc.prepare_healpix_dataset(cache, settings, "converted.zarr")
for group in ("measurements/fine", "measurements/coarse"):
    hc.convert_group_to_healpix(
        group, cache=cache, settings=settings, output_path="converted.zarr"
    )

hc.generate_pyramids(cache, settings, "converted.zarr")
hc.finalize_healpix_dataset(cache, settings, "converted.zarr")
```

{py:func}`~healpix_convert.create_healpix_dataset` does all of that in one
call, pyramids included.

```text
converted.zarr
└── measurements        (multiscale group, its layout describing the pyramid)
    ├── 7               (converted from measurements/fine)
    ├── 6               (converted from measurements/coarse)
    ├── 5               (derived, aggregated from the level 6)
    └── 4               (derived)
```

The structure of the pyramids — which levels are derived, in which groups,
with which HEALPix grid and chunks, and the multiscales metadata describing
them — is computed once, from the settings and the input datasets, when the
staging cache is created: `cache.output_groups.pyramid_multiscale_groups` holds
one entry per **pyramid multiscale group**, i.e. per group of the output
dataset holding the levels converted from the input data (the **native
levels**) and the levels derived from them (the **derived levels**). The
groups that get a pyramid, and how, are given by the `pyramid_groups`
multiscale settings (see {doc}`settings`).

## Dense chunks, and why they are required

Every child group of a pyramid multiscale group must be converted with the
`healpix_cell_dense` chunk method. The cells of such a chunk are all the
children of one HEALPix cell at a coarser (chunk) level, in ascending (nested)
order, so the cells of a coarser cell are `4 ** delta` **consecutive** cells of
the converted arrays: aggregating them is a matter of implicit position (array
indices).

Each derived level is one `Dataset.coarsen` reduction over the converted
cells:

- nothing is grouped and no cell id is read, so nothing proportional to the
  number of cells is held in memory;
- the aggregation is blockwise, hence lazy: all the derived levels of all the
  pyramids are written in a single Dask computation, so a native level is read
  once and feeds every level aggregated from it. Run it on a Dask distributed
  cluster (see {doc}`dask`) to parallelize it;
- each cell of a derived level aggregates **all** the converted cells it
  contains, picked at the corresponding native level: the default `mean` is a
  block mean and not a mean of means, which differ as soon as some cells have no
  data.

Missing (NaN) values are skipped. The `reduce` pyramid setting selects another
reduction (`"min"`, `"max"`, `"sum"`): these are the reductions that give, for
a whole block of cells, the same result as aggregating all its cells at once.
The mean does not suit categorical data such as masks or classifications:
aggregate those with `"mode"`, the most frequent value of the cells of a block
(the smallest of them in case of a tie). Bit flags, i.e. integer variables
where each bit is a flag of its own (e.g. quality flags), are aggregated with
`"bitwise_or"`: a flag is set in a coarser cell if it is set in any of its
cells, even in a minority of them, which `"mode"` would drop. Missing cells
(the fill value) are skipped, and a coarser cell with no data gets the fill
value. Signed and unsigned integers are both supported (the sign bit is a flag
like the others); a floating point variable is an error, as is a 64-bit
integer variable with a fill value, whose values are decoded as floating point
values that do not hold all their bits.

The data type and the fill value of each variable are preserved, an integer
variable being rounded instead of truncated.

A derived level may only hold cells that are entirely covered by converted data:
a cell coarser than the chunk level is entirely covered only if all its chunk
cells have been converted, which is not the case at the edges of a partial
spatial extent. The levels of `derived_levels` that are coarser than the
coarsest entirely covered level are therefore rejected. For example, with one
satellite tile converted at level 17 with level-11 chunks, if no level-10 cell
has all its four level-11 chunk cells converted, the coarsest entirely covered
level is 11: the derived levels 16 to 11 are accepted, 10 and coarser are
rejected. With a full-earth extent, every level down to 0 can be derived.

## Chunks of the derived levels

Like the chunks of a native level, the chunks of a derived level are
HEALPix-aligned, but on a coarser refinement level than the one used for the
associated native level. That chunk level is set so that both levels keep the
same depth (data level minus chunk level), hence chunks of the same size: with a
native level 17 chunked at level 11 (depth 6, i.e., 4096 cells per chunk), a
derived level 15 is chunked at level 9, also 4096 cells per chunk. The minimum
depth is set to 4 (256 cells per chunk).

To ensure HEALPix-aligned chunks for all derived levels, that chunk level is
also bounded by the coarsest level that is entirely covered by converted data (0
for a full-earth extent). Below that level threshold, some of the chunk cells
would be only partly covered, such that the actual chunks would be cut along the
array (i.e., not HEALPix-aligned). A partial extent (e.g., one tile of a
satellite product) raises that threshold, hence smaller chunks for the derived
levels near it.

All in all, the chunk refinement level $l_c$ of a derived level is

$$l_c = \max(l - \max(l_n - l_{n_c}, 4), l_\text{cov})$$

where $l$ is the refinement level of the derived level, $l_n$ and $l_{n_c}$
those of the associated native level and of its chunk cells, and
$l_\text{cov}$ the coarsest level entirely covered by converted data.

## Codecs of the derived levels

The aggregated variables are written with Zarr's default codecs, unless the
`codecs` pyramid setting gives a codec pipeline, applied to the data variables
of every derived level (not to their cell ids). It takes the same form as the
`codecs` group setting:

```python
"pyramid_groups": {
    "measurements/reflectance": {
        "derived_levels": [18, 16, 15, 14, 13, 12],
        "codecs": [
            {
                "name": "hcc-sentinel2-pyramid",
                "configuration": {"quantification_value": 10000.0, "add_offset": -1000.0},
            }
        ],
    },
}
```

Since a chunk of a derived level holds the cells of one HEALPix cell, codecs
that code one parent cell per chunk, like the Sentinel-2 serializers of
`healpix-compress`, can encode them. The converted levels keep the codecs they
were written with: the codecs of a lossy serializer are not reused implicitly,
as they would code the aggregated values a second time.

A `sharding_indexed` serializer writes one shard per chunk of a derived level,
with the inner chunks it gives. The chunks of the coarsest derived levels may
hold fewer cells than those inner chunks: there, the inner chunks are cut to
the chunk, i.e. each shard holds one inner chunk.

The cell ids of the derived levels are written with the codecs of the cell ids
of the converted level (the `healpix_coordinate_codecs` group setting, e.g. a
delta filter), on the chunks of the derived level. Sharded converted cell ids
give one shard per chunk of a derived level, with the same inner chunks, or the
whole chunk where it is smaller.

## Groups converted at several levels

The native levels of a multiscale group are kept as they are. Generating
pyramids only creates down-sampled (aggregated) derived levels, by default from
the nearest native level found. Set `coarsen_from="finest"` to always aggregate
from the finest native level instead.

That native level is selected per variable, so one derived level may hold
variables aggregated from different native levels, e.g., the bands of a
multi-resolution product. Since all the variables of a derived level share the
same cell ids, the cells of the level are the union of the cells of its
variables (no data cells filled with the variable's fill value).

This is also why all the native levels of a multiscale group must be chunked
with HEALPix cells of the same refinement level, which is enforced when the
conversion settings are validated: the cells are aggregated by position and the
variables of a derived level are merged without reading any cell id, so the
native levels must cover exactly the same cells for those variables to be
aligned with each other.

A layout item of the multiscales metadata describes one level, so it can only
name one `derived_from` source: a derived level whose variables come from
different native levels needs one item per array instead, so the whole group is
then described array by array, the native levels included. For a group holding
"a" at the level 3, "b" at the level 2 and a derived level 1 aggregating each of
them from its own native level:

```json
"layout": [
  { "asset": "3/a" },
  { "asset": "2/b" },
  { "asset": "1/a", "derived_from": "3/a", "transform": { "scale": [4.0] } },
  { "asset": "1/b", "derived_from": "2/b", "transform": { "scale": [2.0] } }
]
```

Each item then carries the transformation from the native level of that array
alone, with the aggregation method like any other item. The level group of an
array is the parent path of its asset (`1` for `1/a` above), the group that
holds the cell ids coordinate the arrays of the level share.

The `asset_type` multiscale setting chooses the style: `"auto"` (default)
describes the level groups where that describes every level and switches the
whole layout to the arrays otherwise, `"array"` always describes the arrays, and
`"group"` always describes the level groups (raises an error when a derived
level would aggregate its arrays from different native levels).

## Where the levels are written

The derived levels are groups of the multiscale group, next to the levels
converted from the input data:

```text
measurements        (multiscale group)
├── 7               (converted)
├── 6               (converted)
├── 5               (derived)
└── 4               (derived)
```

With a `subgroup_name`, they are written in a child group of it instead:

```text
measurements        (multiscale group)
├── 7               (converted)
├── 6               (converted)
└── pyramids
    ├── 5           (derived)
    └── 4           (derived)
```

`prepare_healpix_dataset(..., dry_run=True)` shows the derived level groups and
their metadata, like the rest of the structure of the output dataset, without
writing anything.

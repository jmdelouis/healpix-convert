"""Generating pyramids of coarser HEALPix refinement levels."""

from __future__ import annotations

from pathlib import PurePath

import healpix_geo.nested as hpn
import numpy as np
import pytest
import shapely
import xarray as xr
import zarr

from healpix_convert import (
    ConvertSettings,
    ConvertStagingCache,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert.cache import (
    MIN_CHUNK_DEPTH,
    _chunk_cells,
    _coarsest_whole_level,
    _derived_chunk_level,
    _derived_level_sources,
    _multiscales_metadata,
    _set_pyramid_multiscale_groups,
)
from healpix_convert.core import utils
from healpix_convert.core.conversion_models import OutputGroupInfo, OutputSpatialInfo
from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    get_multiscale_group_attrs,
)
from healpix_convert.core.multiscales_conventions import HealpixMultiscales
from healpix_convert.pyramid import create_derived_level_groups
from healpix_convert.settings.common import MetadataSettings
from healpix_convert.tests.test_batch_resample import STAC_ITEM

GROUP = "measurements"
METADATA = MetadataSettings()
HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
FULL_EXTENT = shapely.box(-180.0, -90.0, 180.0, 90.0)
CHUNK_LEVEL = 1


# --- a converted dataset, its staging cache and its conversion settings,
# --- written and built by hand (see the end of this module for the conversion
# --- of an actual input dataset)


def _healpix(level: int) -> Healpix:
    return Healpix.model_validate({"refinement_level": level, **HEALPIX})


def _dense_cells(level: int, extent: shapely.Polygon = FULL_EXTENT) -> np.ndarray:
    """The cells a dense conversion of ``extent`` writes at the given level:
    all the children of the chunk cells covering it.
    """
    chunk_cells, _ = utils.compute_output_chunk_info(extent, CHUNK_LEVEL, "wgs84")
    cell_ids = hpn.zoom_to(np.sort(chunk_cells), CHUNK_LEVEL, level)

    return np.sort(cell_ids.ravel())


def _level_dataset(level: int, cell_ids, **variables) -> xr.Dataset:
    """One (single-scale) HEALPix group of a converted dataset."""
    healpix = _healpix(level)
    chunk_size = 4 ** (level - CHUNK_LEVEL)

    ds = xr.Dataset(
        {name: ("cells", values) for name, values in variables.items()}
        # the CF grid mapping variable, like the one written by the conversion
        | {
            CF_GRID_MAPPING_VARIABLE: (
                (),
                np.int8(0),
                get_cf_grid_mapping_attrs(healpix, METADATA),
            )
        },
        coords={"cell_ids": ("cells", np.asarray(cell_ids, dtype="uint64"))},
        attrs=get_healpix_group_attrs(healpix, METADATA),
    )

    # one chunk per chunk cell, densely filled with its children
    return ds.chunk({"cells": chunk_size})


def _layout_assets(multiscales) -> list[str]:
    """Assets of a multiscales layout, given as metadata or as a model."""
    if isinstance(multiscales, HealpixMultiscales):
        multiscales = multiscales.model_dump()

    return [item["asset"] for item in multiscales["layout"]]


def _layout(multiscales, asset: str) -> dict:
    """One item of a multiscales layout, given as metadata or as a model."""
    if isinstance(multiscales, HealpixMultiscales):
        multiscales = multiscales.model_dump()

    return next(item for item in multiscales["layout"] if item["asset"] == asset)


def _variable_names(data) -> dict[int, list[str]]:
    """Variables converted at each native level of a hand-written dataset
    (native level -> variable names).
    """
    return {level: sorted(variables) for level, variables in data.items()}


def _block_means(values, parents) -> list[float]:
    """Mean of the values of the cells of each parent cell, coarser cells in
    ascending (nested) order.
    """
    return [np.nanmean(values[parents == cell_id]) for cell_id in np.unique(parents)]


def _multiscales(levels) -> HealpixMultiscales:
    """Multiscales metadata of the levels converted from the input data."""
    layout = [
        {"asset": str(level), "dggs": _healpix(level)}
        for level in sorted(levels, reverse=True)
    ]

    return HealpixMultiscales.model_validate({"layout": layout})


def _write_converted(path, levels: dict[int, xr.Dataset], **to_zarr_kwargs) -> str:
    """Write a converted dataset: one multiscale group with one child group per
    refinement level converted from the input data.
    """
    group_attrs = {"description": "converted data"} | get_multiscale_group_attrs(
        _multiscales(levels), METADATA
    )
    nodes = {f"/{GROUP}": xr.Dataset(attrs=group_attrs)}

    for level, ds in levels.items():
        nodes[f"/{GROUP}/{level}"] = ds

    tree = xr.DataTree.from_dict(nodes)
    tree.attrs["stac_discovery"] = STAC_ITEM
    tree.to_zarr(path, mode="w", zarr_format=3, consolidated=True, **to_zarr_kwargs)

    return str(path)


def _cache_and_settings(
    levels,
    pyramid: dict | None = None,
    extent: shapely.Polygon = FULL_EXTENT,
    chunk_method: str = "dense",
    variables: dict[int, list[str]] | None = None,
    asset_type: str = "auto",
    add_positioning: bool = False,
) -> tuple[ConvertStagingCache, ConvertSettings]:
    """The staging cache and the settings of that conversion: where the
    children of the multiscale group have been written, the spatial extent of
    the output and the pyramid settings of the group.

    The structure of the pyramids is computed by the staging cache, as
    :py:func:`~healpix_convert.create_staging_cache` does.

    ``variables`` gives the variables converted at each native level (native
    level -> variable names). By default every native level holds the same
    single variable, i.e. a derived level is aggregated from one native level.

    """
    settings = _settings(levels, pyramid, chunk_method, asset_type, add_positioning)
    output_spatial = OutputSpatialInfo(geometry_latlon=extent)
    io_path_map = {PurePath(GROUP): PurePath(GROUP)}

    for level in levels:
        io_path_map[PurePath(f"{GROUP}/l{level}")] = PurePath(f"{GROUP}/{level}")

    if variables is None:
        variables = {level: ["a"] for level in levels}

    native_variables = {
        PurePath(f"{GROUP}/{level}"): names for level, names in variables.items()
    }

    pyramid_multiscale_groups = _set_pyramid_multiscale_groups(
        settings, io_path_map, output_spatial
    )
    multiscale_groups = {PurePath(GROUP): _multiscales(levels)}

    for path, info in pyramid_multiscale_groups.items():
        # the group holding the pyramid is described by its whole layout
        levels = {}

        for level_path, (level_healpix, _) in (
            info.native_levels | info.derived_levels
        ).items():
            levels[level_path] = level_healpix

        pyramid_settings = settings.multiscale_settings.pyramid_groups[path]
        multiscale_groups[path] = _multiscales_metadata(
            io_path_map[path],
            levels,
            settings.multiscale_settings.add_healpix_positioning,
            variables=native_variables,
            derived_from=_derived_level_sources(
                info, pyramid_settings.coarsen_from, native_variables
            ),
            resampling_method=pyramid_settings.reduce,
            asset_type=settings.multiscale_settings.asset_type,
        )

    cache = ConvertStagingCache(
        output_spatial=output_spatial,
        output_groups=OutputGroupInfo(
            multiscale_groups=multiscale_groups,
            pyramid_multiscale_groups=pyramid_multiscale_groups,
            io_path_map=io_path_map,
        ),
    )

    return cache, settings


def _settings(
    levels,
    pyramid: dict | None = None,
    chunk_method: str = "dense",
    asset_type: str = "auto",
    add_positioning: bool = False,
):
    """The settings of that conversion, with the pyramid settings of the
    multiscale group.
    """
    if chunk_method == "dense":
        chunk: dict = {
            "method": "healpix_cell_dense",
            "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
        }
    else:
        chunk = {
            "method": "healpix_cell_sparse",
            "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
            "chunk_size": 16,
        }

    group_settings = {}

    for level in levels:
        group_settings[f"{GROUP}/l{level}"] = {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
        }

    return ConvertSettings.model_validate(
        {
            "multiscale_settings": {
                "groups": [GROUP],
                "asset_type": asset_type,
                "add_healpix_positioning": add_positioning,
                "pyramid_groups": {} if pyramid is None else {GROUP: pyramid},
            },
            "group_settings": group_settings,
        }
    )


def _generate_pyramids(cache, settings, path):
    """The stages that write a pyramid of a hand-written dataset: creating its
    groups and metadata (as preparing the output dataset does), aggregating the
    converted data onto them, then finalizing the output dataset (which
    consolidates its Zarr metadata).
    """
    store = zarr.open_group(path, mode="a").store

    for input_path, multiscales in cache.output_groups.multiscale_groups.items():
        group = zarr.open_group(
            store,
            path=cache.output_groups.io_path_map[input_path].as_posix(),
            mode="a",
        )
        group.attrs.update(get_multiscale_group_attrs(multiscales, settings.metadata))

    create_derived_level_groups(store, cache, settings)
    generate_pyramids(cache, settings, path)

    return finalize_healpix_dataset(cache, settings, path)


@pytest.fixture
def converted(tmp_path):
    """A full-earth dataset converted at the level 3, with dense chunks of one
    level-1 cell each (16 cells per chunk), holding a float variable (some
    cells have no data) and an integer one.
    """
    cell_ids = _dense_cells(3)
    rng = np.random.default_rng(42)
    temperature = rng.random(cell_ids.size)
    temperature[rng.random(cell_ids.size) < 0.1] = np.nan
    counts = rng.integers(0, 100, cell_ids.size).astype("int16")

    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, temperature=temperature, counts=counts)},
    )

    return path, cell_ids, temperature, counts


@pytest.fixture
def converted_multi_level(tmp_path):
    """A full-earth dataset converted at the levels 3 and 2, where "a" is also
    stored at the level 2 (a copy resampled by the data provider).
    """
    cell_ids = {level: _dense_cells(level) for level in (3, 2)}
    data = {
        3: {"a": np.arange(cell_ids[3].size, dtype="float64")},
        # the provider's copy of "a" is not the mean of the level 3 cells, and
        # "b" is only stored at the level 2
        2: {
            "a": np.full(cell_ids[2].size, 100.0),
            "b": np.arange(cell_ids[2].size, dtype="float64"),
        },
    }
    datasets = {}

    for level in data:
        datasets[level] = _level_dataset(level, cell_ids[level], **data[level])

    path = _write_converted(tmp_path / "converted.zarr", datasets)

    return path, cell_ids, data


def test_derived_levels_and_metadata(converted):
    path, cell_ids, _, _ = converted
    cache, settings = _cache_and_settings(
        [3], {"derived_levels": [2, 1]}, add_positioning=True
    )

    dt = _generate_pyramids(cache, settings, path)

    # the levels derived next to the level converted from the input data
    assert set(dt[GROUP].children) == {"3", "2", "1"}
    assert dt[GROUP]["3"].sizes["cells"] == cell_ids.size
    assert dt[GROUP]["2"].sizes["cells"] == cell_ids.size // 4
    assert dt[GROUP]["1"].sizes["cells"] == cell_ids.size // 16

    # the multiscale group declares the whole pyramid, and keeps its attributes
    multiscales = dt[GROUP].attrs["multiscales"]
    assert _layout_assets(multiscales) == ["3", "2", "1"]
    assert [item["dggs"]["refinement_level"] for item in multiscales["layout"]] == [
        3,
        2,
        1,
    ]
    assert dt[GROUP].attrs["description"] == "converted data"
    assert {c["name"] for c in dt[GROUP].attrs["zarr_conventions"]} == {
        "multiscales",
        "dggs",
    }

    # each derived group describes its own HEALPix grid
    for level in (2, 1):
        group = dt[GROUP][str(level)]
        assert group.attrs["dggs"] == _healpix(level).model_dump()
        assert group[CF_GRID_MAPPING_VARIABLE].attrs == get_cf_grid_mapping_attrs(
            _healpix(level), METADATA
        )

    # the dataset (STAC item included) is otherwise left as it is
    assert dt.attrs["stac_discovery"] == STAC_ITEM


def test_consolidated_metadata_is_up_to_date(converted):
    path, cell_ids, *_ = converted
    cache, settings = _cache_and_settings([3], {"derived_levels": [2, 1]})

    generate_pyramids(cache, settings, path)

    # the derived levels are written next to the consolidated metadata of the
    # hand-written dataset, which is then stale: the stage does not refresh it
    # (and never reads it either)
    dt = xr.open_datatree(path, engine="zarr", consolidated=True)
    assert set(dt[GROUP].children) == {"3"}

    finalize_healpix_dataset(cache, settings, path)

    # the last stage of the conversion does: a reader that uses the
    # consolidated metadata of the dataset then sees the derived levels
    dt = xr.open_datatree(path, engine="zarr", consolidated=True)
    assert set(dt[GROUP].children) == {"3", "2", "1"}


def test_native_levels_are_not_rewritten(tmp_path):
    """The converted levels are not touched, so the codecs they are written
    with, lossy ones included, are kept as they are.
    """
    cell_ids = _dense_cells(3)
    values = np.arange(cell_ids.size, dtype="float64")
    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, a=values)},
        encoding={
            f"/{GROUP}/3": {
                "a": {
                    # a lossy filter: the values are stored as hundredths
                    "filters": [
                        {
                            "name": "numcodecs.fixedscaleoffset",
                            "configuration": {
                                "offset": 0,
                                "scale": 100,
                                "dtype": "<f8",
                                "astype": "<i4",
                            },
                        }
                    ],
                    "compressors": [{"name": "zstd", "configuration": {"level": 9}}],
                }
            }
        },
    )
    native = zarr.open_group(path, mode="r")[f"{GROUP}/3"]["a"]
    metadata = native.metadata.to_dict()
    chunk = native.store.get_sync(f"{GROUP}/3/a/c/0")

    _generate_pyramids(*_cache_and_settings([3], {"derived_levels": [2]}), path)

    rewritten = zarr.open_group(path, mode="r")[f"{GROUP}/3"]["a"]
    assert rewritten.metadata.to_dict() == metadata
    assert rewritten.store.get_sync(f"{GROUP}/3/a/c/0") == chunk

    # the derived level aggregates the values as they are stored
    level2 = xr.open_zarr(path, group=f"{GROUP}/2")
    np.testing.assert_allclose(
        level2["a"].values, values.reshape(-1, 4).mean(axis=1), atol=0.01
    )


def test_derived_levels_are_block_means(converted):
    path, cell_ids, temperature, counts = converted

    _generate_pyramids(*_cache_and_settings([3], {"derived_levels": [2, 1, 0]}), path)

    for level in (2, 1, 0):
        actual = xr.open_zarr(path, group=f"{GROUP}/{level}")
        parents = cell_ids >> np.uint64(2 * (3 - level))
        expected_cell_ids = np.unique(parents)
        np.testing.assert_array_equal(actual["cell_ids"].values, expected_cell_ids)

        for cell_id, mean, count in zip(
            expected_cell_ids, actual["temperature"].values, actual["counts"].values
        ):
            # the mean of all the cells of the native level, not a mean of means
            children = parents == cell_id
            np.testing.assert_allclose(mean, np.nanmean(temperature[children]))
            # an integer variable keeps its data type, rounded (not truncated)
            assert count == np.rint(np.mean(counts[children])).astype("int16")


def test_block_mean_differs_from_mean_of_means(tmp_path):
    # one level-1 cell whose first level-2 cell has a single valid level-3 cell:
    # the mean of means would weigh that cell as much as the 4 valid cells of
    # the other level-2 cells
    cell_ids = _dense_cells(3)
    values = np.ones(cell_ids.size)
    values[1:4] = np.nan
    values[0] = 9.0
    path = _write_converted(
        tmp_path / "converted.zarr", {3: _level_dataset(3, cell_ids, a=values)}
    )

    _generate_pyramids(*_cache_and_settings([3], {"derived_levels": [2, 1]}), path)

    level1 = xr.open_zarr(path, group=f"{GROUP}/1")
    np.testing.assert_allclose(level1["a"].values[0], np.nanmean(values[:16]))
    assert level1["a"].values[0] != np.nanmean(
        [np.nanmean(values[i : i + 4]) for i in range(0, 16, 4)]
    )


def test_reduce_method(converted):
    path, cell_ids, temperature, _ = converted

    _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "reduce": "max"}), path
    )

    level2 = xr.open_zarr(path, group=f"{GROUP}/2")
    parents = cell_ids >> np.uint64(2)
    expected = [np.nanmax(temperature[parents == cell]) for cell in np.unique(parents)]
    np.testing.assert_allclose(level2["temperature"].values, expected)


def test_reduce_mode(tmp_path):
    # a classification at the level 3, aggregated onto the level 2
    cell_ids = _dense_cells(3)
    classes = np.random.default_rng(0).integers(0, 4, cell_ids.size).astype("uint8")
    path = _write_converted(
        tmp_path / "converted.zarr", {3: _level_dataset(3, cell_ids, scl=classes)}
    )

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "reduce": "mode"}), path
    )

    level2 = dt[f"{GROUP}/2"]["scl"]
    assert level2.dtype == np.dtype("uint8")
    # the most frequent class of the 4 children, the smallest one on a tie
    expected = [
        min(np.flatnonzero(counts == counts.max()))
        for counts in (
            np.bincount(block, minlength=4) for block in classes.reshape(-1, 4)
        )
    ]
    np.testing.assert_array_equal(level2.values, expected)


@pytest.mark.parametrize("fill_value", [None, 255])
def test_reduce_bitwise_or(tmp_path, fill_value):
    # bit flags at the level 3, aggregated onto the level 2
    cell_ids = _dense_cells(3)
    rng = np.random.default_rng(0)
    # each bit set in about one cell out of five: often in a minority of the
    # children of a cell
    bits = rng.random((cell_ids.size, 4)) < 0.2
    flags = (bits * (1 << np.arange(4))).sum(axis=1).astype("uint8")
    missing = np.zeros(cell_ids.size, dtype=bool)
    if fill_value is not None:
        # some cells without data
        missing = rng.random(cell_ids.size) < 0.3
        flags[missing] = fill_value
    ds = _level_dataset(3, cell_ids, flags=flags)
    if fill_value is not None:
        ds["flags"].attrs["_FillValue"] = fill_value
    path = _write_converted(tmp_path / "converted.zarr", {3: ds})

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "reduce": "bitwise_or"}),
        path,
    )

    assert dt[f"{GROUP}/2"]["flags"].encoding["dtype"] == np.dtype("uint8")
    # every flag set in one of the 4 children, the missing ones skipped; the
    # fill value where all the children are missing
    blocks = np.where(missing, 0, flags).reshape(-1, 4)
    expected = np.bitwise_or.reduce(blocks, axis=1)
    if fill_value is not None:
        expected[missing.reshape(-1, 4).all(axis=1)] = fill_value
    else:
        # no fill value: a block of cells without any flag gives 0, not missing
        assert (expected == 0).any()
    stored = zarr.open_array(path, path=f"{GROUP}/2/flags")[:]
    assert stored.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(stored, expected)


def test_derived_chunks_are_healpix_cells(converted):
    path, *_ = converted
    cache, settings = _cache_and_settings([3], {"derived_levels": [2, 1]})

    dt = _generate_pyramids(cache, settings, path)

    # the chunks hold the cells of one HEALPix cell, as the staging cache says
    info = cache.output_groups.pyramid_multiscale_groups[PurePath(GROUP)]
    for level_path, (healpix, chunk_level) in info.derived_levels.items():
        written = dt[level_path.as_posix()]["temperature"].encoding["chunks"]
        assert written == (4 ** (healpix.refinement_level - chunk_level),)


def test_derived_chunk_level():
    # a level-20 conversion with dense chunks of one level-12 cell each: the
    # chunks of a derived level hold as many cells as the converted ones
    # (4 ** 8 cells, the children of a cell 8 levels above)
    assert _derived_chunk_level(19, 20, 12) == 11
    assert _derived_chunk_level(13, 20, 12) == 5
    # ... until the chunk cells are the base cells
    assert _derived_chunk_level(8, 20, 12) == 0
    assert _derived_chunk_level(4, 20, 12) == 0

    # a conversion with small chunks still gets reasonably sized ones
    assert _derived_chunk_level(18, 20, 19) == 18 - MIN_CHUNK_DEPTH

    # a partial extent: the chunk cells stay at or below the coarsest level the
    # converted cells cover entirely (here the conversion chunk level, 12)
    assert _derived_chunk_level(19, 20, 12, coarsest_level=12) == 12
    assert _derived_chunk_level(13, 20, 12, coarsest_level=12) == 12


@pytest.mark.parametrize(
    ["extent", "native", "derived"],
    [
        # 2 of the 4 level-1 chunk cells of a level-0 cell
        pytest.param(
            shapely.box(0.0, 0.0, 20.0, 20.0), 6, [4, 2, 1], id="part-of-a-base-cell"
        ),
        # parts of several base cells
        pytest.param(
            shapely.box(-60.0, -10.0, 60.0, 10.0), 7, [6, 5], id="several-base-cells"
        ),
    ],
)
def test_derived_chunks_are_healpix_cells_on_a_partial_extent(
    tmp_path, extent, native, derived
):
    cell_ids = _dense_cells(native, extent)
    path = _write_converted(
        tmp_path / "converted.zarr",
        {
            native: _level_dataset(
                native, cell_ids, a=np.arange(cell_ids.size, dtype="f8")
            )
        },
    )
    cache, settings = _cache_and_settings([native], {"derived_levels": derived}, extent)

    dt = _generate_pyramids(cache, settings, path)

    # each chunk holds all the children of one HEALPix cell, and those only
    info = cache.output_groups.pyramid_multiscale_groups[PurePath(GROUP)]
    for level_path, (healpix, chunk_level) in info.derived_levels.items():
        assert chunk_level >= CHUNK_LEVEL
        depth = healpix.refinement_level - chunk_level
        level = dt[level_path.as_posix()]
        assert level["a"].encoding["chunks"] == (4**depth,)

        ids = level["cell_ids"].values.reshape(-1, 4**depth)
        parents = ids >> np.uint64(2 * depth)
        assert (parents == parents[:, :1]).all()
        np.testing.assert_array_equal(
            ids - (parents[:, :1] << np.uint64(2 * depth)),
            np.broadcast_to(np.arange(4**depth, dtype="uint64"), ids.shape),
        )


class TestPyramidCodecs:
    def test_derived_variables_use_the_codecs(self, converted):
        path, cell_ids, temperature, _ = converted
        codecs = [
            {"name": "bytes", "configuration": {}},
            {"name": "zstd", "configuration": {"level": 5}},
        ]

        dt = _generate_pyramids(
            *_cache_and_settings([3], {"derived_levels": [2], "codecs": codecs}), path
        )

        level2 = zarr.open_group(path, mode="r")[f"{GROUP}/2"]
        written = level2["temperature"].metadata.to_dict()["codecs"]
        assert [c["name"] for c in written] == ["bytes", "zstd"]
        assert written[1]["configuration"]["level"] == 5
        # the cell ids keep Zarr's defaults, the converted level its own codecs
        assert level2["cell_ids"].metadata.to_dict()["codecs"] != written
        native = zarr.open_group(path, mode="r")[f"{GROUP}/3"]["temperature"]
        assert native.metadata.to_dict()["codecs"] != written
        # same values as without codecs
        parents = cell_ids >> np.uint64(2)
        np.testing.assert_allclose(
            dt[f"{GROUP}/2"]["temperature"].values,
            _block_means(temperature, parents),
        )

    def test_unknown_codec(self):
        with pytest.raises(
            ValueError, match="zarr codec 'no-such-codec' is not available"
        ):
            _settings(
                [3],
                {
                    "derived_levels": [2],
                    "codecs": [{"name": "no-such-codec", "configuration": {}}],
                },
            )

    def test_healpix_compress_serializer(self, tmp_path):
        """A serializer that codes the cells of one HEALPix cell per chunk, as the
        chunks of a derived level are."""
        pytest.importorskip("hcc.codecs.sentinel2")
        cell_ids = _dense_cells(3)
        values = np.random.default_rng(0).uniform(0.0, 0.5, cell_ids.size)
        values[:64] = np.nan  # a whole level-0 cell without data
        path = _write_converted(
            tmp_path / "converted.zarr", {3: _level_dataset(3, cell_ids, a=values)}
        )
        # digital numbers DN = round(1e4 * a + 1000), lossless on that lattice
        codecs = [
            {
                "name": "hcc-sentinel2",
                "configuration": {"quantification_value": 1e4, "add_offset": -1000.0},
            }
        ]

        dt = _generate_pyramids(
            *_cache_and_settings([3], {"derived_levels": [2], "codecs": codecs}), path
        )

        level2 = zarr.open_group(path, mode="r")[f"{GROUP}/2"]["a"]
        assert level2.metadata.to_dict()["codecs"][0]["name"] == "hcc-sentinel2"
        expected = np.array(_block_means(values, cell_ids >> np.uint64(2)))
        written = dt[f"{GROUP}/2"]["a"].values
        np.testing.assert_array_equal(np.isnan(written), np.isnan(expected))
        valid = ~np.isnan(expected)
        assert np.abs(written[valid] - expected[valid]).max() <= 0.5e-4 + 1e-9


def test_pyramid_in_a_child_group(converted):
    path, cell_ids, temperature, _ = converted

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "subgroup_name": "pyramids"}),
        path,
    )

    # the derived levels are written in a child group of the multiscale
    # group, next to the levels converted from the input data
    assert set(dt[GROUP].children) == {"3", "pyramids"}
    assert set(dt[f"{GROUP}/pyramids"].children) == {"2"}

    # the multiscale group still describes the whole pyramid, with assets
    # pointing at its own children only
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == ["3", "pyramids/2"]

    parents = cell_ids >> np.uint64(2)
    expected = _block_means(temperature, parents)
    generated = dt[f"{GROUP}/pyramids/2"]["temperature"]
    np.testing.assert_allclose(generated.values, expected)


def test_multiple_native_levels(converted_multi_level):
    path, cell_ids, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2], {"derived_levels": [1]}, variables=_variable_names(data)
        ),
        path,
    )

    assert set(dt[GROUP].children) == {"3", "2", "1"}

    # the levels converted from the input data are left as they are
    level2 = dt[GROUP]["2"].to_dataset()
    np.testing.assert_allclose(level2["a"].values, data[2]["a"])
    np.testing.assert_allclose(level2["b"].values, data[2]["b"])

    # the level 1 is aggregated from the level 2 (the copy of "a" included,
    # "nearest" being the default)
    level1 = dt[GROUP]["1"].to_dataset()
    np.testing.assert_allclose(
        level1["a"].values, np.full(cell_ids[2].size // 4, 100.0)
    )
    np.testing.assert_allclose(
        level1["b"].values, data[2]["b"].reshape(-1, 4).mean(axis=1)
    )


def test_coarsen_from_finest(converted_multi_level):
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2],
            {"derived_levels": [1], "coarsen_from": "finest"},
            variables=_variable_names(data),
        ),
        path,
    )

    # the copy of "a" resampled by the provider is ignored: the level 1 is
    # aggregated from the level 3
    np.testing.assert_allclose(
        dt[GROUP]["1"]["a"].values, data[3]["a"].reshape(-1, 16).mean(axis=1)
    )


def test_sparse_chunks_are_not_supported():
    with pytest.raises(ValueError, match="requires the 'healpix_cell_dense' method"):
        _settings([3], {"derived_levels": [2, 1]}, chunk_method="sparse")


def test_partial_extent_coarsest_level(tmp_path):
    # an extent covering 2 of the 4 level-1 chunk cells of a level-0 cell: the
    # cells of the level 0 are not covered entirely
    extent = shapely.box(0.0, 0.0, 20.0, 20.0)
    cell_ids = _dense_cells(3, extent)
    path = _write_converted(
        tmp_path / "converted.zarr",
        {3: _level_dataset(3, cell_ids, a=np.ones(cell_ids.size))},
    )

    # down to the chunk level: fine
    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2, CHUNK_LEVEL]}, extent), path
    )
    assert set(dt[GROUP].children) == {"3", "2", "1"}

    # the staging cache is where that is checked, before anything is written
    with pytest.raises(ValueError, match="do not cover whole HEALPix cells"):
        _cache_and_settings([3], {"derived_levels": [0]}, extent)


def test_no_pyramid_settings(converted):
    path, *_ = converted

    # nothing to generate: the stage does nothing
    dt = _generate_pyramids(*_cache_and_settings([3]), path)

    assert set(dt[GROUP].children) == {"3"}


# --- the whole conversion workflow, from an actual input dataset


def _write_input(path):
    """An input dataset with two spatial groups, converted at two different
    HEALPix refinement levels.
    """
    lon = np.linspace(-180.0, 180.0, 64)
    lat = np.linspace(-80.0, 80.0, 32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    coords = {
        "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
        "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
    }
    fine = xr.Dataset(
        {"a": (("lat", "lon"), np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy)))},
        coords=coords,
    )
    coarse = xr.Dataset(
        {"b": (("lat", "lon"), np.cos(np.deg2rad(xx)) + 0.5)}, coords=coords
    )

    tree = xr.DataTree.from_dict({f"/{GROUP}/fine": fine, f"/{GROUP}/coarse": coarse})
    tree.attrs["stac_discovery"] = STAC_ITEM | {
        "geometry": shapely.geometry.mapping(FULL_EXTENT),
        "bbox": [-180.0, -90.0, 180.0, 90.0],
    }
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


def _workflow_settings(derived_levels=(1, 0), **group_kwargs) -> ConvertSettings:
    """Convert two input groups at the levels 3 and 2, with dense chunks of one
    level-1 cell each, and a pyramid with the given derived levels (other group
    settings given as keyword arguments).
    """
    chunk = {
        "method": "healpix_cell_dense",
        "healpix": {"refinement_level": CHUNK_LEVEL, **HEALPIX},
        "chunk_buffer_width": 0.0,
    }
    group_settings = {}

    for name, level in (("fine", 3), ("coarse", 2)):
        group_settings[f"{GROUP}/{name}"] = {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
            **group_kwargs,
        }

    return ConvertSettings.model_validate(
        {
            "multiscale_settings": {
                "groups": [GROUP],
                "pyramid_groups": {GROUP: {"derived_levels": list(derived_levels)}},
            },
            "group_settings": group_settings,
        }
    )


@pytest.fixture
def prepared(tmp_path):
    """An output dataset prepared from that input: all its groups, arrays and
    metadata, with no data yet.
    """
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")
    settings = _workflow_settings()

    cache = create_staging_cache([input_path], settings)
    prepare_healpix_dataset(cache, settings, output_path)

    return cache, settings, output_path


@pytest.fixture
def converted_workflow(prepared):
    """... and its data converted (but no derived level aggregated yet)."""
    cache, settings, output_path = prepared

    for group in (f"{GROUP}/fine", f"{GROUP}/coarse"):
        convert_group_to_healpix(
            group, cache=cache, settings=settings, output_path=output_path
        )

    return cache, settings, output_path


def test_prepare_creates_the_derived_level_groups(prepared):
    cache, settings, output_path = prepared
    root = zarr.open_group(output_path, mode="r")

    # the derived levels are groups of the output dataset, like the converted
    # ones, and the multiscale group describes the whole pyramid: "a" is
    # aggregated from the level 3 and "b" from the level 2, which one item per
    # level cannot describe, so every item refers to an array
    assert sorted(root[GROUP].group_keys()) == ["0", "1", "2", "3"]
    assert _layout_assets(root[GROUP].attrs["multiscales"]) == [
        "3/a",
        "2/b",
        "1/a",
        "1/b",
        "0/a",
        "0/b",
    ]

    for level in (1, 0):
        group = root[f"{GROUP}/{level}"]
        assert group.attrs["dggs"]["refinement_level"] == level
        # ... with no array yet: those are written with the aggregated data
        assert list(group.array_keys()) == []


def test_prepare_dry_run_creates_the_derived_level_groups(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    settings = _workflow_settings()
    cache = create_staging_cache([input_path], settings)

    dt = prepare_healpix_dataset(
        cache, settings, str(tmp_path / "converted.zarr"), dry_run=True
    )

    # the structure of the output dataset, derived level groups included,
    # without
    # writing anything
    assert set(dt[GROUP].children) == {"3", "2", "1", "0"}
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == [
        "3/a",
        "2/b",
        "1/a",
        "1/b",
        "0/a",
        "0/b",
    ]
    assert not dt[f"{GROUP}/0"].to_dataset().variables
    assert not (tmp_path / "converted.zarr").exists()


def test_create_healpix_dataset_fills_the_pyramids(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")

    dt = create_healpix_dataset(
        [input_path], _workflow_settings(derived_levels=[1]), output_path
    )

    # a conversion in one call aggregates the derived levels as well
    assert set(dt[GROUP].children) == {"3", "2", "1"}
    converted = dt[GROUP]["3"].to_dataset()
    parents = converted["cell_ids"].values >> np.uint64(4)
    expected = _block_means(converted["a"].values, parents)
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, expected, atol=1e-12)


def test_derived_levels_have_the_dtype_of_the_converted_level(tmp_path):
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")

    dt = create_healpix_dataset(
        [input_path],
        _workflow_settings(derived_levels=[1], output_dtype="float16"),
        output_path,
    )

    converted = dt[GROUP]["3"].to_dataset()
    assert converted["a"].dtype == "float16"
    assert dt[GROUP]["1"]["a"].dtype == "float16"
    parents = converted["cell_ids"].values >> np.uint64(4)
    expected = _block_means(converted["a"].values.astype("float64"), parents)
    np.testing.assert_allclose(dt[GROUP]["1"]["a"].values, expected, rtol=1e-3)


def test_conversion_workflow(converted_workflow):
    cache, settings, output_path = converted_workflow

    dt = generate_pyramids(cache, settings, output_path)

    # the converted levels (3 and 2) and the derived ones (1 and 0)
    assert set(dt[GROUP].children) == {"3", "2", "1", "0"}
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == [
        "3/a",
        "2/b",
        "1/a",
        "1/b",
        "0/a",
        "0/b",
    ]

    converted = dt[GROUP]["3"].to_dataset()
    for level in (1, 0):
        generated = dt[GROUP][str(level)].to_dataset()
        parents = converted["cell_ids"].values >> np.uint64(2 * (3 - level))
        np.testing.assert_array_equal(generated["cell_ids"].values, np.unique(parents))
        # "a" is aggregated from the level 3 and "b" from the level 2
        assert set(generated.data_vars) == {"a", "b", CF_GRID_MAPPING_VARIABLE}
        expected = _block_means(converted["a"].values, parents)
        np.testing.assert_allclose(generated["a"].values, expected, atol=1e-12)


def test_pyramid_structure_comes_from_the_staging_cache(converted_workflow):
    """The staging cache holds the structure of the pyramids: the levels, their
    groups, their HEALPix grid and their chunks.
    """
    cache, settings, output_path = converted_workflow
    info = cache.output_groups.pyramid_multiscale_groups[PurePath(GROUP)]
    # the output dataset is not finalized here, so its metadata is not
    # consolidated yet
    dt = utils.open_datatrees([output_path], chunks={}, consolidated=False)[0]

    # the multiscales metadata of the group describes the whole pyramid
    multiscales = cache.output_groups.multiscale_groups[PurePath(GROUP)]
    assert _layout_assets(multiscales) == [
        "3/a",
        "2/b",
        "1/a",
        "1/b",
        "0/a",
        "0/b",
    ]

    # the converted levels, with the HEALPix grid and the chunks they were
    # converted with
    assert {path.as_posix() for path in info.native_levels} == {
        f"{GROUP}/3",
        f"{GROUP}/2",
    }
    for level_path, (healpix, chunk_level) in info.native_levels.items():
        written = dt[level_path.as_posix()]["cell_ids"]
        assert written.size == 12 * 4**healpix.refinement_level
        assert chunk_level == CHUNK_LEVEL
        # the cells of the level are the children of its chunk cells
        chunk_cells = _chunk_cells(
            cache.output_spatial.geometry_latlon,
            settings.group_settings[PurePath(f"{GROUP}/fine")].chunk,
        )
        np.testing.assert_array_equal(
            hpn.zoom_to(chunk_cells, CHUNK_LEVEL, healpix.refinement_level).ravel(),
            written.values,
        )
        # a full-earth extent covers whole HEALPix cells at every level
        assert _coarsest_whole_level(chunk_cells, CHUNK_LEVEL) == 0

    # the derived levels, with the grid and the chunks they are written with
    assert {path.as_posix() for path in info.derived_levels} == {
        f"{GROUP}/1",
        f"{GROUP}/0",
    }
    for level_path, (healpix, chunk_level) in info.derived_levels.items():
        assert level_path == PurePath(GROUP) / str(healpix.refinement_level)
        assert chunk_level == 0

    # the cache, which the conversion stages are given, round-trips
    assert ConvertStagingCache.from_json(cache.to_json()).output_groups == (
        cache.output_groups
    )


def test_derived_levels_declare_what_they_are_derived_from(converted):
    path, *_ = converted

    dt = _generate_pyramids(*_cache_and_settings([3], {"derived_levels": [2, 1]}), path)
    multiscales = dt[GROUP].attrs["multiscales"]

    # a reader can tell which levels hold converted data and which ones were
    # aggregated from them, and how
    assert "derived_from" not in _layout(multiscales, "3")
    assert "transform" not in _layout(multiscales, "3")

    for level, scale in ((2, 2.0), (1, 4.0)):
        item = _layout(multiscales, str(level))
        # every derived level is aggregated from the converted level itself,
        # not from the derived level above it
        assert item["derived_from"] == "3"
        # a cell of the derived level is `scale` level-3 cells long, so it
        # aggregates `scale ** 2` of them
        assert item["transform"] == {"scale": [scale]}
        assert item["resampling_method"] == "mean"


def test_multiscales_declares_the_resampling_method(converted):
    path, *_ = converted

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "reduce": "max"}), path
    )
    multiscales = dt[GROUP].attrs["multiscales"]

    # the same method aggregates every derived level of the pyramid, so it is
    # also given once for the whole group
    assert multiscales["resampling_method"] == "max"
    assert _layout(multiscales, "2")["resampling_method"] == "max"


def test_transform_scale_follows_the_source_level(converted_multi_level):
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2], {"derived_levels": [1]}, variables=_variable_names(data)
        ),
        path,
    )
    multiscales = dt[GROUP].attrs["multiscales"]

    # "nearest" (the default) aggregates every variable of the level 1 from
    # the level 2, one level away, and not from the level 3, two levels away:
    # one item describes the whole level
    assert _layout_assets(multiscales) == ["3", "2", "1"]

    item = _layout(multiscales, "1")
    assert item["derived_from"] == "2"
    assert item["transform"] == {"scale": [2.0]}


def test_arrays_aggregated_from_different_native_levels(converted_multi_level):
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2],
            {"derived_levels": [1], "coarsen_from": "finest"},
            variables=_variable_names(data),
            add_positioning=True,
        ),
        path,
    )
    multiscales = dt[GROUP].attrs["multiscales"]

    # "a" is aggregated from the level 3 ("finest" ignores the copy of the
    # level 2) and "b" from the level 2, its only converted level: one item
    # per level could not describe both, so every item refers to an array
    # instead (the convention asks for one referencing style per layout)
    assert _layout_assets(multiscales) == ["3/a", "2/a", "2/b", "1/a", "1/b"]

    item_a = _layout(multiscales, "1/a")
    assert item_a["derived_from"] == "3/a"
    assert item_a["transform"] == {"scale": [4.0]}
    assert item_a["resampling_method"] == "mean"

    item_b = _layout(multiscales, "1/b")
    assert item_b["derived_from"] == "2/b"
    assert item_b["transform"] == {"scale": [2.0]}

    # both arrays are at the same refinement level, which their absolute
    # positioning says and their asset paths share
    assert item_a["dggs"] == item_b["dggs"]


def test_absolute_positioning_is_off_by_default(converted):
    path, *_ = converted

    dt = _generate_pyramids(*_cache_and_settings([3], {"derived_levels": [2]}), path)
    multiscales = dt[GROUP].attrs["multiscales"]

    # repeating the grid of a level in its layout items would be verbose for
    # nothing: a consumer reads it in the attributes of the level group, which
    # the asset of an item names
    assert all("dggs" not in item for item in multiscales["layout"])
    assert _layout_assets(multiscales) == ["3", "2"]
    assert dt[GROUP]["2"].attrs["dggs"]["refinement_level"] == 2


def test_asset_type_array_describes_every_array(converted):
    path, *_ = converted

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3],
            {"derived_levels": [2]},
            variables={3: ["counts", "temperature"]},
            asset_type="array",
        ),
        path,
    )
    multiscales = dt[GROUP].attrs["multiscales"]

    # every level is described array by array, the native ones included, even
    # though one item per level would do here
    assert _layout_assets(multiscales) == [
        "3/counts",
        "3/temperature",
        "2/counts",
        "2/temperature",
    ]

    item = _layout(multiscales, "2/temperature")
    assert item["derived_from"] == "3/temperature"
    assert item["transform"] == {"scale": [2.0]}


def test_asset_type_group_rejects_arrays_of_different_native_levels(
    converted_multi_level,
):
    _, _, data = converted_multi_level

    # the level 1 would aggregate "a" from the level 3 and "b" from the level
    # 2, which its single layout item cannot describe
    with pytest.raises(ValueError, match="aggregated from different native levels"):
        _cache_and_settings(
            [3, 2],
            {"derived_levels": [1], "coarsen_from": "finest"},
            variables=_variable_names(data),
            asset_type="group",
        )


def test_asset_type_group_keeps_the_level_groups(converted_multi_level):
    path, _, data = converted_multi_level

    dt = _generate_pyramids(
        *_cache_and_settings(
            [3, 2],
            {"derived_levels": [1]},
            variables=_variable_names(data),
            asset_type="group",
        ),
        path,
    )

    # "nearest" aggregates every variable of the level 1 from the level 2, so
    # one item per level describes the pyramid
    assert _layout_assets(dt[GROUP].attrs["multiscales"]) == ["3", "2", "1"]


def test_derived_from_is_relative_to_the_multiscale_group(converted):
    path, *_ = converted

    dt = _generate_pyramids(
        *_cache_and_settings([3], {"derived_levels": [2], "subgroup_name": "pyramids"}),
        path,
    )
    item = _layout(dt[GROUP].attrs["multiscales"], "pyramids/2")

    # like the assets, the source is a path within the multiscale group
    assert item["derived_from"] == "3"
    assert item["transform"] == {"scale": [2.0]}


def test_multiscales_of_a_group_without_a_pyramid():
    levels = {PurePath(f"{GROUP}/{level}"): _healpix(level) for level in (3, 2)}

    multiscales = _multiscales_metadata(PurePath(GROUP), levels, True).model_dump()

    # nothing is derived, so the convention fields describing it are absent
    assert "resampling_method" not in multiscales
    for asset in ("3", "2"):
        assert "derived_from" not in _layout(multiscales, asset)
        assert "transform" not in _layout(multiscales, asset)
        assert "resampling_method" not in _layout(multiscales, asset)

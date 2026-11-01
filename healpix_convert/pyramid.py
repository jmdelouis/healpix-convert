"""Writing the pyramids of coarser HEALPix refinement levels.

The last stage of the conversion aggregates the levels of the multiscale groups
of the output dataset onto coarser HEALPix refinement levels (the derived
levels), each written in a group of the multiscale group (the converted levels
are never rewritten).

The structure of the pyramids — which levels are derived, in which groups,
with which HEALPix grid and chunks, and the multiscales metadata describing
them — is computed once when the conversion staging cache is created (see
:py:class:`~healpix_convert.core.conversion_models.PyramidMultiscaleGroupInfo`).
What is left here is writing it: :py:func:`create_derived_level_groups` writes
the (empty) groups and their metadata when the output dataset is prepared, like
the rest of its structure, and :py:func:`write_derived_levels` aggregates the
converted data onto them, which creates and fills their arrays.

Only groups converted with the ``healpix_cell_dense`` chunk method are
supported (the conversion settings enforce it): the cells of such a chunk are
all the children of one HEALPix cell at a coarser ("chunk") level, in ascending
(nested) order, so the cells of a coarser cell are consecutive cells of the
converted arrays. Aggregating them is then a matter of position (see
:py:func:`~healpix_convert.core.coarsen.coarsen_healpix_group`): nothing has to
be grouped and no cell id is ever read.

"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import PurePath
from typing import Any

import dask
import numpy as np
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

import healpix_convert.core.utils as utils
from healpix_convert.core.coarsen import (
    coarsen_healpix_group,
    codecs_encoding,
    encode_like,
    select_source_levels,
)
from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    HealpixLevelInfo,
    PyramidMultiscaleGroupInfo,
)
from healpix_convert.core.healpix_conventions import CFHealpixGridMapping, Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_healpix_group_attrs,
)
from healpix_convert.settings.common import ConvertSettings, PyramidSettings

log = structlog.get_logger()


def create_derived_level_groups(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
) -> None:
    """Create the (empty) Zarr groups of the derived levels.

    The arrays of those groups are created when the converted data is
    aggregated onto them (see :py:func:`write_derived_levels`). The multiscales
    metadata describing a pyramid is written with the multiscale group that
    holds it, like the metadata of any other output group.

    Parameters
    ----------
    output_store : Zarr store
        Store of the output Zarr dataset.
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the
        pyramids.
    settings : :py:class:`ConvertSettings`
        The conversion settings (the conventions written here).

    """
    for path, info in cache.output_groups.pyramid_multiscale_groups.items():
        for level_path, (healpix, _) in info.derived_levels.items():
            # the group named by the pyramid settings, if any, is created along
            # the way
            group = zarr.open_group(output_store, path=level_path.as_posix(), mode="a")
            group.attrs.update(get_healpix_group_attrs(healpix, settings.metadata))

        log.info(
            f"••• created the derived level groups of '{path.as_posix()}': "
            f"levels {_levels(info.derived_levels)}"
        )


def write_derived_levels(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    *,
    output_storage_options: dict[str, Any] | None = None,
    direct: bool = False,
    loader: Callable[[str], np.ndarray] | None = None,
) -> None:
    """Aggregate the converted data onto the derived levels of the pyramids
    and write it in their groups, creating their arrays.

    With ``direct``, the block means (``reduce="mean"``) are computed in numpy
    rather than by one Dask graph per derived level: each converted variable
    is read (decoded) once, and every derived level is aggregated from the
    block sums and counts of the finer ones (see :py:class:`_BlockMeans`).
    Same block means, summed in another order. ``loader``, if given, returns
    the decoded values of a converted array from its path in the store (e.g.
    decoded in other processes), instead of reading it through Xarray.

    The groups themselves and the multiscales metadata of the groups holding
    them are written by :py:func:`create_derived_level_groups`. The converted
    levels are not rewritten.

    Parameters
    ----------
    output_store : Zarr store
        Store of the output Zarr dataset, with the converted groups written.
    cache : :py:class:`ConvertStagingCache`
        The conversion staging cache, which holds the structure of the
        pyramids.
    settings : :py:class:`ConvertSettings`
        The conversion settings.
    output_storage_options : dict, optional
        Storage options passed to Zarr.

    """
    # the groups of the output dataset are read from the store itself: its
    # consolidated metadata, if any, is not up to date while it is written
    dt = utils.open_datatrees(
        [output_store],
        chunks={},
        # a store object already holds its storage options: Zarr refuses them
        # next to it (they are only used to open a URL)
        storage_options=(
            output_storage_options if isinstance(output_store, str) else None
        ),
        consolidated=False,
    )[0]

    writes = []

    for path, info in cache.output_groups.pyramid_multiscale_groups.items():
        pyramid = settings.multiscale_settings.pyramid_groups[path]
        log.info(
            f"••• aggregating the derived levels of '{path.as_posix()}' "
            f"({pyramid.reduce} of the converted levels "
            f"{_levels(info.native_levels)}): levels {_levels(info.derived_levels)}"
        )

        for level_path, ds in _build_derived_levels(
            dt, info, pyramid, settings, direct=direct, loader=loader
        ).items():
            writes.append(
                ds.to_zarr(
                    output_store,
                    group=level_path.as_posix(),
                    mode="a",
                    zarr_format=3,
                    # the Zarr metadata of the output dataset is consolidated
                    # once, by the conversion stage that calls this
                    consolidated=False,
                    compute=False,
                )
            )

    # all the levels are written in one Dask computation: a converted level is
    # read once and feeds every level aggregated from it
    dask.compute(*writes)


def _levels(levels: Mapping[PurePath, HealpixLevelInfo]) -> list[int]:
    """Refinement levels of the given level groups, finest first."""
    return sorted(
        (healpix.refinement_level for healpix, _ in levels.values()),  # type: ignore[misc]
        reverse=True,
    )


def _native_datasets(
    dt: xr.DataTree, info: PyramidMultiscaleGroupInfo
) -> dict[int, xr.Dataset]:
    """The converted groups of a pyramid multiscale group, with lazy (Dask)
    arrays (refinement level -> dataset).
    """
    datasets = {}

    for level_path, (healpix, _) in info.native_levels.items():
        assert healpix.refinement_level is not None
        datasets[healpix.refinement_level] = dt[level_path.as_posix()].to_dataset(
            inherit=False
        )

    return datasets


def _variable_sources(
    datasets: Mapping[int, xr.Dataset], level: int, *, coarsen_from: str
) -> dict[int, list[str]]:
    """Converted level each data variable of a derived level is aggregated
    from (converted level -> variable names).

    Which level feeds a variable is decided by
    :py:func:`~healpix_convert.core.coarsen.select_source_levels`, the same
    rule the multiscales metadata of the pyramid describes.

    """
    selected = select_source_levels(
        {lvl: set(ds.data_vars) for lvl, ds in datasets.items()},
        level,
        coarsen_from=coarsen_from,
    )

    sources: dict[int, list[str]] = {}

    for name, source in selected.items():
        sources.setdefault(source, []).append(name)

    return sources


def _build_derived_levels(
    dt: xr.DataTree,
    info: PyramidMultiscaleGroupInfo,
    pyramid: PyramidSettings,
    settings: ConvertSettings,
    *,
    direct: bool = False,
    loader: Callable[[str], np.ndarray] | None = None,
) -> dict[PurePath, xr.Dataset]:
    """Build the datasets of the derived levels of one pyramid multiscale group
    (group path -> dataset).
    """
    datasets = _native_datasets(dt, info)
    native_healpix = {
        healpix.refinement_level: healpix for healpix, _ in info.native_levels.values()
    }

    # the data type and the fill value of a variable are the ones it has at its
    # finest converted level
    templates: dict[str, xr.Variable] = {}
    for level in sorted(datasets, reverse=True):
        for name, var in datasets[level].variables.items():
            templates.setdefault(name, var)

    derived = {}
    means = (
        _BlockMeans(
            datasets,
            native_healpix,
            loader=loader,
            paths={
                healpix.refinement_level: path.as_posix()
                for path, (healpix, _) in info.native_levels.items()
            },
        )
        if direct and pyramid.reduce == "mean"
        else None
    )

    for level_path, (healpix, chunk_level) in info.derived_levels.items():
        assert healpix.refinement_level is not None

        ds = _build_derived_level(
            datasets,
            healpix,
            native_healpix=native_healpix,
            pyramid=pyramid,
            templates=templates,
            # a chunk holds the children of one HEALPix cell of the chunk level
            chunk_size=4 ** (healpix.refinement_level - chunk_level),
            means=means,
        )
        ds = _set_cf_grid_mapping(ds, healpix)
        ds.attrs = get_healpix_group_attrs(healpix, settings.metadata)
        derived[level_path] = ds

    return derived


def _build_derived_level(
    datasets: Mapping[int, xr.Dataset],
    healpix: Healpix,
    *,
    native_healpix: Mapping[int, Healpix],
    pyramid: PyramidSettings,
    templates: Mapping[str, xr.Variable],
    chunk_size: int,
    means: _BlockMeans | None = None,
) -> xr.Dataset:
    """Build the dataset of one derived level, from the converted levels of its
    multiscale group.
    """
    level = healpix.refinement_level
    assert level is not None

    sources = _variable_sources(datasets, level, coarsen_from=pyramid.coarsen_from)
    parts = []

    for source, names in sources.items():
        part = coarsen_healpix_group(
            datasets[source][names],
            level,
            healpix=native_healpix[source],
            reduce=pyramid.reduce,
        )
        if means is not None:
            part = means.fill(part, source, level)
        parts.append(part)

    if len(parts) == 1:
        ds = parts[0]
    else:
        # the converted levels of a group share their chunk cells (same spatial
        # extent, same chunk level), so the cells of the level are the same
        # whichever level its variables are aggregated from: the parts need no
        # alignment, and the cell ids are not read to merge them
        ds = xr.merge(
            parts, join="override", compat="override", combine_attrs="override"
        )

    ds = encode_like(ds, templates)

    coordinate = healpix.coordinate
    if isinstance(coordinate, str) and coordinate in ds.coords:
        ds[coordinate].encoding.update(
            _cell_ids_encoding(templates[coordinate], chunk_size)
        )

    if pyramid.codecs is not None:
        # the aggregated variables only (not the cell ids coordinate)
        encoding = codecs_encoding(pyramid.codecs)
        for name, var in ds.data_vars.items():
            if healpix.spatial_dimension in var.dims:
                var.encoding.update(
                    _fit_inner_chunks(
                        encoding, var.dims.index(healpix.spatial_dimension), chunk_size
                    )
                )

    return ds.chunk({healpix.spatial_dimension: chunk_size})


def _cell_ids_encoding(template: xr.Variable, chunk_size: int) -> dict[str, Any]:
    """Zarr encoding of the cell ids of a derived level: the codecs of the cell
    ids of the converted level (e.g. a delta filter), on the chunks of the
    derived level.

    Sharded converted cell ids give sharded derived ones, one shard per chunk
    of the derived level, with the inner chunks of the converted level or the
    whole shard where it is smaller.

    """
    encoding = {
        key: value
        for key, value in template.encoding.items()
        if key in ("filters", "serializer", "compressors")
    }

    if template.encoding.get("shards") is not None:
        (inner,) = template.encoding["chunks"]
        encoding |= {"chunks": (min(inner, chunk_size),), "shards": (chunk_size,)}

    return encoding


def _fit_inner_chunks(
    encoding: Mapping[str, Any], axis: int, chunk_size: int
) -> dict[str, Any]:
    """The encoding of the ``codecs`` pyramid setting for a variable of a
    derived level whose chunks hold ``chunk_size`` cells along ``axis``.

    The inner chunks of a sharding serializer are the same at every level,
    while the chunks of the coarsest levels may hold fewer cells: there, the
    inner chunks are cut to the chunk (one inner chunk per shard).

    """
    from zarr.codecs import ShardingCodec

    serializer = encoding.get("serializer")
    if not isinstance(serializer, ShardingCodec):
        return dict(encoding)

    inner = list(serializer.chunk_shape)
    if inner[axis] <= chunk_size:
        return dict(encoding)

    inner[axis] = chunk_size
    fitted = ShardingCodec(
        chunk_shape=tuple(inner),
        codecs=serializer.codecs,
        index_codecs=serializer.index_codecs,
        index_location=serializer.index_location,
    )

    return {**encoding, "serializer": fitted}


def _set_cf_grid_mapping(ds: xr.Dataset, healpix: Healpix) -> xr.Dataset:
    """Describe the HEALPix grid of a derived level in the CF grid mapping
    variable of the converted group, if it has one.

    The grid mapping is propagated rather than selected in the metadata
    settings: the data variables of the converted levels point at it, and so do
    the aggregated ones.

    """
    if CF_GRID_MAPPING_VARIABLE not in ds:
        return ds

    ds[CF_GRID_MAPPING_VARIABLE].attrs = CFHealpixGridMapping.from_healpix(
        healpix
    ).model_dump()

    return ds


class _BlockMeans:
    """Block means of the converted variables of a pyramid multiscale group,
    computed in numpy: each variable is read once from its converted level,
    and the sums and the counts of its valid (non-NaN) cells are aggregated
    four children at a time, level after level, so that every derived level
    is the block mean of the converted cells it covers, as
    :py:func:`~healpix_convert.core.coarsen.coarsen_healpix_group` computes it
    (``reduce="mean"``), with the sums in another order.
    """

    def __init__(
        self,
        datasets: Mapping[int, xr.Dataset],
        native_healpix: Mapping[int, Healpix],
        *,
        loader: Callable[[str], np.ndarray] | None = None,
        paths: Mapping[int, str] | None = None,
    ):
        self.datasets = datasets
        self.native_healpix = native_healpix
        self.loader = loader
        self.paths = paths or {}
        # (variable, converted level, level) -> (sums, counts), cell axis last
        self.sums: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _sums(self, name: str, source: int, level: int) -> tuple[np.ndarray, np.ndarray]:
        key = (name, source, level)
        if key not in self.sums:
            if level == source:
                var = self.datasets[source][name]
                axis = var.dims.index(self.native_healpix[source].spatial_dimension)
                if self.loader is not None and source in self.paths:
                    data = self.loader(f"{self.paths[source]}/{name}")
                else:
                    data = var.values
                values = np.moveaxis(np.asarray(data, dtype=np.float64), axis, -1)
                valid = ~np.isnan(values)
                self.sums[key] = (np.where(valid, values, 0.0), valid.astype(np.int64))
            else:
                sums, counts = self._sums(name, source, level + 1)
                shape = (*sums.shape[:-1], sums.shape[-1] // 4, 4)
                self.sums[key] = (
                    sums.reshape(shape).sum(axis=-1),
                    counts.reshape(shape).sum(axis=-1),
                )
        return self.sums[key]

    def fill(self, part: xr.Dataset, source: int, level: int) -> xr.Dataset:
        """``part`` (one coarsened source of a derived level, with lazy data)
        with the data of its variables on the cells replaced by the block
        means, cast as :py:func:`~healpix_convert.core.coarsen.encode_like`
        casts them."""
        dim = self.native_healpix[source].spatial_dimension
        part = part.copy()
        for name, var in part.data_vars.items():
            if dim not in var.dims:
                continue
            sums, counts = self._sums(str(name), source, level)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            mean = np.moveaxis(mean, -1, var.dims.index(dim))
            if np.issubdtype(var.dtype, np.integer):
                fill_value = var.encoding.get("fill_value")
                mean = np.round(np.where(np.isnan(mean), 0 if fill_value is None else fill_value, mean))
            part[name] = var.copy(data=mean.astype(var.dtype))
        return part


def _decode_shard(meta: bytes, key: str, data: bytes) -> tuple[str, np.ndarray]:
    """The decoded values of one chunk object (a shard) of an array, from the
    array metadata and the object, through Zarr (run in another process)."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.storage import MemoryStore

    proto = default_buffer_prototype()
    store = MemoryStore(
        {"zarr.json": proto.buffer.from_bytes(meta), key: proto.buffer.from_bytes(data)}
    )
    array = zarr.open_array(store, mode="r")
    shape = array.metadata.chunk_grid.chunk_shape
    coords = [int(c) for c in key.split("/")[1:]]
    selection = tuple(
        slice(c * s, min((c + 1) * s, n)) for c, s, n in zip(coords, shape, array.shape)
    )
    return key, array[selection]


def decode_arrays_in_processes(
    objects: Mapping[str, bytes], pool: Any
) -> Callable[[str], np.ndarray]:
    """A loader for :py:func:`write_derived_levels`: the decoded values of an
    array of the in-memory dataset ``objects`` (key -> bytes), its chunk
    objects decoded by Zarr in the processes of ``pool`` (a
    ``concurrent.futures`` executor), with the cells holding the fill value
    as NaN, as Xarray reads them. Arrays whose chunk keys are not of the
    default ``c/...`` form are not handled (``KeyError``)."""

    def load(path: str) -> np.ndarray:
        meta = objects[f"{path}/zarr.json"]
        info = json.loads(meta)
        if info.get("chunk_key_encoding", {}).get("name", "default") != "default":
            raise KeyError(path)
        prefix = f"{path}/c/"
        keys = [key for key in objects if key.startswith(prefix)]
        dtype = np.dtype(np.float64 if "float" in str(info["data_type"]) else info["data_type"])
        fill = info.get("fill_value")
        out = np.full(info["shape"], np.nan if fill in (None, "NaN") else fill, dtype=dtype)
        futures = [
            pool.submit(_decode_shard, meta, key[len(path) + 1 :], objects[key])
            for key in keys
        ]
        shape = None
        for future in futures:
            key, values = future.result()
            if shape is None:
                shape = values.shape if values.ndim else ()
            coords = [int(c) for c in key.split("/")[1:]]
            chunk = info["chunk_grid"]["configuration"]["chunk_shape"]
            out[tuple(slice(c * s, c * s + n) for c, s, n in zip(coords, chunk, values.shape))] = values
        values = out.astype(np.float64)
        if fill not in (None, "NaN") and np.issubdtype(out.dtype, np.number):
            values[out == fill] = np.nan
        return values

    return load

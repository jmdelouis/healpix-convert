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

Chunk-local pyramids
--------------------

When every derived level of a pyramid is at or finer than the chunk level of
its native levels, each cell of a derived level aggregates the cells of one
converted chunk only. The chunk conversion tasks then aggregate the cells they
have just resampled (:py:func:`aggregate_chunk`), so the converted levels are
never read back. The derived chunks hold the cells of several converted chunks,
which the tasks cannot write concurrently: they write them in staging arrays
chunked like the converted levels (in the :py:data:`STAGING_GROUP` group of the
output dataset), created with the arrays of the derived levels when the output
dataset is prepared (:py:func:`create_derived_level_arrays`), and
:py:func:`assemble_chunk_local_levels` copies them into the derived levels and
removes them. The derived levels are the same as the ones
:py:func:`write_derived_levels` aggregates, except that with lossy codecs on
the converted levels they aggregate the resampled values rather than the
values decoded from the converted levels.

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
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePath
from typing import Any, NamedTuple

import dask
import numpy as np
import structlog
import xarray as xr
import zarr.api.synchronous as zarr
import zarr.storage
from xarray.backends.zarr import encode_zarr_variable
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import ZARR_JSON
from zarr.core.sync import sync

import healpix_convert.core.utils as utils
from healpix_convert.cache import _derived_level_sources
from healpix_convert.core.coarsen import (
    VALUE_ENCODING,
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

#: Group of the output dataset holding the derived level cells aggregated by the
#: chunk conversion tasks, until :py:func:`assemble_chunk_local_levels` copies
#: them into the derived levels and removes it.
STAGING_GROUP = "_healpix_convert_pyramid_staging"

#: Decoding of the converted levels, as :py:func:`utils.open_datatrees` opens
#: them for :py:func:`write_derived_levels`.
_DECODE_KWARGS: dict[str, Any] = {
    "decode_cf": True,
    "decode_times": False,
    "decode_timedelta": False,
    "create_default_indexes": False,
}


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
    skip: Collection[PurePath] = (),
) -> None:
    """Aggregate the converted data onto the derived levels of the pyramids
    and write it in their groups, creating their arrays.

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
    skip : collection of paths, optional
        Pyramid multiscale groups not to aggregate here (their derived levels
        are written by :py:func:`assemble_chunk_local_levels`).

    """
    paths = [
        path
        for path in cache.output_groups.pyramid_multiscale_groups
        if path not in skip
    ]
    if not paths:
        return

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

    for path in paths:
        info = cache.output_groups.pyramid_multiscale_groups[path]
        pyramid = settings.multiscale_settings.pyramid_groups[path]
        log.info(
            f"••• aggregating the derived levels of '{path.as_posix()}' "
            f"({pyramid.reduce} of the converted levels "
            f"{_levels(info.native_levels)}): levels {_levels(info.derived_levels)}"
        )

        for level_path, ds in _build_derived_levels(
            dt, info, pyramid, settings
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
) -> xr.Dataset:
    """Build the dataset of one derived level, from the converted levels of its
    multiscale group.
    """
    level = healpix.refinement_level
    assert level is not None

    sources = _variable_sources(datasets, level, coarsen_from=pyramid.coarsen_from)
    parts = []

    for source, names in sources.items():
        parts.append(
            coarsen_healpix_group(
                datasets[source][names],
                level,
                healpix=native_healpix[source],
                reduce=pyramid.reduce,
            )
        )

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


# --- chunk-local pyramids


class ChunkLocalTarget(NamedTuple):
    """A derived level that the chunks of one converted (native) level
    aggregate (see :py:func:`aggregate_chunk`)."""

    #: Path of the derived level group.
    level_path: PurePath
    #: HEALPix refinement level of the derived level.
    level: int
    #: The variables of the derived level aggregated from the native level.
    names: tuple[str, ...]
    #: How the cells are aggregated (the ``reduce`` pyramid setting).
    reduce: str
    #: Whether the native level writes the cell ids of the derived level (the
    #: finest native level of the pyramid does).
    coordinate: bool


def chunk_local_pyramid_groups(
    cache: ConvertStagingCache, settings: ConvertSettings
) -> list[PurePath]:
    """Pyramid multiscale groups whose derived levels the chunk conversion
    tasks aggregate.

    Each cell of their derived levels aggregates the cells of one converted
    chunk only: the native levels share one chunk level and no derived level is
    coarser than it.

    """
    groups = []

    for path, info in cache.output_groups.pyramid_multiscale_groups.items():
        if path not in settings.multiscale_settings.pyramid_groups:
            continue
        chunk_levels = {chunk_level for _, chunk_level in info.native_levels.values()}
        if len(chunk_levels) != 1:
            continue
        (chunk_level,) = chunk_levels
        if all(
            healpix.refinement_level is not None
            and healpix.refinement_level >= chunk_level
            for healpix, _ in info.derived_levels.values()
        ):
            groups.append(path)

    return groups


def _native_variables(cache: ConvertStagingCache) -> dict[PurePath, Collection[str]]:
    """Variables converted in each native level (output path -> names)."""
    io_path_map = cache.output_groups.io_path_map

    return {
        io_path_map[input_path]: group_spatial.spatial_arrays
        for input_path, group_spatial in cache.input_spatial_groups.items()
        if input_path in io_path_map
    }


def chunk_local_targets(
    cache: ConvertStagingCache, settings: ConvertSettings, native_path: PurePath
) -> list[ChunkLocalTarget]:
    """The derived levels the chunks of the native level ``native_path`` (an
    output group path) aggregate, if any.
    """
    targets = []

    for path in chunk_local_pyramid_groups(cache, settings):
        info = cache.output_groups.pyramid_multiscale_groups[path]
        if native_path not in info.native_levels:
            continue

        pyramid = settings.multiscale_settings.pyramid_groups[path]
        sources = _derived_level_sources(
            info, pyramid.coarsen_from, _native_variables(cache)
        )
        finest = max(
            info.native_levels,
            key=lambda level_path: info.native_levels[level_path][0].refinement_level,
        )

        for level_path, (healpix, _) in info.derived_levels.items():
            assert healpix.refinement_level is not None
            names = tuple(
                sorted(
                    name
                    for name, source in sources[level_path].items()
                    if source == native_path
                )
            )
            if names or native_path == finest:
                targets.append(
                    ChunkLocalTarget(
                        level_path=level_path,
                        level=healpix.refinement_level,
                        names=names,
                        reduce=pyramid.reduce,
                        coordinate=native_path == finest,
                    )
                )

    return targets


def _staging_path(level_path: PurePath, name: str) -> str:
    return f"{STAGING_GROUP}/{level_path.as_posix()}/{name}"


def create_derived_level_arrays(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    *,
    output_storage_options: dict[str, Any] | None = None,
) -> list[PurePath]:
    """Create the arrays of the derived levels of the chunk-local pyramids
    (see :py:func:`chunk_local_pyramid_groups`) and their staging arrays.

    The arrays are the ones :py:func:`write_derived_levels` would create, with
    no data yet; the variables not defined on the cells (e.g., the CF grid
    mapping) are written here. The converted levels must have their arrays.
    Returns the pyramid multiscale groups whose arrays are created.

    """
    groups = chunk_local_pyramid_groups(cache, settings)
    if not groups:
        return []

    dt = utils.open_datatrees(
        [output_store],
        chunks={},
        storage_options=(
            output_storage_options if isinstance(output_store, str) else None
        ),
        consolidated=False,
    )[0]

    # creating an array is a few store round trips (about 0.5 s per array on
    # object storage): the levels, then the staging arrays, are created in
    # threads, so that these round trips overlap
    def create_level(job):
        level_path, ds, dim = job
        # the variables on the cells are written by the chunk conversion
        # tasks (their Dask graph is never computed), the others here
        for var in ds.variables.values():
            if dim not in var.dims:
                var.load()
        ds.to_zarr(
            output_store,
            group=level_path.as_posix(),
            mode="a",
            zarr_format=3,
            consolidated=False,
            compute=False,
        )

    def create_staging(job):
        level_path, name, cells_per_chunk = job
        array = zarr.open_array(
            output_store, path=f"{level_path.as_posix()}/{name}", mode="r"
        )
        zarr.create_array(
            output_store,
            name=_staging_path(level_path, name),
            shape=array.shape,
            dtype=array.dtype,
            chunks=(*array.shape[:-1], cells_per_chunk),
            fill_value=array.fill_value,
            dimension_names=array.metadata.dimension_names,
            overwrite=True,
        )

    levels = []
    staging = []
    for path in groups:
        info = cache.output_groups.pyramid_multiscale_groups[path]
        pyramid = settings.multiscale_settings.pyramid_groups[path]
        (chunk_level,) = {chunk_level for _, chunk_level in info.native_levels.values()}

        for level_path, ds in _build_derived_levels(
            dt, info, pyramid, settings
        ).items():
            healpix, _ = info.derived_levels[level_path]
            assert healpix.refinement_level is not None
            dim = healpix.spatial_dimension
            levels.append((level_path, ds, dim))
            cells_per_chunk = 4 ** (healpix.refinement_level - chunk_level)
            staging.extend(
                (level_path, str(name), cells_per_chunk)
                for name, var in ds.variables.items()
                if dim in var.dims
            )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(utils.retry_store_conflicts(create_level), levels))
        list(pool.map(utils.retry_store_conflicts(create_staging), staging))

    for path in groups:
        log.info(
            f"••• created the derived level arrays of '{path.as_posix()}' "
            "(aggregated by the chunk conversion tasks)"
        )

    return groups


def chunk_local_native_levels(
    cache: ConvertStagingCache, native_path: PurePath
) -> list[PurePath]:
    """The native levels of the pyramid multiscale group(s) of the native level
    ``native_path``, finest first."""
    paths: list[PurePath] = []

    for info in cache.output_groups.pyramid_multiscale_groups.values():
        if native_path not in info.native_levels:
            continue
        for level_path, _ in sorted(
            info.native_levels.items(),
            key=lambda item: item[1][0].refinement_level,
            reverse=True,
        ):
            if level_path not in paths:
                paths.append(level_path)

    return paths


def chunk_local_templates(
    output_store: zarr.StoreLike, native_levels: Collection[PurePath]
) -> dict[str, xr.Variable]:
    """Data type and value encoding of the variables of the derived levels
    (the templates of :py:func:`~healpix_convert.core.coarsen.encode_like`):
    those of the finest of the ``native_levels`` (given finest first) that has
    them, as :py:func:`write_derived_levels` takes them.
    """
    templates: dict[str, xr.Variable] = {}

    for level_path in native_levels:
        ds = xr.open_dataset(
            output_store,  # type: ignore[arg-type]
            group=level_path.as_posix(),
            engine="zarr",
            chunks={},
            consolidated=False,
            **_DECODE_KWARGS,
        )
        for name, var in ds.variables.items():
            if name in templates:
                continue
            template = xr.Variable((), np.zeros((), dtype=var.dtype))
            template.encoding = {
                k: v for k, v in var.encoding.items() if k in VALUE_ENCODING
            }
            templates[str(name)] = template

    return templates


def staging_arrays(
    output_store: zarr.StoreLike,
    targets: Collection[ChunkLocalTarget],
    coordinate: str,
) -> dict[tuple[PurePath, str], zarr.Array] | None:
    """The staging arrays the chunks of a native level write, or None if the
    output dataset has none (it was not prepared for chunk-local pyramids)."""
    arrays = {}

    for target in targets:
        names = list(target.names) + ([coordinate] if target.coordinate else [])
        for name in names:
            try:
                arrays[(target.level_path, name)] = zarr.open_array(
                    output_store,
                    path=_staging_path(target.level_path, name),
                    mode="r+",
                )
            except FileNotFoundError:
                return None

    return arrays


def _decoded_chunk(
    arrays: Mapping[str, zarr.Array], values: Mapping[str, np.ndarray]
) -> xr.Dataset:
    """The cells of one converted chunk as :py:func:`write_derived_levels`
    reads them back from the converted level.

    ``values`` (the values written in ``arrays`` for this chunk) go through an
    in-memory copy of the arrays with their metadata (codecs and chunks
    included): a lossy codec gives back the values it decodes from the store,
    and Xarray decodes them as it decodes the converted level, into Dask arrays
    chunked like it.

    """
    store = zarr.storage.MemoryStore()
    zarr.open_group(store, mode="w", zarr_format=3)
    prototype = default_buffer_prototype()

    for name, value in values.items():
        metadata = json.loads(
            arrays[name].metadata.to_buffer_dict(prototype)[ZARR_JSON].to_bytes()
        )
        # the chunk only: the same chunks, codecs, fill value and attributes
        metadata["shape"] = list(value.shape)
        sync(
            store.set(
                f"{name}/{ZARR_JSON}",
                prototype.buffer.from_bytes(json.dumps(metadata).encode()),
            )
        )
        zarr.open_array(store, path=name, mode="r+")[...] = value

    # Dask arrays with the chunks of the converted level, as it is read back:
    # a block of cells spanning several of them is reduced chunk by chunk
    return xr.open_dataset(
        store,  # type: ignore[arg-type]
        engine="zarr",
        consolidated=False,
        chunks={},
        **_DECODE_KWARGS,
    )


def aggregate_chunk(
    arrays: Mapping[str, zarr.Array],
    values: Mapping[str, np.ndarray],
    *,
    chunk_index: int,
    healpix: Healpix,
    targets: Collection[ChunkLocalTarget],
    templates: Mapping[str, xr.Variable],
    staging: Mapping[tuple[PurePath, str], zarr.Array],
) -> None:
    """Aggregate the cells of one converted chunk onto the derived levels it is
    the source of, and write them in their staging arrays.

    ``values`` holds the values written in the arrays of the converted level
    for this chunk (cell ids included), with the cell dimension last.

    """
    ds = _decoded_chunk(arrays, values)
    coordinate = healpix.coordinate
    assert isinstance(coordinate, str)

    for target in targets:
        coarse = coarsen_healpix_group(
            ds[list(target.names)],
            target.level,
            healpix=healpix,
            reduce=target.reduce,  # type: ignore[arg-type]
        )
        coarse = encode_like(coarse, templates).compute(scheduler="synchronous")
        n_cells = coarse.sizes[healpix.spatial_dimension]
        region = slice(chunk_index * n_cells, (chunk_index + 1) * n_cells)

        names = list(target.names) + ([coordinate] if target.coordinate else [])
        for name in names:
            encoded = encode_zarr_variable(coarse[name].variable, name=name)
            staging[(target.level_path, name)][..., region] = np.asarray(encoded.values)


def assemble_chunk_local_levels(
    output_store: zarr.StoreLike,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
) -> list[PurePath]:
    """Copy the derived level cells aggregated by the chunk conversion tasks
    into the derived levels, then remove the staging arrays.

    Returns the pyramid multiscale groups written here (the ones with staging
    arrays).

    """
    root = zarr.open_group(output_store, mode="a")
    if STAGING_GROUP not in root:
        return []

    copies = []
    groups = []

    for path in chunk_local_pyramid_groups(cache, settings):
        info = cache.output_groups.pyramid_multiscale_groups[path]
        level_paths = [
            level_path
            for level_path in info.derived_levels
            if f"{STAGING_GROUP}/{level_path.as_posix()}" in root
        ]
        if len(level_paths) != len(info.derived_levels):
            continue

        log.info(
            f"••• writing the derived levels of '{path.as_posix()}' "
            "(aggregated by the chunk conversion tasks): levels "
            f"{_levels(info.derived_levels)}"
        )
        for level_path in level_paths:
            staged = root[f"{STAGING_GROUP}/{level_path.as_posix()}"]
            assert isinstance(staged, zarr.Group)
            for name, source in staged.arrays():
                target = zarr.open_array(
                    output_store, path=f"{level_path.as_posix()}/{name}", mode="r+"
                )
                copies.append((source, target))
        groups.append(path)

    def copy(pair: tuple[zarr.Array, zarr.Array]) -> None:
        source, target = pair
        # one (shard of) chunk of the derived level at a time
        step = (target.shards or target.chunks)[-1]
        n_cells = target.shape[-1]
        for start in range(0, n_cells, step):
            region = slice(start, min(start + step, n_cells))
            target[..., region] = source[..., region]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, copies))

    del root[STAGING_GROUP]

    return groups

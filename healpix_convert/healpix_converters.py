from __future__ import annotations

import asyncio
import importlib.metadata
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import PurePath
from typing import Any, NotRequired, TypedDict, Unpack, cast

import dask.distributed
import healpix_geo
import healpix_resample
import numpy as np
import numpy.typing as npt
import pyproj
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr
from zarr.api.asynchronous import JSON
from zarr.core.sync import sync
from zarr.dtype import parse_dtype

import healpix_convert.core.utils as utils
from healpix_convert.core.conversion_models import (
    ConvertStagingCache,
    InputGroupSpatialInfo,
    InputSpatialInfo,
    OutputSpatialInfo,
)
from healpix_convert.core.healpix_conventions import (
    Healpix,
    WGS84Ellipsoid,
)
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
)
from healpix_convert.resampler_cache import build_resampler
from healpix_convert.settings.common import (
    ConvertSettings,
    ExtraDimensionSettings,
    HealpixDenseChunkSettings,
    HealpixGroupSettings,
    HealpixSparseChunkSettings,
    HealpixUniformChunkSettings,
    MetadataSettings,
    NoChunkSettings,
    broadcast_params,
)

log = structlog.get_logger()


# at most this many output arrays of a chunk are written at the same time
MAX_CONCURRENT_WRITES = 8


async def _write_all(writes: list[tuple[Any, Any, np.ndarray]]) -> None:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WRITES)

    async def write(zarray, key, value):
        async with semaphore:
            await zarray.async_array.setitem(key, value)

    # gather re-raises the first exception of a failed write
    await asyncio.gather(*(write(*w) for w in writes))


def _write_chunk_arrays(writes: list[tuple[Any, Any, np.ndarray]]) -> None:
    """Write the output arrays of one chunk, concurrently.

    Each write encodes one chunk of one array. The writes run together on zarr's
    event loop, and zarr runs the encodings that leave the loop (e.g. ``zstd``,
    or any codec using ``asyncio.to_thread``) in worker threads, so the
    variables of a chunk are encoded in parallel instead of one after the other.

    Parameters
    ----------
    writes : list of (zarr.Array, key, numpy.ndarray)
        Each array, the selection to write and the values.
    """
    # zarr.core.sync.sync is not public API, but it is what zarr's synchronous
    # API runs on (and xarray uses it too): it runs the coroutine on zarr's own
    # event loop thread, also from a Jupyter cell, where a loop already runs
    sync(_write_all(writes))


def _maybe_coerce_str_array_dtype(values: np.ndarray) -> np.ndarray:
    """Coerce an array of strings to numpy's variable-length string dtype.

    Strings read from a Zarr v2 store may come back as a numpy ``object`` array,
    from which zarr 3 refuses to infer a data type
    (zarr-developers/zarr-python#3117); as ``StringDType`` they are written as
    zarr ``string`` (variable-length UTF-8). Other arrays, including object
    arrays that do not hold only strings, are returned unchanged.
    """
    if values.dtype == object and all(isinstance(v, str) for v in values.flat):
        return values.astype(np.dtypes.StringDType())
    return values


class ZarrCreateArrayKwargs(TypedDict):
    shape: tuple[int, ...]
    dtype: npt.DTypeLike | str
    data: NotRequired[np.ndarray]
    chunks: NotRequired[tuple[int, ...] | None]
    dimension_names: NotRequired[Iterable[str]]
    codecs: NotRequired[Iterable[dict[str, JSON]] | None]
    attributes: NotRequired[dict[str, JSON] | None]


def align_extra_dimensions(
    datasets: list[xr.Dataset],
    extra_dimensions: Mapping[str, ExtraDimensionSettings],
    spatial_dims: Iterable[str],
) -> list[xr.Dataset]:
    """Align the input datasets of a group along their non-spatial dimensions.

    Each dimension of ``extra_dimensions`` is aligned on the labels of its
    coordinate, as its settings say (union, intersection, same labels, or fixed
    labels). The inputs must agree on every other non-spatial dimension: same
    size and, where they have a coordinate, same labels. Without that check,
    inputs with different labels of the same size would be concatenated
    position by position, mixing e.g. the values of different detectors.

    """
    if len(datasets) < 2 and not extra_dimensions:
        return datasets

    datasets = list(datasets)

    for dim, dim_settings in extra_dimensions.items():
        present = [ds for ds in datasets if dim in ds.dims]
        if not present:
            continue
        if any(dim not in ds.coords for ds in present):
            raise ValueError(
                f"extra dimension {dim!r} has no coordinate in some input dataset: "
                "its labels are needed to align the inputs"
            )

        datasets = [
            ds.set_xindex(dim) if dim in ds.dims and dim not in ds.xindexes else ds
            for ds in datasets
        ]

        if dim_settings.join == "reindex":
            labels = list(dim_settings.labels or [])
            for ds in datasets:
                if dim not in ds.dims:
                    continue
                unknown = sorted(set(ds[dim].values.tolist()) - set(labels))
                if unknown:
                    raise ValueError(
                        f"extra dimension {dim!r}: input labels {unknown} are not in "
                        f"the labels of the settings {labels}"
                    )
            datasets = [
                ds.reindex({dim: labels}) if dim in ds.dims else ds for ds in datasets
            ]
        else:
            others = set().union(*(ds.dims for ds in datasets)) - {dim}
            datasets = list(xr.align(*datasets, join=dim_settings.join, exclude=others))

    spatial = set(spatial_dims)
    dims = set().union(*(ds.dims for ds in datasets)) - spatial - set(extra_dimensions)

    for dim in sorted(dims, key=str):
        present = [ds for ds in datasets if dim in ds.dims]
        sizes = {ds.sizes[dim] for ds in present}
        differ = len(sizes) > 1
        if not differ and all(dim in ds.coords for ds in present):
            first = present[0][dim].values
            differ = any(
                not np.array_equal(ds[dim].values, first) for ds in present[1:]
            )
        if differ:
            raise ValueError(
                f"the input datasets differ along the non-spatial dimension {dim!r} "
                f"(sizes {sorted(sizes)}): set 'extra_dimensions' in the group "
                "settings to align them (union, intersection or fixed labels)"
            )

    return datasets


class HealpixGroupConverter(ABC):
    """Base (abstract) class for resampling data in a Zarr group to HEALPix."""

    root_spatial_info: InputSpatialInfo
    spatial_info: InputGroupSpatialInfo
    settings: HealpixGroupSettings
    metadata_settings: MetadataSettings
    healpix: Healpix
    group_path: str
    input_paths: list[str]
    datasets: list[xr.Dataset]
    output_store: zarr.StoreLike
    output_path: PurePath
    output_spatial: OutputSpatialInfo
    output_group: zarr.Group
    output_arrays: dict[str, zarr.Array]

    def __init__(
        self,
        path: PurePath,
        *,
        cache: ConvertStagingCache,
        settings: ConvertSettings,
        output_store: str | zarr.StoreLike,
        output_storage_options: dict[str, Any] | None = None,
    ):
        self.root_spatial_info = cache.input_spatial
        self.spatial_info = cache.input_spatial_groups[path]

        group_settings = settings.group_settings[path]
        assert isinstance(group_settings, HealpixGroupSettings)
        self.settings = group_settings
        self.metadata_settings = settings.metadata
        self.healpix = self.settings.healpix

        self.group_path = str(path)
        self.input_paths = [str(p) for p in cache.input_datatrees]
        self.datasets = [
            dt[self.group_path].dataset for dt in cache.input_datatrees.values()
        ]

        # maybe create dataset raster indexes (input datasets with projected CRS)
        if self.spatial_info.is_projected:
            transforms = self.spatial_info.transform
            assert transforms is not None
            spatial_dims = self.spatial_info.spatial_dimensions

            self.datasets = [
                utils.create_raster_index(ds, tr, *spatial_dims)
                for ds, tr in zip(self.datasets, transforms)
            ]

        # the output arrays are created from the first input dataset: align
        # the inputs first, so it has the labels of the output
        self.datasets = align_extra_dimensions(
            self.datasets,
            self.settings.extra_dimensions,
            self.spatial_info.spatial_dimensions,
        )

        if not isinstance(output_store, str):
            root_group = zarr.open_group(
                output_store, mode="a", storage_options=output_storage_options
            )
            output_store = root_group.store

        self.output_store = output_store
        self.output_path = cache.output_groups.io_path_map[path]
        self.output_spatial = cache.output_spatial
        self.output_group = zarr.open_group(
            self.output_store,
            path=str(self.output_path),
            mode="a",
            attributes=get_healpix_group_attrs(self.healpix, self.metadata_settings),
        )

        self.__post_init__()

        self.output_arrays = self._get_arrays()

    def __post_init__(self):
        pass

    @property
    def cell_dim(self) -> str:
        """Returns the name of the HEALPix cell dimension."""
        return self.healpix.spatial_dimension

    @property
    @abstractmethod
    def cell_dim_size(self) -> int:
        """Returns the number of elements along the HEALPix cell dimension.

        Must be implemented in subclasses.

        """
        ...

    @property
    @abstractmethod
    def cell_dim_chunk_size(self) -> int:
        """Returns the fixed chunk size along the HEALPix cell dimension.

        Must be implemented in subclasses.

        """
        ...

    def _get_arrays(self) -> dict[str, zarr.Array]:
        """Get the output zarr arrays, create them if they don't exist yet."""

        zarrays: dict[str, zarr.Array] = {}

        def _get_maybe_create_array(
            name: str, fill_value: Any = None, **kwargs: Unpack[ZarrCreateArrayKwargs]
        ):
            arr = self.output_group.get(name)
            if arr is None:
                path = (
                    f"{self.output_group.path}/{name}"
                    if self.output_group.path
                    else name
                )
                arr = zarr.create(
                    path=path,
                    store=self.output_group.store,
                    fill_value=fill_value,
                    **kwargs,
                )

            assert isinstance(arr, zarr.Array)
            zarrays[name] = arr

        if isinstance(self.healpix.coordinate, str):
            _get_maybe_create_array(
                self.healpix.coordinate,
                shape=(self.cell_dim_size,),
                dtype=np.uint64,
                chunks=(self.cell_dim_chunk_size,),
                dimension_names=(self.cell_dim,),
                fill_value=np.iinfo(np.uint64).max,
                codecs=cast(
                    Iterable[dict[str, JSON]] | None,
                    self.settings.healpix_coordinate_codecs,
                ),
                attributes=get_cf_cell_id_attrs(self.metadata_settings),
            )

        # CF HEALPix grid mapping variable
        grid_mapping_attrs = get_cf_grid_mapping_attrs(
            self.healpix, self.metadata_settings
        )
        if grid_mapping_attrs is not None:
            _get_maybe_create_array(
                CF_GRID_MAPPING_VARIABLE,
                shape=(),
                data=np.array(0, dtype=np.int8),
                dtype=np.int8,
                attributes=grid_mapping_attrs,
            )

        ds0 = self.datasets[0]
        spatial_dims = set(self.spatial_info.spatial_dimensions)

        for name, var in ds0.variables.items():
            if name in self.spatial_info.spatial_coordinates:
                # legacy spatial coordinates are replaced by one HEALPix cell ids coordinate
                continue
            elif name in self.spatial_info.spatial_arrays:
                extra_dims = [d for d in var.dims if d not in spatial_dims]

                if extra_dims:
                    extra_shape = [ds0.sizes[d] for d in extra_dims]
                    var_chunk_sizes = var.encoding.get(
                        "preferred_chunks", {d: ds0.sizes[d] for d in var.dims}
                    )
                    extra_chunk_sizes = [var_chunk_sizes[d] for d in extra_dims]
                    # similarly to how "core" dimensions are handled in numpy
                    # generalized universal functions, the HEALPix cell
                    # dimension is moved to the last axis in the output array.
                    dims = tuple(extra_dims + [self.cell_dim])
                    shape = tuple(extra_shape + [self.cell_dim_size])
                    chunks = tuple(extra_chunk_sizes + [self.cell_dim_chunk_size])
                else:
                    dims = (self.cell_dim,)
                    shape = (self.cell_dim_size,)
                    chunks = (self.cell_dim_chunk_size,)

                if self.settings.output_dtype is not None:
                    dtype = np.dtype(self.settings.output_dtype)
                elif self.settings.resampler.name in [
                    "bilinear",
                    "grid-bilinear",
                    "psf",
                ]:
                    dtype = np.float64
                else:
                    dtype = var.dtype

                # carry the input's own description of the variable: units,
                # standard_name, long_name and the like, minus the attributes
                # the data provider asked to drop. xarray's decoding has
                # already moved scale_factor/add_offset/_FillValue into
                # `encoding`, so what remains in `attrs` is safe to copy.
                attributes = self.settings.metadata.filter_attrs(var.attrs)

                if {"scale_factor", "add_offset"} & set(var.encoding):
                    # CF defines these against the stored (packed) values, and
                    # the output is unpacked, so they would misdescribe it
                    for key in ("valid_min", "valid_max", "valid_range"):
                        attributes.pop(key, None)

                attributes.update(self.settings.metadata.overridden_attrs(str(name)))

                attributes.update(
                    get_cf_data_variable_attrs(self.healpix, self.metadata_settings)
                )

                # cells without data are NaN in a floating output: with zarr's
                # default fill value (0.0) they would be written as zeros, which
                # a lossy serializer codes as values next to the real ones
                if np.dtype(dtype).kind == "f":
                    fill_value = np.nan
                else:
                    fill_value = parse_dtype(
                        np.dtype(dtype), zarr_format=3
                    ).default_scalar()

                _get_maybe_create_array(
                    str(name),
                    shape=shape,
                    dtype=dtype,
                    chunks=chunks,
                    dimension_names=[str(d) for d in dims],
                    codecs=cast(Iterable[dict[str, JSON]], self.settings.codecs),
                    attributes=attributes,
                    fill_value=fill_value,
                )
            else:
                # write array unchanged in output group
                # TODO: use dask.array.to_zarr() if var is chunked? (avoid memory blow-up)
                data = _maybe_coerce_str_array_dtype(var.values)
                _get_maybe_create_array(
                    str(name),
                    shape=var.shape,
                    dtype=data.dtype,
                    data=data,
                    chunks=var.encoding.get("chunks"),
                    dimension_names=[str(d) for d in var.dims],
                    codecs=None,
                )

        return zarrays

    def load_datasets(self, client: dask.distributed.Client | None = None) -> None:
        """Preload all input datasets or persist them in distributed memory when using dask."""
        if client is not None:
            # re-open input datasets with chunks={} -> set chunks aligned with zarr
            reopened_datasets = utils.open_datasets(
                self.input_paths, self.group_path, chunks={}
            )
            reopened_datasets = align_extra_dimensions(
                reopened_datasets,
                self.settings.extra_dimensions,
                self.spatial_info.spatial_dimensions,
            )
            self.datasets = [ds.persist(scheduler=client) for ds in reopened_datasets]
            # block until all data is loaded.
            for ds in self.datasets:
                dask.distributed.wait(ds)
        else:
            self.datasets = [ds.load() for ds in self.datasets]

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset | None:
        """Query input data points to resample as HEALPix cell data.

        Implementation is optional (implement it only if this is relevant).

        Parameters
        ----------
        chunk_cell_id : int, optional
           If provided, this method should return data points for one output chunk
           only (coarse HEALPix cell). If `None`, this method should return all input
           data points.

        Returns
        -------
        xarray.Dataset or None
           An Xarray Dataset that must have `lat` and `lon` 1-dimensional coordinates.
           Or None if no input point is found (for the given chunk).

        """
        raise NotImplementedError

    @abstractmethod
    def convert(self, chunk_index: int | None = None):
        """Convert data to HEALPix and write it in the output zarr arrays.

        Must be implemented in subclasses.

        Parameters
        ----------
        chunk_index : int, optional
            Convert and write only the output chunk given by the index.

        """
        ...


def _warn_if_resampling_lost_data(
    logger, name: str, values: np.ndarray, cell_data: np.ndarray
) -> None:
    """Warn when a variable had valid input points in a chunk but came back from
    the resampler without any finite value (e.g. healpix-resample < 2026.7.0
    returned an all-NaN result whenever one input point was NaN). Such a chunk
    is written as the fill value and zarr does not store it, so the loss would
    otherwise be silent.

    ``values`` and ``cell_data`` have one row per slice of the variable (see
    :func:`_resample_variables`), the points, or the cells, along the last axis.
    """
    lost = np.isfinite(values).any(axis=-1) & ~np.isfinite(cell_data).any(axis=-1)
    if lost.any():
        logger.warning(
            "variable has valid input points in this chunk but no finite resampled value",
            variable=name,
            slices=np.flatnonzero(lost).tolist(),
        )


def _resample_variables(
    resampler,
    ds_input_points: xr.Dataset,
    var_names: list[str],
    resample_params: dict,
    chunk_cell_ids: np.ndarray | None = None,
    logger=log,
) -> dict[str, np.ndarray]:
    """Resample the data variables of a chunk, with one resampler call per group of
    variables that share the same resample parameters.

    All the variables of a chunk are resampled from the same input points, so
    one call can resample several of them: each variable is flattened into an
    array of shape ``(n_slices, n_points)``, one slice per combination of its
    non-spatial dimensions (a single slice when it has none), and the slices of
    the variables of a group are concatenated along the first axis. For the PSF
    resampler, the conjugate gradient then runs once per group instead of once
    per variable.

    Returns, per variable, the resampled values with the non-spatial dimensions
    first and the cell dimension last. With ``chunk_cell_ids`` (dense chunks)
    they are given on those cells, NaN where the resampler returned no value;
    otherwise on the resampler's own cells.

    A variable with valid input points but no finite value after resampling is
    reported on ``logger``.
    """
    slices: dict[str, tuple[tuple[int, ...], np.ndarray]] = {}
    for name in var_names:
        var = ds_input_points.variables[name]
        assert "points" in var.dims
        extra_dims = [d for d in var.dims if d != "points"]
        values = var.transpose(*extra_dims, "points").values.astype("f")
        shape = tuple(var.sizes[d] for d in extra_dims)
        slices[name] = (shape, values.reshape(-1, values.shape[-1]))

    groups: defaultdict[str, list[str]] = defaultdict(list)
    for name in var_names:
        groups[repr(sorted(resample_params[name].items()))].append(name)

    results = {}
    for names in groups.values():
        stacked = np.concatenate([slices[name][1] for name in names], axis=0)
        res = resampler.resample(stacked, **resample_params[names[0]])
        cell_ids = np.asarray(res.cell_ids)
        cell_data = np.asarray(res.cell_data).reshape(stacked.shape[0], -1)

        if chunk_cell_ids is not None:
            # assume nested indexing scheme -> chunk cell ids is a range
            in_chunk = (cell_ids >= chunk_cell_ids[0]) & (
                cell_ids <= chunk_cell_ids[-1]
            )
            positions = (cell_ids[in_chunk] - chunk_cell_ids[0]).astype("int")
            dense = np.full((stacked.shape[0], chunk_cell_ids.size), np.nan)
            dense[:, positions] = cell_data[:, in_chunk]
            cell_data = dense

        start = 0
        for name in names:
            shape, values = slices[name]
            stop = start + values.shape[0]
            _warn_if_resampling_lost_data(logger, name, values, cell_data[start:stop])
            results[name] = cell_data[start:stop].reshape(*shape, cell_data.shape[1])
            start = stop

    return results


_RESAMPLER_NAME_CLS: dict[str, type] = {
    "k-nearest": healpix_resample.KNeighborsResampler,
    "nearest": healpix_resample.NearestResampler,
    "bilinear": healpix_resample.BilinearResampler,
    "psf": healpix_resample.PSFResampler,
    "cell-point": healpix_resample.CellPointResampler,
}
# not in the healpix-resample releases older than its GridBilinearResampler
if hasattr(healpix_resample, "GridBilinearResampler"):
    _RESAMPLER_NAME_CLS["grid-bilinear"] = healpix_resample.GridBilinearResampler

# resamplers built from the 1-D axes of a rectilinear lon/lat grid instead of
# the input points (see `_grid_resampler_axes`)
_GRID_RESAMPLERS = {"grid-bilinear"}

# attribute of the input points of a chunk holding the 1-D axes of each input
# dataset of a rectilinear lon/lat grid, in the order of the points
_GRID_AXES_ATTR = "_healpix_convert_grid_axes"


def _get_resampler_cls(name: str) -> type:
    try:
        return _RESAMPLER_NAME_CLS[name]
    except KeyError:
        raise ValueError(
            f"resampler {name!r} is not available in the installed healpix-resample "
            f"version {importlib.metadata.version('healpix-resample')}"
        ) from None


def _grid_resampler_axes(ds_input_points: xr.Dataset) -> tuple[list, list]:
    """Longitude and latitude axes (one per input dataset) of the input points of
    a chunk, for the resamplers that take the grid axes (``_GRID_RESAMPLERS``)."""
    axes = ds_input_points.attrs.get(_GRID_AXES_ATTR)
    if axes is None:
        raise ValueError(
            "the 'grid-bilinear' resampler needs input data on a rectilinear "
            "longitude/latitude grid (1-D coordinates)"
        )
    # the input datasets are bands of rows, one above the other: two inputs at
    # the same latitudes (e.g. tiles side by side) cannot be interpolated
    # across their seam
    spans = sorted((float(np.min(lat)), float(np.max(lat))) for _, lat in axes)
    north = -np.inf
    for south, top in spans:
        if south <= north:
            raise ValueError(
                "the 'grid-bilinear' resampler needs input datasets that do not "
                "overlap in latitude (one per band of rows), but two inputs of "
                f"this chunk cover the same latitudes ({south} to {min(top, north)}): "
                "mosaic the inputs that lie side by side into one dataset"
            )
        north = max(north, top)
    return [lon for lon, _ in axes], [lat for _, lat in axes]


class UniformChunkConverter(HealpixGroupConverter, ABC):
    """Common class for chunked conversion where chunks have a fixed-size and are aligned with
    HEALPix cells at a given refinement level.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    chunk_healpix: Healpix
    chunk_cell_ids: np.ndarray
    _input_chunk_cell_ids: list[np.ndarray]

    def __post_init__(self):
        assert isinstance(self.settings.chunk, HealpixUniformChunkSettings)
        self.chunk_healpix = self.settings.chunk.healpix

        assert self.healpix.indexing_scheme == "nested"
        assert self.chunk_healpix.indexing_scheme == "nested"

        self._compute_chunk_cell_ids()

        cell_dim = self.healpix.spatial_dimension
        log.info(
            f"fixed chunk size along the {cell_dim!r} dimension: {self.cell_dim_chunk_size}:\n"
            f"  - each chunk represents a HEALPix cell at level {self.chunk_healpix.refinement_level}\n"
            f"  - total: {self.chunk_cell_ids.size} chunk/cells"
        )

        # cache for querying input lat-lon points inside an healpix chunk cell
        self._input_chunk_cell_ids = []

    def _compute_chunk_cell_ids(self):
        assert self.chunk_healpix.refinement_level is not None

        self.chunk_cell_ids, _ = utils.compute_output_chunk_info(
            self.output_spatial.geometry_latlon,
            self.chunk_healpix.refinement_level,
            self.chunk_healpix.ellipsoid.name,
        )

    def _compute_input_chunk_cell_ids(self, level=None):
        if level is None:
            level = self.chunk_healpix.refinement_level

        lon_name, lat_name = self.spatial_info.spatial_coordinates
        cell_ids = []

        for ds in self.datasets:
            cell_ids.append(
                healpix_geo.nested.lonlat_to_healpix(
                    ds[lon_name].values,
                    ds[lat_name].values,
                    level,
                    self.chunk_healpix.ellipsoid.name,
                )
            )

        return cell_ids

    @property
    def input_chunk_cell_ids(self):
        if not len(self._input_chunk_cell_ids):
            self._input_chunk_cell_ids = self._compute_input_chunk_cell_ids()

        return self._input_chunk_cell_ids

    @property
    @abstractmethod
    def chunk_buffer(self) -> int | float: ...

    def _query_chunk_input_points_latlon_curvilinear(
        self, chunk_cell_id: int
    ) -> xr.Dataset | None:
        """Query input lat-lon points on a curvilinear grid.

        Assumes 2-dimensional longitude and latitude coordinates present in the datasets.

        Only input points located within the HEALPix chunk cell are selected. If
        the chunk buffer is non zero, the selection of input points also include
        an outer ring of HEALPix cells around the the chunk at a level determined
        from the buffer width.

        """
        lon_name, lat_name = self.spatial_info.spatial_coordinates
        ds0 = self.datasets[0]
        rdim, cdim = ds0[lon_name].dims

        if self.chunk_buffer > 0:
            ring_level = utils.resolution_to_level(self.chunk_buffer, round_type="ceil")
            input_ring_cell_ids = self._compute_input_chunk_cell_ids(level=ring_level)
        else:
            ring_level = 0
            input_ring_cell_ids = [np.array([], dtype=np.uint64)] * len(self.datasets)

        prepared_datasets = []
        for ds, cids, rids in zip(
            self.datasets, self.input_chunk_cell_ids, input_ring_cell_ids
        ):
            if self.chunk_buffer > 0:
                # TODO: far from optimal (a function in healpix_geo for getting the ring would be better)
                chunk_zoomed = healpix_geo.nested.zoom_to(
                    chunk_cell_id,
                    depth=self.chunk_healpix.refinement_level,
                    new_depth=ring_level,
                ).ravel()

                chunk_with_ring = healpix_geo.nested.kth_neighbourhood(
                    chunk_zoomed, ring_level, ring=1
                )
                chunk_with_ring = np.unique(chunk_with_ring)

                start_id = np.searchsorted(
                    chunk_with_ring, chunk_zoomed[0], side="left"
                )
                stop_id = np.searchsorted(
                    chunk_with_ring, chunk_zoomed[-1], side="right"
                )
                ring_cell_ids = np.delete(chunk_with_ring, slice(start_id, stop_id))

                is_selected = np.logical_or(
                    cids == chunk_cell_id, np.isin(rids, ring_cell_ids)
                )
            else:
                # fastpath
                is_selected = cids == chunk_cell_id

            ridx, cidx = np.nonzero(is_selected)

            if len(ridx) and len(cidx):
                # note: the Xarray advanced indexing operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                # (i.e., either rows or columns)
                ds_clipped = ds.isel(
                    {
                        rdim: xr.Variable("points", ridx),
                        cdim: xr.Variable("points", cidx),
                    }
                )
                prepared_datasets.append(ds_clipped)

        if not len(prepared_datasets):
            # No input data found for the current chunk
            return None

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        ds_input_points["lon"] = ds_input_points[lon_name]
        ds_input_points["lat"] = ds_input_points[lat_name]

        return ds_input_points

    def _query_chunk_input_points_latlon_rectilinear(
        self, chunk_cell_id: int
    ) -> xr.Dataset | None:
        """Select a buffered geographic window without broadcasting the whole grid.

        Longitude axes may use either [-180, 180] or [0, 360], in either
        direction. The window is deliberately conservative: the resampler
        performs the final lookup. Geographic buffer widths are in metres.
        """
        lon_name, lat_name = self.spatial_info.spatial_coordinates
        lon, lat = healpix_geo.nested.vertices(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            ellipsoid=self.chunk_healpix.ellipsoid.name,
        )
        lon, lat = np.asarray(lon).ravel(), np.asarray(lat).ravel()
        # A lower bound on terrestrial curvature radii gives a conservative
        # angular buffer for WGS84 (also suitable for the supported sphere).
        padding = np.rad2deg(max(0.0, self.chunk_buffer) / 6_330_000)
        south = max(-90.0, float(lat.min()) - padding)
        north = min(90.0, float(lat.max()) + padding)
        polar = south <= -90.0 or north >= 90.0
        # Unwrap around the cell centre, not the first vertex (which may lie
        # on the opposite side of the antimeridian).
        centre_lon, _ = healpix_geo.nested.healpix_to_lonlat(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            ellipsoid=self.chunk_healpix.ellipsoid.name,
        )
        centre_lon = float(np.asarray(centre_lon).ravel()[0])
        relative_lon = (lon - centre_lon + 180.0) % 360.0 - 180.0
        lon_padding = (
            180.0
            if polar
            else padding / np.cos(np.deg2rad(max(abs(south), abs(north))))
        )
        west, east = relative_lon.min() - lon_padding, relative_lon.max() + lon_padding

        prepared = []
        axes = []
        grid_resampler = self.settings.resampler.name in _GRID_RESAMPLERS
        for ds in self.datasets:
            x, y = ds[lon_name], ds[lat_name]
            if x.ndim != 1 or y.ndim != 1 or x.dims == y.dims:
                raise ValueError(
                    "rectilinear longitude/latitude must have distinct 1D dimensions"
                )
            # The grid resamplers need the 2 nodes on each side of every cell
            # centre; near the chunk edges they can lie outside the buffered
            # window, so the window grows by one grid step of this input.
            step_x = step_y = 0.0
            if grid_resampler:
                step_x = np.abs((np.diff(x.values) + 180.0) % 360.0 - 180.0).max(
                    initial=0.0
                )
                step_y = np.abs(np.diff(y.values)).max(initial=0.0)
            xvalues = (x.values - centre_lon + 180.0) % 360.0 - 180.0
            xidx = np.flatnonzero(
                np.ones(x.size, dtype=bool)
                if polar
                else (xvalues >= west - step_x) & (xvalues <= east + step_x)
            )
            yidx = np.flatnonzero(
                (y.values >= south - step_y) & (y.values <= north + step_y)
            )
            if not xidx.size or not yidx.size:
                continue
            # Keep the data lazy until after spatial selection. Stack broadcasts
            # coordinates and variables with only one spatial dimension too.
            clipped = ds.isel({x.dims[0]: xidx, y.dims[0]: yidx}).stack(
                points=[y.dims[0], x.dims[0]], create_index=False
            )
            clipped = clipped.assign_coords(
                lon=(clipped[lon_name] + 180.0) % 360.0 - 180.0,
                lat=clipped[lat_name],
            )
            prepared.append(clipped)
            # the points are stacked in C order of (lat, lon): the grid
            # resamplers take these axes and values in that order
            axes.append(
                (
                    (x.values[xidx] + 180.0) % 360.0 - 180.0,
                    np.asarray(y.values[yidx]),
                )
            )
        if not prepared:
            return None
        ds_input_points = xr.concat(prepared, dim="points")
        ds_input_points.attrs[_GRID_AXES_ATTR] = axes
        return ds_input_points

    def _query_chunk_input_points_projected(
        self, chunk_cell_id: int
    ) -> xr.Dataset | None:
        """
        This function works with projected points and
        implements the following procedure:
        - extract output HEALPix cell ids enveloppe as a polygon
        - prepare each input dataset:
          - re-project the enveloppe polygon in the dataset's projected CRS
          - create a fixed-width (meters) buffer around the polygon
          - get overlapping zone between the buffer polygon and the input Zarr dataset coverage
            in projected CRS coordinates
          - maybe exclude input dataset with overlapping zone is empty
          - slice input dataset (in projected coordinates) using the bounding box of the
            overlapping zone.
          - select the data points inside the overlapping zone, exclude input dataset
            if none is found (e.g., overlapping zone is a sliver with no pixel center).
        - flatten (stack) input datasets and convert x/y coordinates to lat-lon.

        """
        crs_wgs84 = pyproj.CRS.from_epsg(4326)

        # compute chunk cell polygon
        lon, lat = healpix_geo.nested.vertices(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            ellipsoid=self.chunk_healpix.ellipsoid.name,
        )
        chunk_poly_latlon = shapely.Polygon(list(zip(lon[0], lat[0])))

        chunk_polys = utils.reproject_to_multiple_crs(
            chunk_poly_latlon, crs_wgs84, self.spatial_info.crs
        )

        # Maybe add buffer around polygon
        if self.chunk_buffer > 0.0:
            chunk_polys = [
                shapely.buffer(poly, self.chunk_buffer) for poly in chunk_polys
            ]

        # compute intersections with input dataset spatial extents
        input_extents = self.root_spatial_info.get_extents()[0]
        overlap_polys = [
            shapely.intersection(chunk_poly, input_extent)
            for chunk_poly, input_extent in zip(chunk_polys, input_extents)
        ]

        # prepare input datasets:
        # - filter and crop input datasets based on overlapping coverage
        #   between (buffered) chunk cell and dataset extents
        # - convert to lat/lon (wgs84) if needed
        # - flatten spatial dimensions
        prepared_datasets = []
        transforms = self.spatial_info.transform or [None] * len(self.datasets)
        for ds, crs, poly, transform in zip(
            self.datasets, self.spatial_info.crs, overlap_polys, transforms
        ):
            if poly.is_empty:
                continue

            xmin, ymin, xmax, ymax = shapely.bounds(poly)

            # slice bounds follow the direction of the x/y coordinates
            # (y is usually decreasing, i.e., north-up rasters)
            xslice = slice(xmin, xmax)
            yslice = slice(ymax, ymin)
            if transform is not None:
                if transform.a < 0:
                    xslice = slice(xmax, xmin)
                if transform.e > 0:
                    yslice = slice(ymin, ymax)

            ds_clipped = (
                ds.sel(x=xslice, y=yslice)
                .compute()
                .pipe(lambda ds: utils.assign_transform_coords(ds, crs, crs_wgs84))
                # note: the Xarray stack operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                # (i.e., either x or y)
                .stack(
                    points=list(self.spatial_info.spatial_dimensions),
                    create_index=False,
                )
            )

            x = ds_clipped.x.values
            y = ds_clipped.y.values
            # same as shapely.within(shapely.points(x, y), poly) (a point on
            # the boundary is not inside), without creating the point objects
            in_poly = shapely.contains_xy(poly, x, y)

            if not in_poly.any():
                # no pixel centre in the overlapping zone, e.g., the (buffered)
                # chunk polygon only grazes the dataset extent (pixel edges):
                # no input data from this dataset
                continue

            ds_clipped = ds_clipped.isel(points=in_poly)

            prepared_datasets.append(ds_clipped)

        if not len(prepared_datasets):
            # No input data found for the current chunk
            return None

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        return ds_input_points

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset | None:
        # this class supports chunked conversion only
        assert chunk_cell_id is not None

        if self.spatial_info.is_projected:
            return self._query_chunk_input_points_projected(chunk_cell_id)
        elif not self.spatial_info.is_rectilinear:
            return self._query_chunk_input_points_latlon_curvilinear(chunk_cell_id)
        else:
            return self._query_chunk_input_points_latlon_rectilinear(chunk_cell_id)


class DenseChunkConverter(UniformChunkConverter):
    """Chunked conversion of input data to HEALPix, where output chunks correspond
    to "coarse" HEALPix cells densely populated by finer HEALPIX cells at a same level
    (continuous cell id range).

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    @property
    def cell_dim_size(self) -> int:
        return self.chunk_cell_ids.size * self.cell_dim_chunk_size

    @property
    def cell_dim_chunk_size(self) -> int:
        level = self.healpix.refinement_level
        chunk_level = self.chunk_healpix.refinement_level
        assert level is not None and chunk_level is not None
        return 4 ** (level - chunk_level)

    @property
    def chunk_buffer(self):
        assert isinstance(self.settings.chunk, HealpixDenseChunkSettings)
        return self.settings.chunk.chunk_buffer_width

    def convert(self, chunk_index: int | None = None):
        # this class supports chunked conversion only
        assert chunk_index is not None

        n_chunks = self.chunk_cell_ids.size

        chunk_cell_id = self.chunk_cell_ids[chunk_index]

        log.debug(
            f"processing chunk {chunk_index + 1}/{n_chunks} (cell id {chunk_cell_id})"
        )

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_cell_id,
            self.chunk_healpix.refinement_level,
            self.healpix.refinement_level,
        )[0]

        start = chunk_index * cell_ids.size
        stop = start + cell_ids.size
        chunk_slice = slice(start, stop)
        self.output_arrays[self.healpix.coordinate][chunk_slice] = cell_ids

        log.debug("query input points")
        ds_input_points = self.query_input_points(chunk_cell_id)

        # no input data point for current chunk
        if ds_input_points is None:
            return

        resampler_settings = self.settings.resampler

        # For the "nearest" resample method: give output cell ids the resampler (forcing)
        # For the grid resamplers too: they interpolate at the given cell centres
        # For the other methods: best to let the resampler algorithm do the lookup
        if (
            resampler_settings.name == "nearest"
            or resampler_settings.name in _GRID_RESAMPLERS
        ):
            out_cell_ids = cell_ids
        else:
            out_cell_ids = None

        resampler_cls = _get_resampler_cls(resampler_settings.name)
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        if resampler_settings.name in _GRID_RESAMPLERS:
            lon_deg, lat_deg = _grid_resampler_axes(ds_input_points)
        else:
            lon_deg = ds_input_points.lon.values
            lat_deg = ds_input_points.lat.values

        log.debug("initialize resampler")
        resampler = build_resampler(
            resampler_cls,
            resampler_settings.name,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            level=self.healpix.refinement_level,
            verbose=False,
            out_cell_ids=out_cell_ids,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.healpix.ellipsoid.name,
            **dict(resampler_settings.init_params),
        )
        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        resampled = _resample_variables(
            resampler,
            ds_input_points,
            var_names,
            resample_params,
            chunk_cell_ids=cell_ids,
            logger=log.bind(group=self.group_path, chunk_index=chunk_index),
        )

        writes = []
        for name, cell_data in resampled.items():
            zarray = self.output_arrays[name]
            # cells without data (TODO: get the right fill_value and dtype)
            cell_data[np.isnan(cell_data)] = zarray.fill_value
            writes.append((zarray, (..., chunk_slice), cell_data.astype(zarray.dtype)))

        _write_chunk_arrays(writes)


class SparseChunkConverter(UniformChunkConverter):
    """Chunked conversion of input data to HEALPix, where output chunks correspond
    to "coarse" HEALPix cells sparsely populated by finer HEALPIX cells at a same level.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    @property
    def cell_dim_size(self) -> int:
        return self.chunk_cell_ids.size * self.cell_dim_chunk_size

    @property
    def cell_dim_chunk_size(self) -> int:
        assert isinstance(self.settings.chunk, HealpixSparseChunkSettings)
        return self.settings.chunk.chunk_size

    @property
    def chunk_buffer(self):
        return 0

    def convert(self, chunk_index: int | None = None):
        # this class supports chunked conversion only
        assert chunk_index is not None

        n_chunks = self.chunk_cell_ids.size

        chunk_cell_id = self.chunk_cell_ids[chunk_index]

        log.debug(
            f"processing chunk {chunk_index + 1}/{n_chunks} (cell id {chunk_cell_id})"
        )

        log.debug("query input points")
        ds_input_points = self.query_input_points(chunk_cell_id)

        # no input data point for current chunk
        if ds_input_points is None:
            return

        log.debug("initialize resampler")
        resampler_settings = self.settings.resampler

        resampler_cls = _get_resampler_cls(resampler_settings.name)
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        resampler = build_resampler(
            resampler_cls,
            resampler_settings.name,
            lon_deg=ds_input_points.lon.values,
            lat_deg=ds_input_points.lat.values,
            # level=self.healpix.refinement_level,  # assumes cell-point (level=29)
            verbose=False,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.healpix.ellipsoid.name,
            **dict(resampler_settings.init_params),
        )

        cell_ids = resampler.get_cell_ids()
        start = chunk_index * self.cell_dim_chunk_size
        stop = start + cell_ids.size
        chunk_slice = slice(start, stop)

        self.output_arrays[self.healpix.coordinate][chunk_slice] = cell_ids

        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        resampled = _resample_variables(
            resampler,
            ds_input_points,
            var_names,
            resample_params,
            logger=log.bind(group=self.group_path, chunk_index=chunk_index),
        )

        writes = []
        for name, cell_data in resampled.items():
            zarray = self.output_arrays[name]
            writes.append((zarray, (..., chunk_slice), cell_data.astype(zarray.dtype)))

        _write_chunk_arrays(writes)

        del resampler


class NoChunkConverter(HealpixGroupConverter):
    """Conversion of input data to HEALPix, where the output consists in
    one single chunk of HEALPix cells that are  fully covering the extent
    (polygon) of the output Zarr dataset at a fixed refinement level given
    by the group conversion settings.

    This class works only with the "nested" HEALPIX cell indexing scheme.

    """

    cell_ids: np.ndarray
    ellipsoid: str

    def __post_init__(self):
        assert isinstance(self.settings.chunk, NoChunkSettings)
        assert self.healpix.indexing_scheme == "nested"

        if isinstance(self.healpix.ellipsoid, WGS84Ellipsoid):
            self.ellipsoid = self.healpix.ellipsoid.name
        else:
            # TODO: is sphere really used here? If yes, need a fix below
            # in `query_input_points` where input x/y coordinates are converted
            # to lat/lon on the WGS84 datum.
            self.ellipsoid = "sphere"

        self.cell_ids, _, _ = healpix_geo.nested.polygon_coverage(
            shapely.coordinates.get_coordinates(self.output_spatial.geometry_latlon),
            self.healpix.refinement_level,
            ellipsoid=self.ellipsoid,
            flat=True,
        )

        cell_dim = self.healpix.spatial_dimension
        log.info(f"one single chunk along the {cell_dim!r} dimension.")

    @property
    def cell_dim_size(self) -> int:
        return self.cell_ids.size

    @property
    def cell_dim_chunk_size(self) -> int:
        return self.cell_dim_size

    def query_input_points(self, chunk_cell_id: int | None) -> xr.Dataset:
        # this class does not support conversion chunk by chunk
        assert chunk_cell_id is None

        crs_wgs84 = pyproj.CRS.from_epsg(4326)

        def maybe_convert_to_latlon(ds: xr.Dataset, crs: pyproj.CRS):
            if self.spatial_info.is_projected:
                return utils.assign_transform_coords(ds, crs, crs_wgs84)
            else:
                lon_name, lat_name = self.spatial_info.spatial_coordinates
                return ds.assign_coords(lon=ds[lon_name], lat=ds[lat_name])

        prepared_datasets = []
        for ds, crs in zip(self.datasets, self.spatial_info.crs):
            ds_latlon_flat = (
                ds.compute()
                .pipe(lambda ds: maybe_convert_to_latlon(ds, crs))
                # note: the Xarray stack operation below already broadcasts
                # the Dataset variables that only have a subset of the spatial dimensions
                .stack(
                    points=list(self.spatial_info.spatial_dimensions),
                    create_index=False,
                )
            )

            prepared_datasets.append(ds_latlon_flat)

        ds_input_points = xr.concat(prepared_datasets, dim="points")
        return ds_input_points

    def convert(self, chunk_index: int | None = None):
        # this class does not support conversion chunk by chunk
        assert chunk_index is None

        cell_ids = self.cell_ids

        self.output_arrays[self.healpix.coordinate][:] = cell_ids

        log.debug("query input points")
        ds_input_points = self.query_input_points(None)

        resampler_settings = self.settings.resampler

        # For the "nearest" resample method: give output cell ids the resampler (forcing)
        # For the other methods: best to let the resampler algorithm do the lookup
        if resampler_settings.name == "nearest":
            out_cell_ids = self.cell_ids
        else:
            out_cell_ids = None

        resampler_cls = _get_resampler_cls(resampler_settings.name)
        # TODO: pass array dtype to resampler
        # in case all arrays to resample have the same dtype

        log.debug("initialize resampler")
        resampler = build_resampler(
            resampler_cls,
            resampler_settings.name,
            lon_deg=ds_input_points.lon.values,
            lat_deg=ds_input_points.lat.values,
            level=self.healpix.refinement_level,
            verbose=False,
            out_cell_ids=out_cell_ids,
            nest=self.healpix.indexing_scheme == "nested",
            ellipsoid=self.ellipsoid,
            **dict(resampler_settings.init_params),
        )
        # resample arrays (data variables)
        var_names = [
            str(name)
            for name in ds_input_points.variables
            if name in self.spatial_info.spatial_arrays
        ]
        resample_params = broadcast_params(
            resampler_settings.resample_params, var_names
        )

        log.debug(f"resample {len(var_names)} arrays")
        for name in var_names:
            var = ds_input_points.variables[name]

            assert "points" in var.dims
            extra_dims = [d for d in var.dims if d != "points"]
            extra_dim_sizes = [var.sizes[d] for d in extra_dims]

            # make sure the "points" core dimension is moved to the last axis
            # maybe stack non-spatial dimensions, to pass to `.resample()` as a single "batch" dimension
            if extra_dims:
                if len(extra_dims) == 1:
                    var = var.transpose(extra_dims[0], "points")
                else:
                    var = var.stack(batch=extra_dims).transpose("batch", "points")

                cell_data_shape = (var.shape[0], cell_ids.size)
            else:
                cell_data_shape = (cell_ids.size,)

            # initialize cell data to nodata (TODO: get the right fill_value and dtype)
            # index/fill the array below with data values returned by the resampler
            cell_data = np.full(cell_data_shape, np.nan)

            data = var.values.astype("f")

            res = resampler.resample(data, **resample_params[name])

            zarray = self.output_arrays[str(name)]

            _, res_cell_data_idx, cell_data_idx = np.intersect1d(
                res.cell_ids, cell_ids, assume_unique=True, return_indices=True
            )

            cell_data[..., cell_data_idx] = res.cell_data[..., res_cell_data_idx]
            cell_data[..., np.isnan(cell_data)] = zarray.fill_value

            # maybe unstack non-spatial dimensions from the resampled batch dimension
            if len(extra_dims) > 1:
                cell_data = cell_data.reshape(tuple(extra_dim_sizes + [cell_ids.size]))

            zarray[..., :] = cell_data.astype(zarray.dtype)


_CONVERTER_NAME_CLS: dict[str, type[HealpixGroupConverter]] = {
    "no_chunk": NoChunkConverter,
    "healpix_cell_dense": DenseChunkConverter,
    "healpix_cell_sparse": SparseChunkConverter,
}


def init_converter(
    path: PurePath,
    *,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_store: str | zarr.StoreLike,
    output_storage_options: dict[str, Any] | None = None,
) -> HealpixGroupConverter:
    group_settings = settings.group_settings[path]
    cls = _CONVERTER_NAME_CLS[group_settings.chunk.method]

    return cls(
        path,
        cache=cache,
        settings=settings,
        output_store=output_store,
        output_storage_options=output_storage_options,
    )

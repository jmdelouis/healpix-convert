"""
Aggregating HEALPix data onto the cells of a coarser refinement level.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray as xr

from healpix_convert.core.healpix_conventions import Healpix

#: Supported methods for aggregating the values of the children cells. These
#: are the reductions that Xarray's coarsen applies to a block of cells and
#: that give, for a whole block, the same result as aggregating all its cells
#: at once (unlike, e.g., the median). ``"mode"`` (the most frequent value) is
#: for categorical variables: masks, classifications. ``"bitwise_or"`` is for
#: bit flags (integer variables where each bit is a flag of its own): a flag is
#: set in a coarser cell if it is set in any of its cells.
ReduceMethod: TypeAlias = Literal["mean", "min", "max", "sum", "mode", "bitwise_or"]

#: Encoding of an input variable reused for the aggregated ones: how the values
#: are stored (data type, fill value, CF packing), not how the chunks are
#: encoded.
VALUE_ENCODING = ("dtype", "fill_value", "_FillValue", "scale_factor", "add_offset")


def select_source_levels(
    variables: Mapping[int, Collection[str]],
    level: int,
    *,
    coarsen_from: str = "nearest",
) -> dict[str, int]:
    """Converted refinement level each variable of a derived level is
    aggregated from (variable name -> converted level).

    ``variables`` gives the variables of every level converted from the input
    data (converted level -> variable names).

    A variable is absent from the levels finer than its finest converted level
    (it is never upsampled). A variable converted at several levels, e.g., the
    copies of the finest bands that the provider of a multi-resolution product
    resampled itself, is aggregated from the nearest converted level at or
    finer than ``level`` (``"nearest"``) or from its finest converted level
    (``"finest"``). The variables of a derived level may therefore be
    aggregated from different converted levels.

    """
    finer = {lvl: set(names) for lvl, names in variables.items() if lvl >= level}
    names = {name for level_names in finer.values() for name in level_names}

    sources: dict[str, int] = {}

    for name in sorted(names):
        converted = [lvl for lvl in sorted(finer) if name in finer[lvl]]
        sources[name] = min(converted) if coarsen_from == "nearest" else max(converted)

    return sources


def nanmode(values: np.ndarray, axis: int | tuple[int, ...] = -1) -> np.ndarray:
    """Most frequent value along ``axis``, missing (NaN) values skipped.

    A tie goes to the smallest of the most frequent values, so that the result
    does not depend on the order of the cells. All values missing gives NaN.

    """
    if isinstance(axis, tuple):
        (axis,) = axis
    values = np.moveaxis(np.asarray(values), axis, -1)
    if values.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan)

    ordered = np.sort(values, axis=-1)  # NaN last
    n = ordered.shape[-1]
    position = np.broadcast_to(np.arange(n), ordered.shape)
    run_start = np.ones(ordered.shape, dtype=bool)
    run_start[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    # within a run of equal values, how many of them up to this position
    count = (
        position - np.maximum.accumulate(np.where(run_start, position, 0), axis=-1) + 1
    )
    if np.issubdtype(ordered.dtype, np.floating):
        count = np.where(np.isnan(ordered), 0, count)

    # the first maximum is the end of the earliest longest run: the smallest
    # of the most frequent values
    best = np.argmax(count, axis=-1)
    mode = np.take_along_axis(ordered, best[..., None], axis=-1)[..., 0]

    if np.issubdtype(ordered.dtype, np.floating):
        mode = np.where(count.max(axis=-1) == 0, np.nan, mode)

    return mode


def _blocks_mode(var: xr.DataArray, dim: str, window: int) -> xr.DataArray:
    """Most frequent value of each block of ``window`` consecutive cells,
    computed blockwise (lazily on Dask arrays), like Xarray's own reductions.
    """
    var = var.reset_coords(drop=True)
    windows = var.coarsen({dim: window}).construct(
        {dim: (dim, "window")}, keep_attrs=True
    )
    if windows.chunks is not None:
        windows = windows.chunk({"window": -1})

    return xr.apply_ufunc(
        nanmode,
        windows,
        input_core_dims=[["window"]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[var.dtype],
        keep_attrs=True,
    )


def _flags_dtype(var: xr.DataArray, name: Hashable) -> np.dtype:
    """Integer data type of the bit flags of ``var``: its own, or the stored
    one when its fill value has been decoded as NaN.
    """
    dtype = np.dtype(var.encoding.get("dtype", var.dtype))
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(
            f"the 'bitwise_or' reduction needs integer bit flags, but the variable "
            f"{name!r} has the data type {dtype}"
        )
    if np.issubdtype(var.dtype, np.floating) and dtype.itemsize > 4:
        # a 64-bit integer decoded as float64 has lost its high bits already
        raise ValueError(
            f"the 'bitwise_or' reduction does not support the variable {name!r}: "
            f"its {dtype} values are decoded as floating point values (it has a "
            "fill value), which do not hold all their bits"
        )

    return dtype


def nanbitwise_or(
    values: np.ndarray, axis: int | tuple[int, ...] = -1, *, dtype=None
) -> np.ndarray:
    """Bitwise OR along ``axis``, missing (NaN) values skipped.

    ``values`` are integers, or the floating point values of integers with
    missing values (``dtype`` gives their integer data type). A bit is set in
    the result if it is set in any of the values. All values missing gives NaN.

    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        return np.bitwise_or.reduce(values, axis=axis)

    missing = np.isnan(values)
    flags = np.where(missing, 0, values).astype(dtype)
    result = np.bitwise_or.reduce(flags, axis=axis).astype(values.dtype)

    return np.where(missing.all(axis=axis), np.nan, result)


def _blocks_bitwise_or(
    var: xr.DataArray, dim: str, window: int, dtype: np.dtype
) -> xr.DataArray:
    """Bitwise OR of each block of ``window`` consecutive cells, computed
    blockwise (lazily on Dask arrays), like Xarray's own reductions.
    """
    var = var.reset_coords(drop=True)
    windows = var.coarsen({dim: window}).construct(
        {dim: (dim, "window")}, keep_attrs=True
    )
    if windows.chunks is not None:
        windows = windows.chunk({"window": -1})

    return xr.apply_ufunc(
        nanbitwise_or,
        windows,
        input_core_dims=[["window"]],
        kwargs={"axis": -1, "dtype": dtype},
        dask="parallelized",
        output_dtypes=[var.dtype],
        keep_attrs=True,
    )


def coarsen_healpix_group(
    ds: xr.Dataset,
    level: int,
    *,
    healpix: Healpix,
    reduce: ReduceMethod = "mean",
) -> xr.Dataset:
    """Aggregate the data variables of a HEALPix group onto the cells of a
    coarser refinement level.

    The cells of ``ds`` must densely cover whole HEALPix cells of ``level``, in
    ascending (nested) order, as the ``healpix_cell_dense`` chunks of a
    converted group do: the cells of a coarser cell are then ``4 ** delta``
    consecutive cells, so they are aggregated by position. Nothing is grouped
    and the cell ids are not read: the aggregation is blockwise, hence lazy and
    bounded in memory.

    Each cell of the output aggregates all the cells of the input it contains
    (and no other), i.e., the mean is a block mean and not a mean of the means
    of the intermediate levels.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        One (single-scale) HEALPix group of a converted dataset.
    level : int
        HEALPix refinement level of the output (coarser than the level of
        ``ds``).
    healpix : object
        HEALPix grid of ``ds`` (a ``Healpix`` object): its refinement level,
        cell dimension and cell ids coordinate.
    reduce : str, optional
        How the values of the children cells are aggregated (default: "mean");
        "mode" gives their most frequent value (see :py:func:`nanmode`),
        "bitwise_or" the bits set in any of them, for integer bit flags (see
        :py:func:`nanbitwise_or`). Missing (NaN) values are skipped.

    Returns
    -------
    :py:class:`xarray.Dataset`
        The group at the coarser refinement level, with its ``dggs`` attribute
        updated and the data type of its variables preserved.

    """
    assert healpix.indexing_scheme == "nested"
    assert healpix.refinement_level is not None
    assert isinstance(healpix.coordinate, str)

    delta = healpix.refinement_level - level
    assert delta > 0

    dim = healpix.spatial_dimension
    coordinate = healpix.coordinate
    window = 4**delta
    # the cells of the group cover whole HEALPix cells of the coarser level
    # (checked against the spatial extent of the output before any level is built)
    assert ds.sizes[dim] % window == 0

    cell_vars = [name for name, var in ds.data_vars.items() if dim in var.dims]
    # one block of cells per cell of the coarser level
    blocks = ds[cell_vars].coarsen({dim: window}, coord_func={coordinate: "min"})
    if reduce == "mode":
        # the structure and the coordinates of the blocks (their values are
        # replaced below, so this reduction is never computed)
        coarsened = blocks.min(keep_attrs=True)
        for name in cell_vars:
            coarsened[name] = _blocks_mode(ds[name], dim, window)
    elif reduce == "bitwise_or":
        dtypes = {name: _flags_dtype(ds[name], name) for name in cell_vars}
        coarsened = blocks.min(keep_attrs=True)
        for name in cell_vars:
            coarsened[name] = _blocks_bitwise_or(ds[name], dim, window, dtypes[name])
    else:
        reduce_blocks = getattr(blocks, reduce)
        coarsened = reduce_blocks(keep_attrs=True)

    # the smallest cell id of a block is the first child of the coarser cell
    coarsened = coarsened.assign_coords(
        {coordinate: coarsened[coordinate] >> np.uint64(2 * delta)}
    )

    # add back the variables that are not defined on the cells (e.g., the CF
    # grid mapping) and the attributes, with the new refinement level
    coarsened = coarsened.merge(ds.drop_dims(dim), combine_attrs="override")
    coarsened.attrs = dict(ds.attrs) | {
        "dggs": healpix.model_copy(update={"refinement_level": level}).model_dump()
    }

    return encode_like(coarsened, ds.variables)


def codecs_encoding(codecs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Xarray's Zarr encoding of a codec pipeline given as codec settings.

    ``to_zarr`` takes a pipeline as ``filters`` (array to array codecs), one
    ``serializer`` (array to bytes) and ``compressors`` (bytes to bytes), not
    as one list: each codec is created from Zarr's codec registry and goes to
    its part.

    """
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
    from zarr.registry import get_codec_class

    encoding: dict[str, Any] = {"filters": [], "compressors": []}

    for codec_settings in codecs:
        cls = get_codec_class(codec_settings["name"])
        # an empty configuration gives the codec's defaults (e.g. the byte
        # order of the "bytes" serializer)
        codec = (
            cls.from_dict(dict(codec_settings))
            if codec_settings.get("configuration")
            else cls()
        )
        if isinstance(codec, ArrayArrayCodec):
            encoding["filters"].append(codec)
        elif isinstance(codec, ArrayBytesCodec):
            encoding["serializer"] = codec
        elif isinstance(codec, BytesBytesCodec):
            encoding["compressors"].append(codec)
        else:
            raise TypeError(f"unknown kind of zarr codec: {codec!r}")

    return encoding


def encode_like(ds: xr.Dataset, templates: Mapping[str, xr.Variable]) -> xr.Dataset:
    """Give the variables of ``ds`` the data type and the fill value of the
    matching template (input) variables.

    Aggregating cells returns floating point values: an integer variable is
    rounded (instead of being truncated) and gets its fill value back where it
    has no value.

    The codecs of the input are not reused: the derived levels have their own
    cells and chunks, and the codecs of a lossy serializer would code the
    aggregated values a second time. They are set by the ``codecs`` pyramid
    setting (see :py:func:`codecs_encoding`).

    """
    ds = ds.copy()

    for name, template in templates.items():
        if name not in ds.variables:
            continue

        encoding = {k: v for k, v in template.encoding.items() if k in VALUE_ENCODING}
        # xarray would otherwise declare NaN as the missing value of the
        # floating point variables, which the converted dataset does not do
        encoding.setdefault("_FillValue", None)

        if ds[name].dtype != template.dtype:
            values = ds[name]
            if np.issubdtype(template.dtype, np.integer):
                fill_value = encoding.get("fill_value")
                values = values.fillna(0 if fill_value is None else fill_value).round()
            ds[name] = values.astype(template.dtype)

        ds[name].encoding = encoding

    return ds

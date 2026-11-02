"""Utility methods used internally for converting datasets onto HEALPix."""

import errno
import random
import re
import time
from collections import Counter
from collections.abc import Iterable
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

import affine
import healpix_geo
import numpy as np
import pyproj
import rasterix
import shapely
import xarray as xr


def check_patterns(patterns: Iterable[str]) -> None:
    """Raise a ValueError if any of the given regular expressions is invalid."""
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as err:
            raise ValueError(f"invalid regular expression {pattern!r}: {err}") from None


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    """Returns True if the given name matches any of the given regular
    expressions (matched against the whole name).
    """
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def open_datatrees(
    paths: list[Any],
    chunks: dict | None = None,
    storage_options: dict[str, Any] | None = None,
    consolidated: bool | None = None,
) -> list[xr.DataTree]:
    """Open Zarr datasets, given as paths or stores, as "raw" xarray.DataTree
    objects.

    Fast-version (disable most decoding and index creation).

    ``consolidated`` is passed to Zarr: by default the consolidated metadata of
    the dataset is read if it has any. Pass False for a dataset that is being
    written, whose consolidated metadata (if any) is not up to date yet.

    """
    return [
        xr.open_datatree(
            path,
            engine="zarr",
            chunks=chunks,
            storage_options=storage_options,
            consolidated=consolidated,
            decode_cf=True,
            decode_times=False,
            decode_timedelta=False,
            create_default_indexes=False,
        )
        for path in paths
    ]


def open_datasets(
    paths: list[str], group: str, chunks: dict | None = None
) -> list[xr.Dataset]:
    """Open a given group in input Zarr datasets as "raw" xarray.Dataset objects.

    Fast-version (disable most decoding and index creation).
    """
    return [
        xr.open_zarr(
            path,
            group=group,
            chunks=chunks,
            decode_cf=True,
            decode_times=False,
            decode_timedelta=False,
            create_default_indexes=False,
        )
        for path in paths
    ]


def _get_stac_proj_crs(attrs: dict[str, Any]) -> pyproj.CRS | None:
    crss = []
    if "proj:epsg" in attrs:
        # STAC projection extension version < 2.0
        crss.append(pyproj.CRS.from_epsg(attrs["proj:epsg"]))
    if "proj:code" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:code"]))
    if "proj:wkt2" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:wkt2"]))
    if "proj:projjson" in attrs:
        crss.append(pyproj.CRS.from_user_input(attrs["proj:projjson"]))

    if len(crss):
        if len(crss) > 1 and not all(crs == crss[0] for crs in crss[1:]):
            raise ValueError("found conficting CRS in metadata")
        return crss[0]
    else:
        return None


def get_wgs84_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    geom_dict = xr_obj.attrs["stac_discovery"]["geometry"]
    return shapely.geometry.shape(geom_dict)


def get_proj_polygon_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> shapely.Polygon:
    bbox = xr_obj.attrs["stac_discovery"]["properties"].get("proj:bbox")
    if bbox is not None:
        return shapely.geometry.box(*bbox)
    else:
        return shapely.Polygon()


def get_crs_from_stac(xr_obj: xr.DataTree | xr.Dataset) -> pyproj.CRS:
    crs = _get_stac_proj_crs(xr_obj.attrs["stac_discovery"]["properties"])
    if crs is None:
        # Return WGS84 as default if no CRS is found
        # TODO: what should be the default CRS? Undefined (sphere)?
        crs = pyproj.CRS.from_epsg(4326)
    return crs


def extract_spatial_info_stac(ds: xr.Dataset, default_crs: pyproj.CRS) -> dict | None:
    """Extract spatial information from a Zarr group using STAC conventions.

    Assume that spatial information is represented as STAC (projection)
    attributes in data variables (EOPF Zarr groups).

    Assume x/y spatial dimensions.

    Returns None if no spatial information is detected.

    """
    # TODO: check consistent STAC attrs in each variables?
    # (assume they are consistent for now)
    var0 = next(iter(ds.data_vars.values()))

    crs = _get_stac_proj_crs(var0.attrs)
    spatial_dims = {"x", "y"}

    if crs is None:
        # TODO: remove the if condition below
        # The case of x/y coordinates with no CRS nor affine transform metadata in
        # a Zarr group shouldn't be covered, but it is here to support some groups
        # in UTM EOPF datasets published on EOPF sample service where metadata is missing.
        # In that case, the `default_crs` (extracted
        # from the root group metadata) is passed here and the transform is computed from the
        # x/y coordinates (rasterix assumes monotonic coordinate values)
        if "x" not in ds.variables and "y" not in ds.variables:
            return None
        elif any(name in ds.variables and ds[name].ndim != 1 for name in ("x", "y")):
            # 2-D x/y (e.g., curvilinear grid) skip here, best handled by corresponding
            # lat-lon coordinates (CF decoding) if present 
            return None
        else:
            crs = default_crs
            ds_ = rasterix.assign_index(ds, x_dim="x", y_dim="y", crs=False)
            # important: assume here that x/y coordinates are pixel centers!
            transform = cast(rasterix.RasterIndex, ds_.xindexes["x"]).transform()
    else:
        transform = affine.Affine(*var0.attrs["proj:transform"])

    spatial_arrays = [
        name
        for name, var in ds.variables.items()
        if spatial_dims.intersection(var.dims) and name not in spatial_dims
    ]

    return {
        "crs": [crs],
        "transform": [transform],
        "is_rectilinear": transform.is_rectilinear,
        "spatial_dimensions": {"x": ds.sizes["x"], "y": ds.sizes["y"]},
        "spatial_coordinates": ["x", "y"],
        "spatial_attrs": [],
        "spatial_arrays": spatial_arrays,
        "spatial_var_attrs": [name for name in var0.attrs if name.startswith("proj:")],
    }


def extract_spatial_info_cf(ds: xr.Dataset) -> dict | None:
    """Extract spatial information from a Zarr group using CF conventions.

    Only longitude-latitude is supported for now (WGS84 is assumed). The order
    (longitude, latitude) is used here.

    Returns None if no spatial information is detected.

    """
    std_lonlat = ("longitude", "latitude")
    lonlat_coords = {}
    lonlat_dims = set()
    is_rectilinear = True

    for name, var in ds.coords.items():
        std_name = var.attrs.get("standard_name")
        if std_name in std_lonlat:
            lonlat_coords[std_name] = name
            lonlat_dims.update(var.dims)
            if len(var.dims) > 1:
                is_rectilinear = False

    if lonlat_coords:
        lonlat_coords_ = [lonlat_coords.get(name) for name in std_lonlat]
        spatial_arrays = [
            name
            for name, var in ds.variables.items()
            if set(var.dims).intersection(lonlat_dims) and name not in lonlat_coords_
        ]

        return {
            # TODO: EPSG generic spherical model? Or allow something other?
            "crs": [pyproj.CRS.from_epsg(4326)],
            "transform": None,
            "is_rectilinear": is_rectilinear,
            "spatial_dimensions": {dim: ds.sizes[dim] for dim in lonlat_dims},
            "spatial_coordinates": [c for c in lonlat_coords_ if c is not None],
            "spatial_attrs": [],
            "spatial_arrays": spatial_arrays,
            "spatial_var_attrs": [],
        }
    else:
        return None


def reproject_to_common_crs(
    polygons: list[shapely.Polygon],
    from_crs: list[pyproj.CRS],
    to_crs: pyproj.CRS,
) -> list[shapely.Polygon]:
    """Re-project polygons each to a single, common target CRS."""

    assert len(from_crs) == len(polygons)

    transformers: dict[pyproj.CRS, pyproj.Transformer] = {
        crs: pyproj.Transformer.from_crs(crs, to_crs, always_xy=True)
        for crs in set(from_crs)
    }

    out_polygons: list[shapely.Polygon] = []
    for poly, crs in zip(polygons, from_crs):
        tr = transformers[crs]
        out_polygons.append(shapely.transform(poly, tr.transform, interleaved=False))

    return out_polygons


def reproject_to_multiple_crs(
    polygon: shapely.Polygon,
    from_crs: pyproj.CRS,
    to_crs: list[pyproj.CRS],
) -> list[shapely.Polygon]:
    """Re-project a polygon to multiple target CRSs."""

    transformers: dict[pyproj.CRS, pyproj.Transformer] = {
        crs: pyproj.Transformer.from_crs(from_crs, crs, always_xy=True)
        for crs in set(to_crs)
    }

    out_polygons: list[shapely.Polygon] = []
    for crs in to_crs:
        tr = transformers[crs]
        out_polygons.append(shapely.transform(polygon, tr.transform, interleaved=False))

    return out_polygons


def compute_output_extent_latlon(
    input_extents: list[shapely.Polygon],
    input_crss: list[pyproj.CRS],
) -> shapely.Polygon:
    """Compute the spatial extent of the output HEALPix dataset
    as the union of the spatial extents of the input Zarr datasets.

    Returns a polygon with lat-lon WGS84 coordinates.
    """
    # find the most common CRS
    crs_common, _ = Counter(input_crss).most_common(1)[0]

    # TODO: ensure that the common CRS is a projected one
    # (shapely uses a planar geometry engine for unions)

    # re-project all input extents into the common CRS
    input_extents_ = reproject_to_common_crs(input_extents, input_crss, crs_common)

    # compute union and convert to lat-lon WGS84
    output_extent_ = shapely.union_all(input_extents_)

    tr = pyproj.Transformer.from_crs(
        crs_common, pyproj.CRS.from_epsg(4326), always_xy=True
    )
    output_extent = shapely.transform(output_extent_, tr.transform, interleaved=False)

    return output_extent


def create_transform_coords(
    x: xr.DataArray, y: xr.DataArray, transformer: pyproj.Transformer
) -> xr.Coordinates:
    xx, yy = xr.broadcast(
        x.drop_indexes("x", errors="ignore"),
        y.drop_indexes("y", errors="ignore"),
    )

    assert transformer.target_crs is not None
    axis_names = [ax.abbrev.lower() for ax in transformer.target_crs.axis_info]

    converted = xr.apply_ufunc(
        transformer.transform, xx, yy, output_core_dims=[[]] * len(axis_names)
    )

    return xr.Coordinates(
        {name: coord.variable for name, coord in zip(axis_names, converted)},
        indexes={},
    )


def _transform_on_device(
    x: xr.DataArray, y: xr.DataArray, source_crs: pyproj.CRS, target_crs: pyproj.CRS
) -> xr.Coordinates | None:
    """With ``HEALPIX_CONVERT_DEVICE`` set, the lon/lat of a transverse
    Mercator grid (e.g. UTM) to the geographic CRS of its datum, computed in
    torch on that device (see :mod:`healpix_convert.core.tmerc`); else None."""
    import os

    device = os.environ.get("HEALPIX_CONVERT_DEVICE")
    if not device or source_crs.geodetic_crs is None:
        return None
    if not target_crs.equals(source_crs.geodetic_crs, ignore_axis_order=True):
        return None
    from healpix_convert.core.tmerc import tmerc_inverse, tmerc_params

    params = tmerc_params(source_crs)
    if params is None:
        return None
    xx, yy = xr.broadcast(
        x.drop_indexes("x", errors="ignore"),
        y.drop_indexes("y", errors="ignore"),
    )
    lon, lat = tmerc_inverse(
        xx.values.ravel(), yy.values.ravel(), device=device, **params
    )
    if device.startswith("cuda"):
        import torch

        # outside the device slots (see resampler_cache.device_slot): give
        # the memory back to the other processes on the device
        torch.cuda.empty_cache()
    return xr.Coordinates(
        {
            "lon": xr.Variable(xx.dims, lon.reshape(xx.shape)),
            "lat": xr.Variable(xx.dims, lat.reshape(xx.shape)),
        },
        indexes={},
    )


def assign_transform_coords(
    ds: xr.Dataset, source_crs: pyproj.CRS, target_crs: pyproj.CRS
) -> xr.Dataset:
    new_coords = _transform_on_device(ds["x"], ds["y"], source_crs, target_crs)
    if new_coords is None:
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
        new_coords = create_transform_coords(ds["x"], ds["y"], transformer)
    return ds.assign_coords(new_coords)


def compute_output_chunk_info(
    output_extent: shapely.Polygon,
    chunk_level: int,
    ellipsoid_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    # A world rectangle has coincident spherical edges at +/-180 degrees;
    # passing it to polygon_coverage does not describe full-sphere coverage.
    if output_extent.equals(shapely.box(-180, -90, 180, 90)):
        cell_ids = np.arange(12 * 4**chunk_level, dtype=np.uint64)
        return cell_ids, np.ones(cell_ids.size, dtype=bool)

    chunk_cell_ids, _, chunk_is_full = healpix_geo.nested.polygon_coverage(
        shapely.coordinates.get_coordinates(output_extent),
        chunk_level,
        ellipsoid=ellipsoid_name.upper(),
        flat=True,
    )

    return chunk_cell_ids, chunk_is_full


def create_raster_index(
    ds: xr.Dataset, transform: affine.Affine, x_dim: str, y_dim: str
):
    width = ds.sizes[x_dim]
    height = ds.sizes[y_dim]

    raster_index = rasterix.RasterIndex.from_transform(
        transform,
        width=width,
        height=height,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    coords = xr.Coordinates.from_xindex(raster_index)

    return ds.assign_coords(coords)


EARTH_RADIUS_M = 6_371_000


def resolution_to_level(
    resolution: int | float, round_type: Literal["round", "floor", "ceil"] = "round"
) -> int:
    """Determine (approx.) the HEALPix refinement level from a resolution given in meters."""
    if round_type == "floor":
        round_func = np.floor
    elif round_type == "ceil":
        round_func = np.ceil
    else:
        round_func = np.round

    base = EARTH_RADIUS_M * np.sqrt(np.pi / 3.0)
    level = int(round_func(np.log2(base / resolution)))
    return level


_T = TypeVar("_T")


def retry_store_conflicts(
    func: Callable[..., _T], *, attempts: int = 6, delay: float = 0.2
) -> Callable[..., _T]:
    """``func``, retried when the object store refuses a write because of a
    concurrent conditional operation on the same key (S3 ``OperationAborted``,
    which s3fs raises as an ``OSError`` with errno ``EBUSY``).

    Zarr writes the metadata of the parent groups of a new node with a
    conditional put: nodes created in threads under common parents conflict on
    those keys. The creation of a node is idempotent, so it is just retried,
    after a growing random delay.
    """

    def wrapped(*args: Any, **kwargs: Any) -> _T:
        for attempt in range(attempts):
            try:
                return func(*args, **kwargs)
            except OSError as e:
                if e.errno != errno.EBUSY or attempt == attempts - 1:
                    raise
                time.sleep(delay * 2**attempt * (1.0 + random.random()))
        raise AssertionError("unreachable")

    return wrapped

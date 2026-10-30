"""Input datasets of one group may have spatial dimensions of different sizes."""

from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.core.conversion_models import InputGroupSpatialInfo
from healpix_convert.core.utils import extract_spatial_info_cf
from healpix_convert.settings.common import ConvertSettings

GRID = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}


def _grid(lon, lat):
    lon2d, lat2d = np.meshgrid(lon, lat)
    return xr.Dataset(
        {"v": (("lat", "lon"), (1000 * lat2d + lon2d).astype("float32"))},
        coords={
            "lon": (
                "lon",
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                "lat",
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    )


def _write_input(path, ds):
    # footprint of the pixels (half a row of the 0.02° latitude spacing around
    # the centres), so that the footprints of adjacent inputs touch
    pad = 0.011
    west, east = float(ds.lon.min()) - pad, float(ds.lon.max()) + pad
    south, north = float(ds.lat.min()) - pad, float(ds.lat.max()) + pad
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [],
        "id": "synthetic-input",
        "bbox": [west, south, east, north],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ],
        },
        "properties": {"datetime": "2026-10-23T00:00:00Z"},
        "links": [],
        "assets": {},
    }
    tree.to_zarr(path, mode="w", consolidated=True)
    return str(path)


def _convert(inputs, output_path, resampler):
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {**GRID, "refinement_level": 9},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {**GRID, "refinement_level": 4},
                    },
                    "resampler": resampler,
                }
            }
        }
    )
    create_healpix_dataset(inputs, settings, str(output_path))
    group = zarr.open_group(str(output_path), mode="r")["measurements"]
    return group["cell_ids"][:], group["v"][:]


def test_spatial_info_sizes_may_differ():
    a = InputGroupSpatialInfo(
        **extract_spatial_info_cf(_grid(np.arange(4.0), np.arange(3.0)))
    )
    b = InputGroupSpatialInfo(
        **extract_spatial_info_cf(_grid(np.arange(6.0), np.arange(5.0)))
    )
    combined = a + b
    assert len(combined.crs) == 2

    renamed = _grid(np.arange(4.0), np.arange(3.0)).rename(lon="x", lat="y")
    c = InputGroupSpatialInfo(**extract_spatial_info_cf(renamed))
    with pytest.raises(ValueError, match="incompatible spatial info"):
        a + c


def test_split_grid_same_as_single_input(tmp_path):
    # one grid, and the same grid split into a south and a north part with
    # different numbers of rows: same points in the same order, same result
    lon = np.arange(0.0, 2.0, 0.02)
    lat = np.arange(0.0, 2.0, 0.02)
    full = _grid(lon, lat)
    south, north = full.isel(lat=slice(0, 40)), full.isel(lat=slice(40, None))
    resampler = {"name": "bilinear", "init_params": {"threshold": 0.1}}

    single = _convert(
        [_write_input(tmp_path / "full.zarr", full)],
        tmp_path / "single.zarr",
        resampler,
    )
    split = _convert(
        [
            _write_input(tmp_path / "south.zarr", south),
            _write_input(tmp_path / "north.zarr", north),
        ],
        tmp_path / "split.zarr",
        resampler,
    )
    np.testing.assert_array_equal(single[0], split[0])
    np.testing.assert_array_equal(single[1], split[1])
    assert np.isfinite(split[1]).any()


def test_different_longitude_spacing(tmp_path):
    # the longitude spacing changes at 1°N, as in sources with coarser
    # columns towards the poles
    lat = np.arange(0.0, 2.0, 0.02)
    south = _grid(np.arange(0.0, 2.0, 0.02), lat[lat < 1.0])
    north = _grid(np.arange(0.0, 2.0, 0.03), lat[lat >= 1.0])
    cell_ids, v = _convert(
        [
            _write_input(tmp_path / "south.zarr", south),
            _write_input(tmp_path / "north.zarr", north),
        ],
        tmp_path / "out.zarr",
        {"name": "nearest"},
    )
    cell_lon, cell_lat = healpix_geo.nested.healpix_to_lonlat(
        cell_ids, 9, ellipsoid="WGS84"
    )
    inside = (cell_lon > 0.1) & (cell_lon < 1.9) & (cell_lat > 0.1) & (cell_lat < 1.9)
    # both parts of the input reach the output: the latitude encoded in the
    # values is the one of the cell, south and north of the change
    assert (inside & (cell_lat < 0.9)).any() and (inside & (cell_lat > 1.1)).any()
    np.testing.assert_allclose(v[inside] / 1000, cell_lat[inside], atol=0.03)

import numpy as np
import pyproj
import pytest
import xarray as xr

from healpix_convert.core import utils


@pytest.mark.parametrize("epsg", [32630, 32629, 32731])
def test_transform_on_device_same_as_pyproj(epsg, monkeypatch):
    """With HEALPIX_CONVERT_DEVICE, the lon/lat of a UTM grid are computed in
    torch; they agree with pyproj to a few units in the last place."""
    x = xr.DataArray(399965.0 + 10.0 * np.arange(0, 10980, 61), dims="x")
    y = xr.DataArray(4499995.0 - 10.0 * np.arange(0, 10980, 53), dims="y")
    ds = xr.Dataset(coords={"x": x, "y": y})
    src, dst = pyproj.CRS.from_epsg(epsg), pyproj.CRS.from_epsg(4326)

    monkeypatch.delenv("HEALPIX_CONVERT_DEVICE", raising=False)
    ref = utils.assign_transform_coords(ds, src, dst)
    monkeypatch.setenv("HEALPIX_CONVERT_DEVICE", "cpu")
    assert utils._transform_on_device(ds["x"], ds["y"], src, dst) is not None
    got = utils.assign_transform_coords(ds, src, dst)

    for name in ("lon", "lat"):
        assert got[name].dims == ref[name].dims
        np.testing.assert_allclose(got[name].values, ref[name].values, rtol=0, atol=1e-12)


def test_transform_on_device_other_crs(monkeypatch):
    """Not a transverse Mercator grid (here Lambert-93): pyproj, as before."""
    monkeypatch.setenv("HEALPIX_CONVERT_DEVICE", "cpu")
    x = xr.DataArray(np.linspace(1e5, 1e6, 7), dims="x")
    y = xr.DataArray(np.linspace(6.1e6, 7.1e6, 5), dims="y")
    src, dst = pyproj.CRS.from_epsg(2154), pyproj.CRS.from_epsg(4326)
    assert utils._transform_on_device(x, y, src, dst) is None

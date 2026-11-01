"""Reuse of the resamplers between groups (``healpix_convert.resampler_cache``)."""

from __future__ import annotations

import healpix_resample
import numpy as np
import pytest

from healpix_convert import resampler_cache
from healpix_convert.resampler_cache import (
    CACHE_DIR_ENV,
    CACHE_METHODS_ENV,
    build_resampler,
)

LEVEL = 12


@pytest.fixture
def points():
    lon, lat = np.meshgrid(np.linspace(0.0, 0.2, 40), np.linspace(45.0, 45.2, 40))
    lon, lat = lon.ravel(), lat.ravel()
    val = np.sin(lon * 50.0) + lat
    return lon, lat, val


def _kwargs(lon, lat, out_cell_ids=None):
    return dict(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False, out_cell_ids=out_cell_ids,
                nest=True, ellipsoid="WGS84")


def test_second_group_loads_the_resampler(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, val = points
    cls = healpix_resample.NearestResampler
    built = []
    real_init = cls.__init__
    monkeypatch.setattr(cls, "__init__", lambda self, *a, **k: (built.append(1), real_init(self, *a, **k))[1])

    first = build_resampler(cls, "nearest", **_kwargs(lon, lat))
    second = build_resampler(cls, "nearest", **_kwargs(lon.copy(), lat.copy()))
    assert len(built) == 1
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    a, b = first.resample(val), second.resample(val)
    np.testing.assert_array_equal(a.cell_ids, b.cell_ids)
    np.testing.assert_array_equal(a.cell_data, b.cell_data)


def test_other_inputs_other_key(points):
    lon, lat, _ = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    assert key == resampler_cache.resampler_key(cls, _kwargs(lon.copy(), lat.copy()))
    moved = lon.copy()
    moved[0] += 1e-9
    assert key != resampler_cache.resampler_key(cls, _kwargs(moved, lat))
    assert key != resampler_cache.resampler_key(cls, _kwargs(lon, lat, out_cell_ids=np.arange(4, dtype=np.uint64)))
    assert key != resampler_cache.resampler_key(cls, {**_kwargs(lon, lat), "level": LEVEL + 1})
    assert key != resampler_cache.resampler_key(healpix_resample.PSFResampler, _kwargs(lon, lat))


def test_only_the_listed_methods(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, _ = points
    build_resampler(healpix_resample.BilinearResampler, "bilinear", **_kwargs(lon, lat))
    assert not list(tmp_path.glob("*.pkl"))
    monkeypatch.setenv(CACHE_METHODS_ENV, "nearest, bilinear")
    build_resampler(healpix_resample.BilinearResampler, "bilinear", **_kwargs(lon, lat))
    assert len(list(tmp_path.glob("*.pkl"))) == 1


def test_unreadable_file_is_rebuilt(points, monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(CACHE_METHODS_ENV, raising=False)
    lon, lat, val = points
    cls = healpix_resample.NearestResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    (tmp_path / f"{key}.pkl").write_bytes(b"partial")
    r = build_resampler(cls, "nearest", **_kwargs(lon, lat))
    ref = cls(**_kwargs(lon, lat)).resample(val)
    np.testing.assert_array_equal(r.resample(val).cell_data, ref.cell_data)

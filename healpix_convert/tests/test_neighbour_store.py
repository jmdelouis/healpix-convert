"""Neighbour links store of ``healpix_convert.resampler_cache``: the links of a
resampler's neighbour search are kept under its key, and a resampler built on
the same geometry from them is the same, bit for bit."""

from __future__ import annotations

import healpix_resample
import numpy as np
import pytest
import torch

from healpix_convert import resampler_cache
from healpix_convert.resampler_cache import (
    CACHE_DIR_ENV,
    NEIGHBOUR_STORE_ENV,
    build_resampler,
    pack_links,
    unpack_links,
)

pytestmark = pytest.mark.skipif(
    not hasattr(healpix_resample, "NeighbourLinks"),
    reason="healpix-resample without neighbour links",
)


@pytest.fixture
def points():
    lon, lat = np.meshgrid(np.linspace(0.0, 0.2, 50), np.linspace(45.0, 45.2, 50))
    lon, lat = lon.ravel(), lat.ravel()
    return lon, lat, np.sin(lon * 50.0) + lat


def _kwargs(lon, lat):
    return dict(lon_deg=lon, lat_deg=lat, level=12, verbose=False, out_cell_ids=None,
                nest=True, ellipsoid="WGS84", Npt=9, threshold=0.1)


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv(NEIGHBOUR_STORE_ENV, str(tmp_path / "links"))
    return tmp_path / "links"


def test_pack_unpack_round_trip(points):
    lon, lat, _ = points
    links = healpix_resample.PSFResampler(**_kwargs(lon, lat)).neighbours
    links.hi[3, 5] = -1  # a missing neighbour
    back = unpack_links(pack_links(links))
    for name in ("cell_ids", "hi", "tie_rows", "tie_hi"):
        assert torch.equal(getattr(back, name), getattr(links, name).to(torch.int64))


def test_second_build_uses_the_stored_links(points, store, monkeypatch):
    lon, lat, val = points
    cls = healpix_resample.PSFResampler
    first = build_resampler(cls, "psf", **_kwargs(lon, lat))
    assert len(list(store.glob("*.links"))) == 1

    searched = []
    real = healpix_resample.knn.healpix_weighted_nearest
    monkeypatch.setattr(healpix_resample.knn, "healpix_weighted_nearest",
                        lambda *a, **k: (searched.append(1), real(*a, **k))[1])
    second = build_resampler(cls, "psf", **_kwargs(lon.copy(), lat.copy()))
    assert not searched
    for name in ("cell_ids", "hi", "d_m", "M", "MT"):
        a, b = getattr(first, name), getattr(second, name)
        if a.layout != torch.strided:
            a, b = a.to_dense(), b.to_dense()
        assert torch.equal(a, b), name
    assert np.array_equal(first.resample(val, lam=0.1).cell_data, second.resample(val, lam=0.1).cell_data)


def test_other_geometry_other_links(points, store):
    lon, lat, _ = points
    build_resampler(healpix_resample.PSFResampler, "psf", **_kwargs(lon, lat))
    build_resampler(healpix_resample.PSFResampler, "psf", **_kwargs(lon + 1e-6, lat))
    assert len(list(store.glob("*.links"))) == 2


def test_unreadable_links_are_searched_again(points, store):
    lon, lat, val = points
    cls = healpix_resample.PSFResampler
    key = resampler_cache.resampler_key(cls, _kwargs(lon, lat))
    store.mkdir(parents=True)
    (store / f"{key}.links").write_bytes(b"partial")
    r = build_resampler(cls, "psf", **_kwargs(lon, lat))
    ref = cls(**_kwargs(lon, lat))
    assert np.array_equal(r.resample(val, lam=0.1).cell_data, ref.resample(val, lam=0.1).cell_data)


def test_not_for_resamplers_without_links(points, store):
    lon, lat, _ = points
    built = []

    class Other:  # not a KNeighborsResampler: no neighbour search
        def __init__(self, **kwargs):
            built.append(kwargs)

    build_resampler(Other, "other", **_kwargs(lon, lat))
    assert len(built) == 1
    assert not store.exists()

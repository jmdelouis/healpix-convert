"""ERA5Converter on one or several analysis times (MARS time lists, time_chunk).

The GRIB files are written from the ecCodes samples at the cache paths, so
nothing is downloaded; the PSF resampling is replaced by a constant map holding
the first source value, which is enough to check which member and which time
ends up where.
"""

import numpy as np
import pytest

pytest.importorskip("cfgrib")
eccodes = pytest.importorskip("eccodes")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from healpix_convert.converters import era5  # noqa: E402
from healpix_convert.converters.era5 import (  # noqa: E402
    ENDA_GROUP,
    OPER_GROUP,
    ERA5Converter,
    _cache_tag,
)
from healpix_convert.settings.era5 import (  # noqa: E402
    ERA5_ENDA_VARIABLE_META,
    ERA5_OPER_VARIABLE_META,
)

N_CELLS = 12 * 4**7


def _value(param: int, hour: int, number: int) -> float:
    return 1000.0 * param + 10.0 * hour + number


def _write_grib(path, params, hours, numbers=(None,)):
    with open(path, "wb") as f:
        for hour in hours:
            for number in numbers:
                for param in params:
                    gid = eccodes.codes_grib_new_from_samples("reduced_gg_pl_32_grib1")
                    eccodes.codes_set(gid, "localDefinitionNumber", 1)
                    eccodes.codes_set_string(gid, "marsClass", "ea")
                    eccodes.codes_set_string(gid, "marsType", "an")
                    if number is not None:
                        eccodes.codes_set_string(gid, "marsStream", "enda")
                        eccodes.codes_set(gid, "perturbationNumber", number)
                        eccodes.codes_set(gid, "numberOfForecastsInEnsemble", 10)
                    eccodes.codes_set(gid, "table2Version", 128)
                    eccodes.codes_set(gid, "indicatorOfParameter", param)
                    eccodes.codes_set_string(gid, "typeOfLevel", "surface")
                    eccodes.codes_set(gid, "level", 0)
                    eccodes.codes_set(gid, "dataDate", 20260812)
                    eccodes.codes_set(gid, "dataTime", hour * 100)
                    size = eccodes.codes_get_size(gid, "values")
                    eccodes.codes_set_values(
                        gid, np.full(size, _value(param, hour, number or 0))
                    )
                    eccodes.codes_write(gid, f)
                    eccodes.codes_release(gid)


def _param(meta_entry) -> int:
    return int(meta_entry[0].split(".")[0])


@pytest.fixture
def no_psf(monkeypatch):
    """Replace the PSF resampling by a map of the first source value."""
    monkeypatch.setattr(era5, "_get_resampler", lambda lon, lat: None)

    def full_map(self, group, var, t, ds, nr, t_dim):
        value = ds[var].isel({t_dim: t}).values.ravel()[0]
        return np.full(N_CELLS, value, dtype=np.float32)

    monkeypatch.setattr(ERA5Converter, "_full_map", full_map)


def _convert(tmp_path, **kwargs):
    converter = ERA5Converter(date="2026-08-12", local_dir=tmp_path, **kwargs)
    paths = converter.grib_paths()
    enda_hours = [int(t[:2]) for t in converter.times if int(t[:2]) % 3 == 0]
    oper_hours = [int(t[:2]) for t in converter.times]
    _write_grib(
        paths[ENDA_GROUP],
        [_param(v) for v in ERA5_ENDA_VARIABLE_META.values()],
        enda_hours,
        numbers=range(10),
    )
    _write_grib(
        paths[OPER_GROUP],
        [_param(v) for v in ERA5_OPER_VARIABLE_META.values()],
        oper_hours,
    )
    out = str(tmp_path / "out.zarr")
    result = converter.prepare(out)
    for i in range(result.n_chunks):
        converter.convert_group(i)
    converter.consolidate(out)
    return out, result


@pytest.mark.usefixtures("no_psf")
def test_single_time_unchanged(tmp_path) -> None:
    out, result = _convert(tmp_path)  # time="00:00:00"

    assert result.n_times == 1
    root = zarr.open_group(out, mode="r")
    for group in (ENDA_GROUP, OPER_GROUP):
        assert "time" not in root[group]  # no time coordinate for one time
        assert root[group]["t2m" if group == ENDA_GROUP else "tcwv"].chunks == (1, 1024)

    t2m = root[ENDA_GROUP]["t2m"][:]
    assert t2m.shape == (1, N_CELLS)
    np.testing.assert_array_equal(t2m, _value(167, 0, 0))  # member 0


@pytest.mark.usefixtures("no_psf")
def test_several_times_one_chunk(tmp_path) -> None:
    hours = [9, 10, 11, 12]
    out, result = _convert(
        tmp_path,
        time="/".join(f"{h:02d}:00:00" for h in hours),
        time_chunk="all",
    )

    assert result.n_times == 4
    assert [t.hour for t in result.times[ENDA_GROUP]] == [9, 12]  # 3-hourly EDA
    assert [t.hour for t in result.times[OPER_GROUP]] == hours

    root = zarr.open_group(out, mode="r")
    assert root[ENDA_GROUP]["t2m"].chunks == (2, 1024)
    assert root[OPER_GROUP]["tcwv"].chunks == (4, 1024)

    tree = xr.open_datatree(out, engine="zarr", consolidated=True)
    enda = tree[ENDA_GROUP].to_dataset()
    oper = tree[OPER_GROUP].to_dataset()
    assert [str(t)[11:16] for t in enda.time.values] == ["09:00", "12:00"]
    assert [str(t)[11:16] for t in oper.time.values] == [f"{h:02d}:00" for h in hours]

    np.testing.assert_array_equal(
        enda.t2m.isel(cells=0).values, [_value(167, 9, 0), _value(167, 12, 0)]
    )
    np.testing.assert_array_equal(
        oper.tcwv.isel(cells=12345).values, [_value(137, h, 0) for h in hours]
    )
    picked = oper.tcwv.sel(time="2026-08-12T10:46", method="nearest")
    assert str(picked.time.values)[11:16] == "11:00"

    stac = tree.attrs["stac_discovery"]["properties"]
    assert stac["start_datetime"].startswith("2026-08-12T09:00")
    assert stac["end_datetime"].startswith("2026-08-12T12:00")


@pytest.mark.usefixtures("no_psf")
def test_time_chunk_smaller_than_times(tmp_path) -> None:
    out, _ = _convert(tmp_path, time="00:00:00/03:00:00/06:00:00", time_chunk=2)
    root = zarr.open_group(out, mode="r")
    assert root[ENDA_GROUP]["t2m"].chunks == (2, 1024)
    np.testing.assert_array_equal(
        root[ENDA_GROUP]["t2m"][:, 0], [_value(167, h, 0) for h in (0, 3, 6)]
    )


def test_cache_tag() -> None:
    # a single date and time keep the file names of earlier versions
    assert _cache_tag("2025-01-01", ["00:00:00"]) == "20250101T00"
    assert _cache_tag("20300101/to/20300131", ["12:00:00"]) == "20300101-to-20300131T12"
    several = _cache_tag("20260812", ["00:00:00", "12:00:00"])
    assert several.startswith("20260812T00-12n2h")
    assert "/" not in several
    assert several != _cache_tag("20260812", ["00:00:00", "06:00:00", "12:00:00"])


def test_no_eda_time(tmp_path) -> None:
    converter = ERA5Converter(date="2026-08-12", time="10:00:00", local_dir=tmp_path)
    with pytest.raises(ValueError, match="every 3 hours"):
        converter.download()


@pytest.mark.parametrize("time_chunk", [0, -1, "some", 1.5])
def test_time_chunk_checked(time_chunk) -> None:
    with pytest.raises(ValueError, match="time_chunk"):
        ERA5Converter(date="2026-08-12", time_chunk=time_chunk)


def test_masked_resampler_cached_per_mask(monkeypatch) -> None:
    built = []

    class FakePSF:
        def __init__(self, lon_deg, **kwargs):
            built.append(len(lon_deg))

    monkeypatch.setattr(era5, "PSFResampler", FakePSF)
    converter = ERA5Converter(date="2026-08-12")
    lon = lat = np.zeros(10)
    monkeypatch.setattr(converter, "_group_lonlat", lambda group: (lon, lat))

    land = np.arange(10) < 7
    a = converter._valid_resampler(OPER_GROUP, "sst", land)
    b = converter._valid_resampler(OPER_GROUP, "sic", land.copy())
    c = converter._valid_resampler(OPER_GROUP, "sst", np.arange(10) < 5)

    assert a is b  # same mask, shared
    assert c is not a  # the mask changed
    assert built == [7, 5]

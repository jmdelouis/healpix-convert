"""
healpix_convert/converters/era5.py
======================================
ERA5 legacy converter.

Downloads ERA5 from CDS via two MARS requests (enda N256 + oper N320),
projects onto HEALPix level 7 using PSFResampler, and writes an
EOPF-compliant zarr.

Usage
-----
    from healpix_convert.converters.era5 import ERA5Converter
    converter = ERA5Converter(date="2025-01-01")
    result    = converter.prepare(output_path="out.zarr")
    for idx in range(result.n_chunks):
        converter.convert_group(idx)
    converter.consolidate("out.zarr")

Several analysis times go in one store when ``date`` and ``time`` are MARS
lists or ranges, e.g. a whole day with all its times in one chunk::

    converter = ERA5Converter(
        date="20260812",
        time="/".join(f"{h:02d}:00:00" for h in range(24)),
        time_chunk="all",
    )
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import healpix_geo
import numpy as np
import xarray as xr
import zarr
from healpix_resample import PSFResampler

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import (
    CF_GRID_MAPPING_VARIABLE,
    get_cf_cell_id_attrs,
    get_cf_data_variable_attrs,
    get_cf_grid_mapping_attrs,
    get_healpix_group_attrs,
    write_root_conventions,
)
from healpix_convert.core.stac import StacItem
from healpix_convert.settings.common import MetadataSettings
from healpix_convert.settings.era5 import (
    CDS_DATASET,
    CDS_URL,
    DEFAULT_NUMBERS,
    ERA5_ENDA_GRID,
    ERA5_ENDA_VARIABLE_META,
    ERA5_OPER_GRID,
    ERA5_OPER_VARIABLE_META,
)

log = logging.getLogger(__name__)

# ── HEALPix constants ─────────────────────────────────────────────────────────
_CHILD_LEVEL = 7
_CHUNK_LEVEL = 2
_N_CHILD = 12 * 4**_CHILD_LEVEL  # 196,608
_N_PARENT = 12 * 4**_CHUNK_LEVEL  # 192
_CHUNK_SIZE = 4 ** (_CHILD_LEVEL - _CHUNK_LEVEL)  # 1,024

_resampler_cache: dict = {}

ENDA_GROUP = "measurements/enda"
OPER_GROUP = "measurements/oper"


def _mars_list(value: str) -> list[str]:
    """The items of a MARS list (``a/b/c``)."""
    return [item for item in str(value).split("/") if item]


def _enda_times(times: list[str]) -> list[str]:
    """The times at which ERA5 has EDA (stream=enda) analyses: every 3 hours."""
    return [t for t in times if int(t[:2]) % 3 == 0]


def _cache_tag(date: str, times: list[str]) -> str:
    """File-name-safe tag of a MARS date (or date list / range) and time list.

    A single date and time give the tag of earlier versions, e.g.
    ``20250101T00``; a range ``20300101/to/20300131`` becomes
    ``20300101-to-20300131``. Several times are summarised by the first and the
    last hour, their number and a hash of the list.
    """
    tag = date.replace("-", "").replace("/", "-")
    if len(times) == 1:
        return f"{tag}T{times[0][:2]}"
    digest = hashlib.sha1("/".join(times).encode()).hexdigest()[:6]
    return f"{tag}T{times[0][:2]}-{times[-1][:2]}n{len(times)}h{digest}"


def _get_resampler(
    lon: np.ndarray, lat: np.ndarray, level: int = _CHILD_LEVEL, threshold: float = 0.5
) -> PSFResampler:
    key = (id(lon), level)
    if key not in _resampler_cache:
        log.info(f"Building PSFResampler level={level} on {len(lon):,} pts...")
        _resampler_cache[key] = PSFResampler(
            lon_deg=lon,
            lat_deg=lat,
            level=level,
            threshold=threshold,
            verbose=False,
        )
        log.info("PSFResampler ready.")
    return _resampler_cache[key]


@dataclass
class ERA5PrepareResult:
    output_path: str
    n_chunks: int
    n_times: int
    start_dt: datetime
    end_dt: datetime
    grib_enda: Path
    grib_oper: Path
    ds_enda: xr.Dataset
    ds_oper: xr.Dataset
    nr_enda: PSFResampler
    nr_oper: PSFResampler
    lon_enda: np.ndarray
    lat_enda: np.ndarray
    lon_oper: np.ndarray
    lat_oper: np.ndarray
    vars_enda: list[str]
    vars_oper: list[str]
    #: analysis times of each group (the EDA has fewer: it is 3-hourly)
    times: dict[str, list[datetime]] = field(default_factory=dict)


class ERA5Converter:
    """
    Converter for ERA5 reanalysis surface fields to EOPF-DGGS HEALPix zarr.

    Parameters
    ----------
    date       : "YYYY-MM-DD", or a MARS date list or range
                 ("20260812/20260813", "20260801/to/20260831")
    time       : "HH:MM:SS", or a MARS list of times
                 ("00:00:00/06:00:00/12:00:00/18:00:00"). The EDA (enda group)
                 has analyses every 3 hours only: it gets the times among these
                 that fall on 00, 03, ..., 21 UTC.
    step       : MARS step (default "0")
    numbers    : ensemble member numbers (default 0-9); the first one is written
    local_dir  : directory for local GRIB cache (see :py:meth:`grib_paths`)
    metadata   : conventions followed by the output dataset
                 (default: all of them, see :py:class:`MetadataSettings`)
    time_chunk : number of analysis times in one zarr chunk, or "all" for all
                 the times of a group (one chunk then holds the time series of
                 1024 cells). Default 1, one chunk per time.

    With several times, each group gets a ``time`` coordinate (the enda and oper
    groups may have different times).
    """

    def __init__(
        self,
        date: str,
        time: str = "00:00:00",
        step: str = "0",
        numbers: list[int] = None,
        local_dir: Path = Path("."),
        metadata: MetadataSettings | None = None,
        time_chunk: int | Literal["all"] = 1,
    ):
        if time_chunk != "all" and (not isinstance(time_chunk, int) or time_chunk < 1):
            raise ValueError(
                f"time_chunk must be a positive int or 'all', got {time_chunk!r}"
            )
        self.date = date
        self.time = time
        self.times = _mars_list(time)
        if not self.times:
            raise ValueError(f"no time in {time!r}")
        self.time_chunk = time_chunk
        self.step = step
        self.numbers = numbers if numbers is not None else DEFAULT_NUMBERS
        self.local_dir = Path(local_dir)
        self.metadata = metadata if metadata is not None else MetadataSettings()
        self._result: ERA5PrepareResult | None = None
        self._map_cache: dict = {}
        self._sub_resampler_cache: dict = {}

    def grib_paths(self) -> dict[str, Path]:
        """The local GRIB files of each group (downloaded if missing).

        GRIB files already at these paths are used as they are, so data
        downloaded beforehand can be converted by placing it there.
        """
        enda_times = _enda_times(self.times) or self.times
        return {
            ENDA_GROUP: self.local_dir
            / f"era5_enda_{_cache_tag(self.date, enda_times)}_{ERA5_ENDA_GRID}.grib",
            OPER_GROUP: self.local_dir
            / f"era5_oper_{_cache_tag(self.date, self.times)}_{ERA5_OPER_GRID}.grib",
        }

    def download(self, force: bool = False) -> dict[str, Path]:
        """Download the GRIB files from the CDS unless they are already there."""
        if not _enda_times(self.times):
            raise ValueError(
                f"none of the times {self.times} has an ERA5 EDA analysis "
                "(stream=enda, every 3 hours: 00, 03, ..., 21 UTC)"
            )
        paths = self.grib_paths()
        missing = [p for p in paths.values() if force or not p.exists()]
        if missing:
            import cdsapi

            if force or not paths[ENDA_GROUP].exists():
                self._download_enda(paths[ENDA_GROUP], cdsapi.Client(url=CDS_URL))
            if force or not paths[OPER_GROUP].exists():
                self._download_oper(paths[OPER_GROUP], cdsapi.Client(url=CDS_URL))
        return paths

    def dataset(self) -> dict[str, xr.Dataset]:
        """The downloaded GRIB files opened with cfgrib, one dataset per group.

        Each dataset has a ``time`` dimension (of length 1 for a single time)
        and holds the first requested ensemble member only.
        """
        import cfgrib

        datasets = {}
        for group, path in self.grib_paths().items():
            log.info(f"Loading cfgrib {path.name}...")
            ds = xr.merge(
                cfgrib.open_datasets(str(path)), join="override", compat="override"
            )
            # cfgrib names some oper fields by their native shortName, which
            # differs from the keys used in ERA5_OPER_VARIABLE_META. Rename so the
            # variables are recognised (otherwise mslp/sic are silently dropped).
            ds = ds.rename(
                {
                    k: v
                    for k, v in {
                        "msl": "mslp",
                        "siconc": "sic",
                    }.items()
                    if k in ds.data_vars and v not in ds.data_vars
                }
            )
            if "number" in ds.dims:
                ds = ds.isel(number=0)
            if "time" not in ds.dims:
                ds = ds.expand_dims("time")
            datasets[group] = ds
        return datasets

    def prepare(
        self, output_path: str, force_download: bool = False
    ) -> ERA5PrepareResult:
        """Download ERA5 GRIBs (if needed), build PSFResamplers, init zarr."""
        import pandas as pd

        self._map_cache = {}
        self._sub_resampler_cache = {}

        paths = self.download(force=force_download)
        datasets = self.dataset()
        ds_enda, ds_oper = datasets[ENDA_GROUP], datasets[OPER_GROUP]

        lon_enda, lat_enda = self._extract_lonlat(ds_enda)
        lon_oper, lat_oper = self._extract_lonlat(ds_oper)

        nr_enda = _get_resampler(lon_enda, lat_enda)
        nr_oper = _get_resampler(lon_oper, lat_oper)

        times = {
            group: [
                pd.Timestamp(t).tz_localize(None).to_pydatetime().replace(tzinfo=UTC)
                for t in np.atleast_1d(ds["valid_time"].values).ravel()
            ]
            for group, ds in datasets.items()
        }
        all_times = sorted({t for group_times in times.values() for t in group_times})

        vars_enda = [v for v in ERA5_ENDA_VARIABLE_META if v in ds_enda.data_vars]
        vars_oper = [v for v in ERA5_OPER_VARIABLE_META if v in ds_oper.data_vars]
        start_dt, end_dt = all_times[0], all_times[-1]
        n_times = len(all_times)
        n_chunks = _N_PARENT

        self._init_zarr(output_path, times, vars_enda, vars_oper, start_dt, end_dt)

        self._result = ERA5PrepareResult(
            output_path=output_path,
            n_chunks=n_chunks,
            n_times=n_times,
            start_dt=start_dt,
            end_dt=end_dt,
            grib_enda=paths[ENDA_GROUP],
            grib_oper=paths[OPER_GROUP],
            ds_enda=ds_enda,
            ds_oper=ds_oper,
            nr_enda=nr_enda,
            nr_oper=nr_oper,
            lon_enda=lon_enda,
            lat_enda=lat_enda,
            lon_oper=lon_oper,
            lat_oper=lat_oper,
            vars_enda=vars_enda,
            vars_oper=vars_oper,
            times=times,
        )
        log.info(f"ERA5 prepare done: {n_chunks} chunks, {n_times} timestep(s)")
        return self._result

    def convert_group(self, chunk_index: int) -> None:
        """Project one spatial chunk to HEALPix and write to zarr.

        The full-globe PSF resampling is computed once per variable and cached,
        then sliced for each chunk. This avoids recomputing the global resample
        for every one of the spatial chunks.
        """
        assert self._result is not None, "call prepare() first"
        r = self._result

        cell_ids = healpix_geo.nested.zoom_to(
            chunk_index,
            depth=_CHUNK_LEVEL,
            new_depth=_CHILD_LEVEL,
        )[0]
        start = chunk_index * _CHUNK_SIZE
        chunk_sl = slice(start, start + _CHUNK_SIZE)

        root = zarr.open_group(r.output_path, mode="a")

        for group, ds, nr, variables in [
            (ENDA_GROUP, r.ds_enda, r.nr_enda, r.vars_enda),
            (OPER_GROUP, r.ds_oper, r.nr_oper, r.vars_oper),
        ]:
            grp = root[group]

            # all the times of the chunk at once: a zarr chunk may hold several
            for var in variables:
                zarr_a = grp[var]
                zarr_a[:, chunk_sl] = np.stack(
                    [
                        self._full_map(group, var, t, ds, nr, "time")[cell_ids]
                        for t in range(len(r.times[group]))
                    ]
                )

            grp["cell_ids"][chunk_sl] = cell_ids

        log.debug(f"Chunk {chunk_index + 1}/{_N_PARENT} written.")

    def _full_map(
        self,
        group: str,
        var: str,
        t: int,
        ds: xr.Dataset,
        nr: PSFResampler,
        t_dim: str | None,
    ) -> np.ndarray:
        """Return the full-globe HEALPix map for one variable/timestep.

        Computed once via the PSFResampler and cached for reuse across all
        spatial chunks. The PSF is run raw (lam=0, no smoothing) for maximum
        fidelity. Land-masked fields (NaN in the source, e.g. SST/sea-ice over
        land) are resampled from their valid points only, so masked output cells
        stay NaN instead of poisoning the global solve. The result is clipped to
        the source value range [min, max]: this removes only the few Gibbs-ringing
        cells that overshoot beyond what physically exists in the source, never
        alters in-range values, and preserves the sign of signed fields.
        """
        key = (group, var, t)
        full_map = self._map_cache.get(key)
        if full_map is not None:
            return full_map

        nside = 2**_CHILD_LEVEL
        da = ds[var]
        vals = (
            da.isel({t_dim: t}).values.ravel().astype(np.float64)
            if t_dim
            else da.values.ravel().astype(np.float64)
        )
        full_map = np.full(12 * nside**2, np.nan, dtype=np.float32)

        valid = np.isfinite(vals)
        n_valid = int(valid.sum())
        if n_valid == 0:
            self._map_cache[key] = full_map
            return full_map

        if n_valid == vals.size:
            res = nr.resample(vals, lam=0.0)
        else:
            # source has NaN (land mask): resample valid points only
            sub = self._valid_resampler(group, var, valid)
            res = sub.resample(vals[valid], lam=0.0)
        full_map[res.cell_ids] = res.cell_data.astype(np.float32)

        # Clip to the source range: removes deconvolution overshoot only, never
        # extrapolates beyond values present in the source (NaN cells stay NaN).
        src_lo = float(np.nanmin(vals[valid]))
        src_hi = float(np.nanmax(vals[valid]))
        full_map = np.clip(full_map, src_lo, src_hi)

        self._map_cache[key] = full_map
        return full_map

    def _group_lonlat(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        r = self._result
        if group.endswith("enda"):
            return r.lon_enda, r.lat_enda
        return r.lon_oper, r.lat_oper

    def _valid_resampler(self, group: str, var: str, valid: np.ndarray) -> PSFResampler:
        """PSFResampler built on the valid (non-NaN) source points of a field.

        Cached per mask, so that a field whose mask changes with time gets the
        right resampler, and fields sharing a mask (sst, sic) share it.
        """
        k = (group, hashlib.sha1(np.packbits(valid).tobytes()).hexdigest())
        sub = self._sub_resampler_cache.get(k)
        if sub is None:
            lon, lat = self._group_lonlat(group)
            log.info(
                f"Building masked PSFResampler for {group}/{var} "
                f"on {int(valid.sum()):,}/{valid.size:,} valid pts..."
            )
            sub = PSFResampler(
                lon_deg=lon[valid],
                lat_deg=lat[valid],
                level=_CHILD_LEVEL,
                threshold=0.5,
                verbose=False,
            )
            self._sub_resampler_cache[k] = sub
        return sub

    def consolidate(
        self,
        output_path: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """Inject STAC metadata and consolidate zarr."""
        assert self._result is not None
        path = output_path or self._result.output_path
        if storage_options:
            root = zarr.open_group(path, mode="a", storage_options=storage_options)
        else:
            root = zarr.open_group(path, mode="a")
        root.attrs["stac_discovery"] = self._build_stac(path).model_dump()
        zarr.consolidate_metadata(root.store)
        log.info(f"ERA5 zarr consolidated: {path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _download_enda(self, path: Path, client: Any) -> None:
        log.info(f"Downloading ERA5 enda → {path}")
        client.retrieve(
            CDS_DATASET,
            {
                "class": "ea",
                "date": self.date,
                "expver": "1",
                "levtype": "sfc",
                "param": "/".join(v[0] for v in ERA5_ENDA_VARIABLE_META.values()),
                "step": self.step,
                "stream": "enda",
                "time": "/".join(_enda_times(self.times)),
                "type": "an",
                "grid": ERA5_ENDA_GRID,
                "number": "/".join(str(n) for n in self.numbers),
            },
            str(path),
        )

    def _download_oper(self, path: Path, client: Any) -> None:
        log.info(f"Downloading ERA5 oper → {path}")
        client.retrieve(
            CDS_DATASET,
            {
                "class": "ea",
                "date": self.date,
                "expver": "1",
                "levtype": "sfc",
                "param": "/".join(v[0] for v in ERA5_OPER_VARIABLE_META.values()),
                "step": self.step,
                "stream": "oper",
                "time": "/".join(self.times),
                "type": "an",
                "grid": ERA5_OPER_GRID,
            },
            str(path),
        )

    @staticmethod
    def _extract_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        if "values" in ds.dims:
            return (
                ds.longitude.values.ravel().astype(float),
                ds.latitude.values.ravel().astype(float),
            )
        lat2d, lon2d = np.meshgrid(
            ds.latitude.values, ds.longitude.values, indexing="ij"
        )
        return lon2d.ravel().astype(float), lat2d.ravel().astype(float)

    def _init_zarr(
        self,
        output_path: str,
        times: dict[str, list[datetime]],
        vars_enda: list,
        vars_oper: list,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        root = zarr.open_group(output_path, mode="w")
        write_root_conventions(root, self.metadata)

        healpix_model = Healpix(
            refinement_level=_CHILD_LEVEL,
            indexing_scheme="nested",
            ellipsoid={"name": "wgs84"},
        )

        # a time coordinate is only needed to tell several times apart
        several_times = any(len(group_times) > 1 for group_times in times.values())

        for group, variables, meta in [
            (ENDA_GROUP, vars_enda, ERA5_ENDA_VARIABLE_META),
            (OPER_GROUP, vars_oper, ERA5_OPER_VARIABLE_META),
        ]:
            n_times = len(times[group])
            time_chunk = (
                n_times if self.time_chunk == "all" else min(self.time_chunk, n_times)
            )
            grp = root.require_group(group)
            grp.attrs.update(get_healpix_group_attrs(healpix_model, self.metadata))
            grp.create_array(
                "cell_ids",
                shape=(_N_CHILD,),
                dtype=np.int64,
                chunks=(_CHUNK_SIZE,),
                dimension_names=("cells",),
                attributes=get_cf_cell_id_attrs(self.metadata),
            )

            # CF HEALPix grid mapping variable, alongside the DGGS-Zarr convention
            grid_mapping_attrs = get_cf_grid_mapping_attrs(healpix_model, self.metadata)
            if grid_mapping_attrs is not None:
                crs = grp.create_array(
                    CF_GRID_MAPPING_VARIABLE,
                    shape=(),
                    dtype=np.int8,
                    attributes=grid_mapping_attrs,
                )
                crs[...] = 0

            if several_times:
                # analysis times in hours since 00 UTC of the first day (CF)
                epoch = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                time_arr = grp.create_array(
                    "time",
                    shape=(n_times,),
                    dtype=np.int64,
                    chunks=(n_times,),
                    dimension_names=("time",),
                    attributes={
                        "standard_name": "time",
                        "long_name": "analysis time",
                        "axis": "T",
                        "units": f"hours since {epoch:%Y-%m-%d %H:%M:%S}",
                        "calendar": "proleptic_gregorian",
                    },
                )
                time_arr[:] = [(t - epoch) // timedelta(hours=1) for t in times[group]]

            cf_variable_attrs = get_cf_data_variable_attrs(healpix_model, self.metadata)

            for var in variables:
                _, unit, long_name = meta[var]
                grp.create_array(
                    var,
                    shape=(n_times, _N_CHILD),
                    dtype=np.float32,
                    chunks=(time_chunk, _CHUNK_SIZE),
                    fill_value=np.nan,
                    dimension_names=("time", "cells"),
                    attributes={
                        "units": unit,
                        "long_name": long_name,
                        **cf_variable_attrs,
                    },
                )
        zarr.consolidate_metadata(root.store)
        log.info(f"ERA5 zarr skeleton initialised: {output_path}")

    def _build_stac(self, output_path: str) -> StacItem:
        assert self._result is not None
        import shapely

        r = self._result

        # The STAC id must match the product name on disk: derive it from the
        # output path rather than a fresh timestamp (which would not match the
        # filename's creation stamp set at prepare() time).
        name = Path(str(output_path)).name
        product_id = name[:-5] if name.endswith(".zarr") else name

        return StacItem(
            stac_version="1.1.0",
            stac_extensions=[
                "https://stac-extensions.github.io/product/v1.0.0/schema.json"
            ],
            id=product_id,
            bbox=[-180.0, -90.0, 180.0, 90.0],
            geometry=shapely.geometry.mapping(shapely.box(-180, -90, 180, 90)),
            properties={
                "datetime": r.start_dt.isoformat(),
                "start_datetime": r.start_dt.isoformat(),
                "end_datetime": r.end_dt.isoformat(),
                "created": datetime.now(UTC).isoformat(),
                "product:type": "ADF_ECMWA",
                "description": (
                    f"ERA5 reanalysis surface fields (enda + oper streams) on "
                    f"HEALPix NESTED level {_CHILD_LEVEL} (~50 km/pixel)"
                ),
                "healpix:level": _CHILD_LEVEL,
                "healpix:nside": 2**_CHILD_LEVEL,
                "healpix:ordering": "NESTED",
                "source_grid": f"Gaussian {ERA5_ENDA_GRID}/{ERA5_OPER_GRID}",
                "source_dataset": CDS_DATASET,
                "resampling:method": (f"PSFResampler(level={_CHILD_LEVEL})"),
            },
            links=[],
            assets={},
        )

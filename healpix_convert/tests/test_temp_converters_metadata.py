"""The ERA5 / CAMS / ClimateDT converters build their Zarr groups themselves.

Check that they emit the metadata selected via :py:class:`MetadataSettings`,
like the shared conversion path does.

NOTE: those converters are temporary and will be refactored soon, hence the
`temp` in this module name (it should be removed along with them).

"""

from datetime import UTC, datetime

import pytest
import zarr.api.synchronous as zarr

from healpix_convert.settings.common import MetadataSettings

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)

#: Arrays that are not resampled data variables.
NON_DATA_ARRAYS = {"cell_ids", "crs", "number", "time"}


def _init_era5_zarr(path, metadata) -> str:
    from healpix_convert.converters.era5 import ERA5Converter
    from healpix_convert.settings.era5 import (
        ERA5_ENDA_VARIABLE_META,
        ERA5_OPER_VARIABLE_META,
    )

    ERA5Converter(date="2024-01-01", metadata=metadata)._init_zarr(
        str(path),
        {"measurements/enda": [START], "measurements/oper": [START]},
        list(ERA5_ENDA_VARIABLE_META)[:1],
        list(ERA5_OPER_VARIABLE_META)[:1],
        START,
        END,
    )

    return "measurements/enda"


def _init_cams_zarr(path, metadata) -> str:
    pytest.importorskip("requests")

    from healpix_convert.converters.cams import CAMSConverter

    CAMSConverter(date="2024-01-01", metadata=metadata)._init_zarr(
        str(path), 1, START, END
    )

    return "measurements/aod"


def _init_climatedt_zarr(path, metadata) -> str:
    pytest.importorskip("cfgrib")

    from healpix_convert.converters.climatedt import ClimateDTConverter
    from healpix_convert.settings.climatedt import CDT_SFC_VARIABLE_META

    ClimateDTConverter(date="2024-01-01", metadata=metadata)._init_zarr(
        str(path), 1, START, END, CDT_SFC_VARIABLE_META, False
    )

    return "measurements/sfc"


@pytest.fixture(params=["era5", "cams", "climatedt"])
def write_skeleton(request, tmp_path):
    """Write a converter's Zarr skeleton and return its root and HEALPix group."""
    init_zarr = {
        "era5": _init_era5_zarr,
        "cams": _init_cams_zarr,
        "climatedt": _init_climatedt_zarr,
    }[request.param]

    def write(metadata: MetadataSettings | None = None):
        path = tmp_path / f"{request.param}.zarr"
        group_path = init_zarr(path, metadata)
        root = zarr.open_group(str(path), mode="r")

        return root, root[group_path]

    return write


def test_converter_writes_all_conventions(write_skeleton) -> None:
    root, group = write_skeleton()

    # the CF version applies to the whole dataset, so it lives in the root group
    assert root.attrs["Conventions"] == "CF-1.13"

    assert [c["name"] for c in group.attrs["zarr_conventions"]] == ["dggs"]
    assert group.attrs["dggs"]["name"] == "healpix"

    assert group["cell_ids"].attrs["standard_name"] == "healpix_index"
    assert group["crs"].attrs["grid_mapping_name"] == "healpix"

    for name in set(group.array_keys()) - NON_DATA_ARRAYS:
        attrs = group[name].attrs
        assert attrs["grid_mapping"] == "crs"
        assert attrs["coordinates"] == "cell_ids"
        # the input's own description of the variable is kept
        assert "units" in attrs


def test_converter_without_cf(write_skeleton) -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["dggs"]})
    root, group = write_skeleton(metadata)

    assert "Conventions" not in root.attrs
    assert "crs" not in group
    assert dict(group["cell_ids"].attrs) == {}

    # the DGGS convention is still declared
    assert [c["name"] for c in group.attrs["zarr_conventions"]] == ["dggs"]

    for name in set(group.array_keys()) - NON_DATA_ARRAYS:
        attrs = group[name].attrs
        assert "grid_mapping" not in attrs
        assert "coordinates" not in attrs


def test_converter_without_dggs(write_skeleton) -> None:
    metadata = MetadataSettings.model_validate({"zarr_conventions": ["cf"]})
    root, group = write_skeleton(metadata)

    assert "zarr_conventions" not in group.attrs
    assert "dggs" not in group.attrs

    # the CF metadata is still emitted
    assert root.attrs["Conventions"] == "CF-1.13"
    assert group["crs"].attrs["grid_mapping_name"] == "healpix"

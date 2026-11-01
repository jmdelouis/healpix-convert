"""Derived levels aggregated by the chunk conversion tasks (chunk-local
pyramids), against the ones aggregated from the converted levels."""

from __future__ import annotations

from pathlib import PurePath

import numpy as np
import pytest
import xarray as xr
import zarr

from healpix_convert import (
    ConvertSettings,
    convert_group_to_healpix,
    create_healpix_dataset,
    create_staging_cache,
    finalize_healpix_dataset,
    generate_pyramids,
    prepare_healpix_dataset,
)
from healpix_convert import pyramid as pyramid_module
from healpix_convert.pyramid import STAGING_GROUP, chunk_local_pyramid_groups
from healpix_convert.tests.test_pyramid import GROUP, HEALPIX, _write_input


def _settings(
    chunk_level: int, derived_levels, codecs=None, **pyramid_kwargs
) -> ConvertSettings:
    """The input groups converted at the levels 3 and 2, with dense chunks at
    ``chunk_level``, and a pyramid with the given derived levels."""
    chunk = {
        "method": "healpix_cell_dense",
        "healpix": {"refinement_level": chunk_level, **HEALPIX},
        "chunk_buffer_width": 0.0,
    }
    group_settings = {
        f"{GROUP}/{name}": {
            "healpix": {"refinement_level": level, **HEALPIX},
            "chunk": chunk,
            "resampler": {"name": "nearest"},
            **({"codecs": codecs} if codecs is not None else {}),
        }
        for name, level in (("fine", 3), ("coarse", 2))
    }

    return ConvertSettings.model_validate(
        {
            "multiscale_settings": {
                "groups": [GROUP],
                "pyramid_groups": {
                    GROUP: {"derived_levels": list(derived_levels), **pyramid_kwargs}
                },
            },
            "group_settings": group_settings,
        }
    )


SETTINGS = {
    "mean": lambda: _settings(0, [1, 0]),
    "level-1-chunks": lambda: _settings(1, [1]),
    "subgroup": lambda: _settings(0, [1, 0], subgroup_name="pyramid"),
    "max": lambda: _settings(0, [1], reduce="max"),
    "codecs": lambda: _settings(
        0,
        [1, 0],
        codecs=[
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 3}},
        ],
    ),
}


def _workflow(input_path, output_path, settings) -> None:
    """The multi-stage conversion workflow."""
    cache = create_staging_cache([input_path], settings)
    prepare_healpix_dataset(cache, settings, output_path)
    for group in (f"{GROUP}/fine", f"{GROUP}/coarse"):
        convert_group_to_healpix(
            group, cache=cache, settings=settings, output_path=output_path
        )
    generate_pyramids(cache, settings, output_path)
    finalize_healpix_dataset(cache, settings, output_path)


def _one_call(input_path, output_path, settings) -> None:
    create_healpix_dataset([input_path], settings, output_path)


def _assert_same_store(chunk_local: str, reference: str) -> None:
    """Same groups, attributes, arrays (metadata and values) in both stores."""
    a = zarr.open_group(chunk_local, mode="r")
    b = zarr.open_group(reference, mode="r")

    assert STAGING_GROUP not in a
    groups_a = {path for path, _ in a.members(max_depth=None)}
    groups_b = {path for path, _ in b.members(max_depth=None)}
    assert groups_a == groups_b

    # the STAC metadata of the root group refers to the path of the dataset
    assert {k: v for k, v in a.attrs.items() if k != "stac_discovery"} == {
        k: v for k, v in b.attrs.items() if k != "stac_discovery"
    }

    for path, node in b.members(max_depth=None):
        other = a[path]
        assert dict(other.attrs) == dict(node.attrs), path
        if isinstance(node, zarr.Array):
            assert other.metadata.to_dict() == node.metadata.to_dict(), path
            np.testing.assert_array_equal(other[...], node[...], err_msg=path)


@pytest.mark.parametrize("workflow", [_workflow, _one_call])
@pytest.mark.parametrize("name", list(SETTINGS))
def test_same_as_from_the_converted_levels(tmp_path, monkeypatch, name, workflow):
    settings = SETTINGS[name]()
    input_path = _write_input(tmp_path / "input.zarr")
    cache = create_staging_cache([input_path], settings)
    assert chunk_local_pyramid_groups(cache, settings) == [PurePath(GROUP)]

    chunk_local = str(tmp_path / "chunk_local.zarr")
    workflow(input_path, chunk_local, settings)

    # the derived levels aggregated from the converted levels
    with monkeypatch.context() as m:
        m.setattr(pyramid_module, "chunk_local_pyramid_groups", lambda *_: [])
        reference = str(tmp_path / "reference.zarr")
        workflow(input_path, reference, settings)

    _assert_same_store(chunk_local, reference)


def test_derived_level_coarser_than_the_chunks_is_not_chunk_local(tmp_path):
    # the level-0 cells hold the cells of several level-1 chunks
    settings = _settings(1, [1, 0])
    input_path = _write_input(tmp_path / "input.zarr")
    cache = create_staging_cache([input_path], settings)

    assert chunk_local_pyramid_groups(cache, settings) == []


def test_chunks_aggregate_without_reading_the_converted_levels(tmp_path, monkeypatch):
    """generate_pyramids only copies the staged cells: the converted levels are
    not opened (write_derived_levels is not called for the pyramid)."""
    settings = SETTINGS["mean"]()
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")
    cache = create_staging_cache([input_path], settings)
    prepare_healpix_dataset(cache, settings, output_path)
    for group in (f"{GROUP}/fine", f"{GROUP}/coarse"):
        convert_group_to_healpix(
            group, cache=cache, settings=settings, output_path=output_path
        )

    aggregated = []
    real = pyramid_module._build_derived_levels
    monkeypatch.setattr(
        pyramid_module,
        "_build_derived_levels",
        lambda *args, **kwargs: aggregated.append(1) or real(*args, **kwargs),
    )
    generate_pyramids(cache, settings, output_path)

    assert not aggregated
    assert STAGING_GROUP not in zarr.open_group(output_path, mode="r")


def test_output_prepared_without_staging_arrays(tmp_path, monkeypatch):
    """An output dataset prepared without the staging arrays (e.g., by an older
    version): the chunks aggregate nothing and generate_pyramids aggregates the
    derived levels from the converted ones."""
    settings = SETTINGS["mean"]()
    input_path = _write_input(tmp_path / "input.zarr")
    output_path = str(tmp_path / "converted.zarr")
    cache = create_staging_cache([input_path], settings)

    with monkeypatch.context() as m:
        m.setattr(pyramid_module, "create_derived_level_arrays", lambda *a, **k: [])
        m.setattr(
            "healpix_convert.convert.create_derived_level_arrays", lambda *a, **k: []
        )
        prepare_healpix_dataset(cache, settings, output_path)

    for group in (f"{GROUP}/fine", f"{GROUP}/coarse"):
        convert_group_to_healpix(
            group, cache=cache, settings=settings, output_path=output_path
        )
    generate_pyramids(cache, settings, output_path)
    finalize_healpix_dataset(cache, settings, output_path)

    reference = str(tmp_path / "reference.zarr")
    with monkeypatch.context() as m:
        m.setattr(pyramid_module, "chunk_local_pyramid_groups", lambda *_: [])
        _workflow(input_path, reference, settings)

    _assert_same_store(output_path, reference)


# --- lossy codecs on the converted levels: the derived levels aggregate the
# --- values decoded from the converted levels, as when they are read back

#: The serializer of the Sentinel-2 reflectance (sharded, as in production).
HCC_PYRAMID = [
    {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [16],
            "codecs": [
                {
                    "name": "hcc-sentinel2-pyramid",
                    "configuration": {
                        "relative_step": 0.57,
                        "min_step": 1.0,
                        "predictor_size": 5,
                        "quantification_value": 10000.0,
                        "add_offset": -1000.0,
                        "effort": 7,
                    },
                }
            ],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"},
            ],
            "index_location": "end",
        },
    }
]

QUANTIZE = [
    {"name": "numcodecs.quantize", "configuration": {"digits": 2, "dtype": "<f8"}},
    {"name": "bytes", "configuration": {"endian": "little"}},
]


def _write_reflectance_input(path) -> str:
    """The input of :py:func:`_write_input` with reflectance-like values."""
    _write_input(path)
    tree = xr.open_datatree(path, engine="zarr")
    for name, var in (("fine", "a"), ("coarse", "b")):
        ds = tree[f"{GROUP}/{name}"].to_dataset()
        ds[var] = 0.2 + 0.15 * (ds[var] / abs(ds[var]).max())
        tree[f"{GROUP}/{name}"] = xr.DataTree(ds)
    tree = tree.load()
    tree.to_zarr(path, mode="w", consolidated=True)

    return str(path)


@pytest.mark.parametrize("workflow", [_workflow, _one_call])
@pytest.mark.parametrize(
    "codecs",
    [
        pytest.param(HCC_PYRAMID, id="hcc-sentinel2-pyramid"),
        pytest.param(QUANTIZE, id="quantize"),
    ],
)
def test_lossy_codecs_same_as_from_the_converted_levels(
    tmp_path, monkeypatch, codecs, workflow
):
    if codecs is HCC_PYRAMID:
        pytest.importorskip("hcc.codecs.sentinel2_pyramid")
    settings = _settings(0, [1, 0], codecs=codecs)
    input_path = _write_reflectance_input(tmp_path / "input.zarr")

    chunk_local = str(tmp_path / "chunk_local.zarr")
    workflow(input_path, chunk_local, settings)

    with monkeypatch.context() as m:
        m.setattr(pyramid_module, "chunk_local_pyramid_groups", lambda *_: [])
        reference = str(tmp_path / "reference.zarr")
        workflow(input_path, reference, settings)

    _assert_same_store(chunk_local, reference)

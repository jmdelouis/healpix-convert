"""Aggregating a HEALPix group onto the cells of a coarser refinement level."""

from __future__ import annotations

import healpix_geo.nested as hpn
import numpy as np
import pytest
import xarray as xr

from healpix_convert.core.coarsen import (
    coarsen_healpix_group,
    nanbitwise_or,
    nanmode,
)
from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.metadata import get_healpix_group_attrs
from healpix_convert.settings.common import MetadataSettings

LEVEL = 7
HEALPIX = Healpix.model_validate(
    {
        "refinement_level": LEVEL,
        "indexing_scheme": "nested",
        "ellipsoid": {"name": "wgs84"},
    }
)


def _dense_cells(parent_level: int, parents) -> np.ndarray:
    """All the cells of the level 7 within the given coarser cells."""
    cell_ids = hpn.zoom_to(np.asarray(parents, dtype="uint64"), parent_level, LEVEL)

    return np.sort(cell_ids.ravel())


def _group(cell_ids, **variables) -> xr.Dataset:
    """One (single-scale) HEALPix group of a converted dataset."""
    return xr.Dataset(
        {name: ("cells", values) for name, values in variables.items()},
        coords={"cell_ids": ("cells", np.asarray(cell_ids, dtype="uint64"))},
        attrs=get_healpix_group_attrs(HEALPIX, MetadataSettings()),
    )


def test_coarsen_healpix_group():
    # the 16 cells of one level-5 cell
    ds = _group(_dense_cells(5, [100]), a=np.arange(16.0))

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX)

    np.testing.assert_array_equal(coarsened["cell_ids"].values, np.arange(400, 404))
    np.testing.assert_allclose(coarsened["a"].values, [1.5, 5.5, 9.5, 13.5])
    assert coarsened.attrs["dggs"]["refinement_level"] == 6
    assert coarsened["a"].dims == ("cells",)


def test_coarsen_healpix_group_stays_lazy():
    ds = _group(_dense_cells(5, [100, 101]), a=np.arange(32.0)).chunk({"cells": 16})

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX)

    # aggregating the cells of a block needs no cell id and no grouping: it is
    # blockwise, so it stays lazy
    assert coarsened["a"].chunks is not None
    assert coarsened["cell_ids"].chunks is not None
    np.testing.assert_allclose(
        coarsened["a"].values, np.arange(32.0).reshape(-1, 4).mean(axis=1)
    )


def test_coarsen_healpix_group_several_levels():
    cell_ids = _dense_cells(4, [25])
    values = np.arange(cell_ids.size, dtype="float64")
    ds = _group(cell_ids, a=values)

    coarsened = coarsen_healpix_group(ds, 4, healpix=HEALPIX)

    # every cell of the level 7 is aggregated at once (a block mean, not a mean
    # of the means of the levels 6 and 5)
    np.testing.assert_array_equal(coarsened["cell_ids"].values, [25])
    np.testing.assert_allclose(coarsened["a"].values, [values.mean()])


def test_coarsen_healpix_group_reduce_methods():
    ds = _group(_dense_cells(6, [25]), a=np.array([1.0, 2.0, 3.0, 10.0]))

    for reduce, expected in (("min", 1.0), ("max", 10.0), ("sum", 16.0)):
        coarsened = coarsen_healpix_group(
            ds, 6, healpix=HEALPIX, reduce=reduce  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(coarsened["a"].values, [expected])


def test_coarsen_healpix_group_skips_missing_values():
    ds = _group(_dense_cells(6, [25]), a=np.array([1.0, np.nan, 3.0, np.nan]))

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX)

    np.testing.assert_allclose(coarsened["a"].values, [2.0])


def test_coarsen_healpix_group_extra_dimension():
    ds = _group(_dense_cells(6, [25]), a=np.arange(4.0))
    ds["b"] = (("time", "cells"), np.arange(8.0).reshape(2, 4))

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX)

    # like for a generalized ufunc, the HEALPix cell dimension comes last
    assert coarsened["b"].dims == ("time", "cells")
    np.testing.assert_allclose(coarsened["b"].values, [[1.5], [5.5]])


def test_coarsen_healpix_group_keeps_data_type():
    ds = _group(_dense_cells(6, [25]), a=np.array([1, 2, 2, 4], dtype="int16"))

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX)

    # the mean is rounded, not truncated
    assert coarsened["a"].dtype == np.dtype("int16")
    np.testing.assert_array_equal(coarsened["a"].values, [2])


class TestNanmode:
    @pytest.mark.parametrize(
        ["values", "expected"],
        [
            pytest.param([3.0, 1.0, 3.0, 2.0], 3.0, id="most-frequent"),
            # a tie goes to the smallest value, whatever the order of the cells
            pytest.param([5.0, 2.0, 2.0, 5.0], 2.0, id="tie"),
            pytest.param([5.0, 5.0, 2.0, 2.0], 2.0, id="tie-other-order"),
            # missing values are skipped, even when they are the majority
            pytest.param([np.nan, np.nan, np.nan, 4.0], 4.0, id="missing-skipped"),
            pytest.param([np.nan] * 4, np.nan, id="all-missing"),
        ],
    )
    def test_values(self, values, expected):
        np.testing.assert_array_equal(nanmode(np.array([values]), axis=-1), [expected])

    def test_integers(self):
        values = np.array([[7, 7, 1, 2], [0, 3, 3, 0]], dtype="uint8")

        np.testing.assert_array_equal(nanmode(values, axis=(1,)), [7, 0])

    def test_axis(self):
        values = np.array([[1.0, 2.0], [1.0, 3.0], [4.0, 3.0]])

        np.testing.assert_array_equal(nanmode(values, axis=0), [1.0, 3.0])


def test_coarsen_healpix_group_mode():
    # a classification: two level-6 cells, the second one partly without data
    classes = np.array([4, 4, 9, 1, 8, np.nan, np.nan, 2])
    ds = _group(_dense_cells(5, [100])[:8], a=classes).chunk({"cells": 4})

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX, reduce="mode")

    # blockwise, hence lazy, like the other reductions
    assert coarsened["a"].chunks is not None
    np.testing.assert_array_equal(coarsened["a"].values, [4.0, 2.0])


def test_coarsen_healpix_group_mode_keeps_data_type():
    ds = _group(_dense_cells(6, [25]), a=np.array([3, 1, 3, 2], dtype="uint8"))

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX, reduce="mode")

    assert coarsened["a"].dtype == np.dtype("uint8")
    np.testing.assert_array_equal(coarsened["a"].values, [3])


class TestNanbitwiseOr:
    def test_integers(self):
        # a flag set in one cell only is kept
        values = np.array([[0b0001, 0b0000, 0b0100, 0b0000], [0, 0, 0, 0]], "uint8")

        result = nanbitwise_or(values, axis=-1)

        assert result.dtype == np.dtype("uint8")
        np.testing.assert_array_equal(result, [0b0101, 0])

    def test_signed_integers(self):
        # the sign bit is a flag like the others
        values = np.array([[-128, 1, 0, 2]], dtype="int8")

        np.testing.assert_array_equal(nanbitwise_or(values, axis=-1), [-125])

    @pytest.mark.parametrize(
        ["values", "expected"],
        [
            pytest.param([1.0, np.nan, 8.0, np.nan], 9.0, id="missing-skipped"),
            pytest.param([np.nan] * 4, np.nan, id="all-missing"),
            # the highest bits of the integer data type are kept
            pytest.param([2.0**31, np.nan, 1.0, 1.0], 2.0**31 + 1, id="high-bit"),
        ],
    )
    def test_missing_values(self, values, expected):
        # integer flags whose fill value is decoded as NaN
        result = nanbitwise_or(np.array([values]), axis=-1, dtype="uint32")

        np.testing.assert_array_equal(result, [expected])


def test_coarsen_healpix_group_bitwise_or():
    # bit flags: two level-6 cells, the second one partly without data
    flags = np.array([1, 0, 0, 4, 2, np.nan, np.nan, 16])
    ds = _group(_dense_cells(5, [100])[:8], a=flags).chunk({"cells": 4})
    ds["a"].encoding["dtype"] = np.dtype("uint16")

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX, reduce="bitwise_or")

    # blockwise, hence lazy, like the other reductions
    assert coarsened["a"].chunks is not None
    np.testing.assert_array_equal(coarsened["a"].values, [5.0, 18.0])


def test_coarsen_healpix_group_bitwise_or_keeps_data_type():
    flags = np.array([0b1000, 0b0001, 0b0000, 0b1000], dtype="uint8")
    ds = _group(_dense_cells(6, [25]), a=flags)

    coarsened = coarsen_healpix_group(ds, 6, healpix=HEALPIX, reduce="bitwise_or")

    assert coarsened["a"].dtype == np.dtype("uint8")
    # the mode would give 0b1000 only
    np.testing.assert_array_equal(coarsened["a"].values, [0b1001])


@pytest.mark.parametrize(
    ["values", "encoded_dtype", "match"],
    [
        pytest.param(np.arange(4.0), None, "integer bit flags", id="float"),
        # 64-bit flags with a fill value, decoded as float64
        pytest.param(np.arange(4.0), "uint64", "do not hold all", id="decoded-64-bit"),
    ],
)
def test_coarsen_healpix_group_bitwise_or_needs_integers(values, encoded_dtype, match):
    ds = _group(_dense_cells(6, [25]), a=values)
    if encoded_dtype is not None:
        ds["a"].encoding["dtype"] = np.dtype(encoded_dtype)

    with pytest.raises(ValueError, match=match):
        coarsen_healpix_group(ds, 6, healpix=HEALPIX, reduce="bitwise_or")

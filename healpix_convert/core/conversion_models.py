"""
Pydantic model classes for temporary data used during conversion.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Annotated, Any, TypeAlias

import affine
import pyproj
import shapely
import xarray as xr
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_serializer,
    field_validator,
)

from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.multiscales_conventions import HealpixMultiscales
from healpix_convert.core.utils import open_datatrees

#: One (single-scale) HEALPix group of the output dataset: the HEALPix grid of
#: its cells and the refinement level of its chunk cells (each chunk holds the
#: children of one such cell, as the ``healpix_cell_dense`` chunk method
#: writes them).
HealpixLevelInfo: TypeAlias = tuple[Healpix, int]

#: Path of the root group of the output dataset, as Zarr addresses it.
ROOT_GROUP_PATH = ""


def zarr_group_path(path: PurePath | str) -> str:
    """Return the path of a group of the output dataset as Zarr addresses it:
    relative to the root group, which is the empty path.
    """
    posix = PurePath(str(path).strip("/")).as_posix()

    return ROOT_GROUP_PATH if posix == "." else posix


def serialize_crss(value: list[pyproj.CRS]) -> list[dict]:
    return [crs.to_json_dict() for crs in value]


def maybe_deserialize_crss(value: list[dict] | list[pyproj.CRS]) -> list[pyproj.CRS]:
    return [
        v if isinstance(v, pyproj.CRS) else pyproj.CRS.from_json_dict(v) for v in value
    ]


CRSS = Annotated[
    list[pyproj.CRS],
    BeforeValidator(maybe_deserialize_crss),
    PlainSerializer(serialize_crss),
]


def serialize_polys(value: list[shapely.Polygon]) -> list[dict]:
    return [shapely.geometry.mapping(poly) for poly in value]


def maybe_deserialize_polys(
    value: list[dict] | list[shapely.Polygon],
) -> list[shapely.Polygon]:
    return [
        v if isinstance(v, shapely.Polygon) else shapely.geometry.shape(v)
        for v in value
    ]


POLYGONS = Annotated[
    list[shapely.Polygon],
    BeforeValidator(maybe_deserialize_polys),
    PlainSerializer(serialize_polys),
]


class InputSpatialInfo(BaseModel):
    """Spatial information extracted from input Zarr datasets (root group),"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crs: CRSS = Field(default_factory=list)
    """CRS, one per input dataset."""

    geometry_latlon: POLYGONS = Field(default_factory=list)
    """Spatial extent (lat-lon coordinates), one per input dataset."""

    geometry_xy: POLYGONS = Field(default_factory=list)
    """Spatial extent (x-y projected), one per input dataset."""

    def get_extents(self) -> tuple[list[shapely.Polygon], list[pyproj.CRS]]:
        """Return spatial extents (geometry + crs)."""
        geoms = []
        for crs, geom_latlon, geom_xy in zip(
            self.crs, self.geometry_latlon, self.geometry_xy
        ):
            if crs.is_projected:
                if shapely.is_empty(geom_xy):
                    raise ValueError("x-y geometry is missing for projected input CRS")
                geoms.append(geom_xy)
            else:
                geoms.append(geom_latlon)

        return geoms, self.crs


class InputGroupSpatialInfo(BaseModel):
    """Spatial information extracted from an input Zarr group.

    All spatial arrays and/or metadata will be removed in the
    output group converted to HEALPix.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crs: CRSS
    """CRS of the input group, one per input Zarr dataset."""

    transform: list[affine.Affine] | None
    """Affine transforms (if found, if projected CRS), one per input Zarr dataset"""

    is_rectilinear: bool
    """True if x/y or lat/lon gridded input data is rectilinear."""

    spatial_dimensions: dict[str, int]
    """Spatial dimension names and their size."""

    spatial_coordinates: list[str]
    """Spatial coordinate names (should be ordered like X, Y axes or longitude, latitude)."""

    spatial_attrs: list[str]
    """Spatial (group) attribute names."""

    spatial_arrays: list[str]
    """Spatial arrays (data variables) in group."""

    spatial_var_attrs: list[str]
    """Spatial (data-)variable attribute names."""

    @field_validator("transform", mode="before")
    @classmethod
    def maybe_deserialize_transform(
        cls, value: list[tuple] | list[affine.Affine] | None
    ) -> list[affine.Affine] | None:
        if value is None:
            return value

        return [
            val if isinstance(val, affine.Affine) else affine.Affine.from_gdal(*val)
            for val in value
        ]

    @field_serializer("transform", mode="plain")
    def serialize_transform(
        self, value: list[affine.Affine] | None
    ) -> list[tuple] | None:
        if value is None:
            return value

        return [val.to_gdal() for val in value]

    def __add__(self, other: InputGroupSpatialInfo) -> InputGroupSpatialInfo:
        # validate and add another input dataset to the group spatial info
        # (the inputs may have different sizes, e.g. grids of different
        # spacing next to each other: their points are resampled together)
        invalid = (
            list(self.spatial_dimensions) != list(other.spatial_dimensions)
            or self.spatial_coordinates != other.spatial_coordinates
            or self.spatial_arrays != other.spatial_arrays
            or self.is_projected is not other.is_projected
            or self.is_rectilinear is not other.is_rectilinear
        )

        if self.transform is None and other.transform is None:
            transform = None
        elif self.transform is not None and other.transform is not None:
            transform = self.transform + other.transform
        else:
            invalid = True

        if invalid:
            raise ValueError("incompatible spatial info")

        return InputGroupSpatialInfo(
            crs=self.crs + other.crs,
            transform=transform,
            is_rectilinear=self.is_rectilinear,
            spatial_dimensions=self.spatial_dimensions,
            spatial_coordinates=self.spatial_coordinates,
            spatial_attrs=self.spatial_attrs,
            spatial_arrays=self.spatial_arrays,
            spatial_var_attrs=self.spatial_var_attrs,
        )

    @property
    def is_projected(self) -> bool:
        return all(crs.is_projected for crs in self.crs)


class OutputSpatialInfo(BaseModel):
    """Spatial information of the output HEALPix Zarr dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    geometry_latlon: Annotated[
        shapely.Polygon,
        BeforeValidator(
            lambda v: v if isinstance(v, shapely.Polygon) else shapely.geometry.shape(v)
        ),
        PlainSerializer(lambda v: shapely.geometry.mapping(v)),
    ] = Field(default_factory=shapely.Polygon)
    """Spatial extent (lat-lon coordinates)."""

    @property
    def bbox(self) -> list:
        return shapely.bounds(self.geometry_latlon)


class PyramidMultiscaleGroupInfo(BaseModel):
    """Information about a pyramid multiscale group in the output Zarr dataset,
    i.e., a group holding the HEALPix levels converted from the input data
    (native levels) and the coarser levels derived from them (derived
    levels).

    The derived levels are groups of the multiscale group, next to the
    converted ones or under a child group of it (the ``subgroup_name`` pyramid
    setting).

    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    native_levels: dict[PurePath, HealpixLevelInfo] = Field(default_factory=dict)
    """HEALPix grid and chunk level of each level converted from the input data
    (keys: paths of the converted groups in the output dataset).
    """

    derived_levels: dict[PurePath, HealpixLevelInfo] = Field(default_factory=dict)
    """HEALPix grid and chunk level of each derived level (keys: paths of the
    derived level groups in the output dataset).
    """

    @field_serializer("native_levels", "derived_levels")
    def serialize_levels(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}


class OutputGroupInfo(BaseModel):
    """Information about groups in the output Zarr dataset."""

    multiscale_groups: dict[PurePath, HealpixMultiscales] = Field(default_factory=dict)
    """Multiscales groups (including metadata)."""

    pyramid_multiscale_groups: dict[PurePath, PyramidMultiscaleGroupInfo] = Field(
        default_factory=dict
    )
    """Multiscale groups holding a pyramid of derived HEALPix levels
    (keys: paths of those groups in the input dataset(s), like
    :py:attr:`multiscale_groups`, whose metadata describes their whole
    pyramid).
    """

    excluded_groups: list[PurePath] = Field(default_factory=list)
    """Input dataset groups excluded from the output dataset."""

    io_path_map: dict[PurePath, PurePath] = Field(default_factory=dict)
    """Mapping input->output group paths."""

    def group_paths(self) -> set[str]:
        """Return the paths of all the groups of the output dataset, as Zarr
        addresses them (the root group being the empty path).

        The groups converted from the input dataset(s), the HEALPix ones
        included, are the values of :py:attr:`io_path_map`; the derived levels
        of the pyramids are groups of their own.
        """
        paths = {zarr_group_path(path) for path in self.io_path_map.values()}

        for info in self.pyramid_multiscale_groups.values():
            paths |= {zarr_group_path(path) for path in info.derived_levels}

        return paths

    @field_serializer("multiscale_groups", "pyramid_multiscale_groups")
    def serialize_group_paths(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @field_serializer("io_path_map")
    def serialize_io_path_map(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v.as_posix() for k, v in value.items()}

    @field_serializer("excluded_groups")
    def serialize_excluded_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]


class ConvertStagingCache(BaseModel):
    """Stores temporary data and objects reused during the conversion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_datatrees: dict[str, xr.DataTree] = Field(default_factory=dict)
    """Input Zarr datasets opened as raw xarray.DataTree objects."""

    input_spatial: InputSpatialInfo = Field(default_factory=InputSpatialInfo)
    """Spatial information of input Zarr datasets (root group)."""

    input_spatial_groups: dict[PurePath, InputGroupSpatialInfo] = Field(
        default_factory=dict
    )
    """Spatial groups (to convert to HEALPix) in input datasets."""

    output_spatial: OutputSpatialInfo = Field(default_factory=OutputSpatialInfo)
    """Spatial information of the output HEALPix Zarr datasets."""

    output_groups: OutputGroupInfo = Field(default_factory=OutputGroupInfo)
    """Information about groups in the output Zarr dataset."""

    @field_validator("input_datatrees", mode="before")
    @classmethod
    def maybe_open_datatrees(
        cls, inputs: list[str] | dict[str, xr.DataTree]
    ) -> dict[str, xr.DataTree]:
        if isinstance(inputs, list):
            dts = open_datatrees(inputs)
            return {path: dt for path, dt in zip(inputs, dts)}
        elif isinstance(inputs, dict):
            return inputs
        else:
            raise ValueError("invalid input_datatrees argument")

    @field_serializer("input_datatrees", mode="plain")
    def serialize_datatrees(self, dts: dict[str, xr.DataTree]) -> list[str]:
        return list(dts)

    @field_serializer("input_spatial_groups")
    def serialize_input_spatial_groups(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConvertStagingCache:
        """Create a new cache and populate it from a dictionary."""
        return cls.model_validate(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the cache content as a Python dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, value: str) -> ConvertStagingCache:
        """Create a new cache and populate it from a JSON-formatted string."""
        return cls.model_validate_json(value)

    def to_json(self) -> str:
        """Return the cache content as a JSON-formatted string."""
        return self.model_dump_json()

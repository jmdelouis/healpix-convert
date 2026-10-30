"""
Pydantic model classes for conversion settings.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

# pydantic requires typing_extensions.TypedDict (not typing.TypedDict) on Python < 3.12
from typing_extensions import TypedDict

from healpix_convert.core.coarsen import ReduceMethod
from healpix_convert.core.healpix_conventions import Healpix
from healpix_convert.core.utils import check_patterns, matches_any
from healpix_convert.settings.cams import CAMS_CONVERT_SETTINGS
from healpix_convert.settings.climatedt import CDT_SFC_CONVERT_SETTINGS
from healpix_convert.settings.conventions import (
    CFConventionSelector,
    ConventionSelector,
    DGGSConventionSelector,
    MultiscalesConventionSelector,
)
from healpix_convert.settings.dem import DEM_GLO30_CONVERT_SETTINGS
from healpix_convert.settings.era5 import ERA5_CONVERT_SETTINGS
from healpix_convert.settings.stac import STACSettings

log = structlog.get_logger()


class BaseResamplerSettings(BaseModel):
    """Base class settings for healpix-resample."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    init_params: dict[str, Any] = Field(default_factory=dict)
    """Parameters passed to the resampler constructor."""

    resample_params: dict[str, dict[str, Any] | Any] = Field(default_factory=dict)
    """Parameters passed to the resampler's `resample()` method.

    Parameter values may be given either like:
    - a common value used for all arrays (data variables) in the group
    - a value for each array in the group (dictionary where keys are
      variable names)

    """


def broadcast_params(
    params: dict[str, dict[str, Any] | Any],
    var_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Broadcast `params` values for all variables given by `var_names`.

    Returns a dictionary where keys are variable names and values are
    dictionaries of parameter values.
    """

    broadcasted: dict[str, dict[str, Any]] = {vname: {} for vname in var_names}

    for pname, pval in params.items():
        if isinstance(pval, dict):
            missing_vars = set(var_names) - set(pval)
            if missing_vars:
                raise ValueError(
                    f"missing parameter values for variables {missing_vars}"
                )
            for vname, vval in pval.items():
                broadcasted[vname][pname] = vval
        else:
            for vname in var_names:
                broadcasted[vname][pname] = pval

    return dict(broadcasted)


class KNearestNeighborsResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using k-nearest neighbor interpolation."""

    name: Literal["k-nearest"] = "k-nearest"


class NearestResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using nearest neighbor (k=1) interpolation."""

    name: Literal["nearest"] = "nearest"


class BilinearResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using bilinear (k=4) interpolation."""

    name: Literal["bilinear"] = "bilinear"


class PSFResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data onto HEALPix using PSF interpolation."""

    name: Literal["psf"] = "psf"


class CellPointResamplerSettings(BaseResamplerSettings):
    """Settings for resampling data as HEALPix "cell-points" (i.e., cells with maximum
    refinement level 29).
    """

    name: Literal["cell-point"] = "cell-point"


ResamplerSettings: TypeAlias = (
    KNearestNeighborsResamplerSettings
    | NearestResamplerSettings
    | BilinearResamplerSettings
    | PSFResamplerSettings
    | CellPointResamplerSettings
)


class BaseChunkSettings(ABC, BaseModel):
    """Base abstract class for chunking strategy and settings."""

    method: str

    @property
    @abstractmethod
    def is_fixed_size(self) -> bool:
        """Returns True if chunks have a uniform, fixed size."""
        ...

    @property
    @abstractmethod
    def is_healpix_aligned(self) -> bool:
        """Returns True if chunk bounds and/or content are aligned with HEALPix cells."""
        ...


class NoChunkSettings(BaseChunkSettings):
    """Do not chunk output data along the HEALPix cell dimension."""

    method: Literal["no_chunk"] = "no_chunk"

    @property
    def is_fixed_size(self) -> bool:
        return True

    @property
    def is_healpix_aligned(self) -> bool:
        return False


class HealpixUniformChunkSettings(BaseChunkSettings):
    """Common class for fixed-size chunks aligned with HEALPix at a given refinement level."""

    healpix: Healpix
    """HEALPix grid settings used to define "chunk" cells."""

    @property
    def is_fixed_size(self) -> bool:
        return True

    @property
    def is_healpix_aligned(self) -> bool:
        return True

    @model_validator(mode="after")
    def validate_healpix_settings(self) -> HealpixUniformChunkSettings:
        # TODO: add tests
        if self.healpix.refinement_level is None:
            raise ValueError("refinement level must be defined for HEALPix chunks.")

        if self.healpix.indexing_scheme != "nested":
            raise ValueError(
                f"chunk method {self.method!r} only supports the 'nested' indexing scheme."
            )

        return self


class HealpixDenseChunkSettings(HealpixUniformChunkSettings):
    r"""Chunk output data along the HEALPix cell dimension according to
    HEALPix cells given at a parent, "coarse" refinement level (dense version).

    Each of those "chunk" cells are densely populated with adjacent children
    HEALPix cells all at the same "fine" refinement level.

    Chunk size is thus fixed and is given by the following formula:

    .. math:: 4^{(\Delta - \delta)}

    where :math:`\Delta` is the parent level (output chunks) and :math:`\delta` is
    the child level (output data values).

    This chunking strategy works only with the "nested" HEALPIX cell indexing scheme.

    """

    method: Literal["healpix_cell_dense"] = "healpix_cell_dense"

    chunk_buffer_width: float = 60.0
    """Fixed, uniform width to apply around the edges of a HEALPix coarse
    "chunk" cell.

    This parameter is used for spatial filtering of input data prior to
    resampling it onto the HEALPix cells within the chunk. For projected
    coordinates it is in CRS units (usually metres); for geographic longitude
    and latitude coordinates it is in always in meters. It should be greater or equal
    to zero.

    Set the value to zero if no buffer is needed (e.g., some resampling methods
    do not require it).

    """


class HealpixSparseChunkSettings(HealpixUniformChunkSettings):
    r"""Chunk output data along the HEALPix cell dimension according to
    HEALPix cells given at a parent, "coarse" refinement level (sparse version).

    Each of those "chunk" cells are populated with sparsely distributed children
    HEALPix cells given at a "fine" refinement level.

    Chunk size is user defined. There's no strict constraint on the minimum
    size, which should be large enough such that a parent "chunk" cell can
    contain all output data values (children cells). If all children cells are
    at the same refinement level, the maximum size is given by the following
    formula:

    .. math:: 4^{(\Delta - \delta)}

    where :math:`\Delta` is the parent level (output chunks) and :math:`\delta` is
    the child level (output data values).

    This chunking strategy works only with the "nested" HEALPIX cell indexing scheme.

    """

    method: Literal["healpix_cell_sparse"] = "healpix_cell_sparse"

    chunk_size: int
    """User-defined, uniform chunk size."""


ChunkSettings: TypeAlias = (
    NoChunkSettings | HealpixDenseChunkSettings | HealpixSparseChunkSettings
)


class CodecSettings(TypedDict):
    """Settings for a single Zarr codec.

    This class follows Zarr's codec (extension) metadata model.

    """

    name: str
    configuration: dict[str, Any]


class InputMetadataSettings(BaseModel):
    """Settings for the metadata read from the input dataset(s)."""

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    exclude_attrs: list[str] = Field(default_factory=list)
    """Names of the input dataset attributes that must not be propagated to the output.

    Each item is either the exact name of an attribute or a regular expression
    matched against the whole attribute name (e.g., ``"history.*"`` excludes both
    ``"history"`` and ``"history_of_appended_files"``).
    """

    override_attrs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Attributes set on the output variables, overriding the input ones.

    Each key is either the exact name of a variable or a regular expression
    matched against the whole name (e.g., ``"b0[1-9]|b8a"``), and each value the
    attributes set on the matching variables.

    An override wins over ``exclude_attrs``, and the last of several matching
    patterns wins. The attributes written by healpix-convert itself
    (``grid_mapping``, ``coordinates``) keep precedence.
    """

    @field_validator("exclude_attrs")
    @classmethod
    def validate_patterns(cls, value: list[str]) -> list[str]:
        check_patterns(value)

        return value

    @field_validator("override_attrs")
    @classmethod
    def validate_override_patterns(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        check_patterns(value)

        return value

    def is_excluded(self, name: str) -> bool:
        """Returns True if the input attribute with the given name must not be
        propagated to the output.
        """
        return matches_any(name, self.exclude_attrs)

    def filter_attrs(self, attrs: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy of the given input attributes without the excluded ones."""
        return {k: v for k, v in attrs.items() if not self.is_excluded(k)}

    def overridden_attrs(self, name: str) -> dict[str, Any]:
        """Return the attributes set on the output variable with the given name,
        merged from all the matching ``override_attrs`` patterns.
        """
        attrs: dict[str, Any] = {}

        for pattern, overrides in self.override_attrs.items():
            if re.fullmatch(pattern, name):
                attrs.update(overrides)

        return attrs


class ExtraDimensionSettings(BaseModel):
    """How the input datasets of a group are aligned along one of their
    non-spatial dimensions, e.g. the ``detector`` dimension of the Sentinel-2
    ``conditions/geometry`` group, whose labels differ from one tile to the
    next.

    The dimension needs a coordinate variable in every input dataset: its
    labels are what the inputs are aligned on.

    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    join: Literal["outer", "inner", "exact", "reindex"]
    """How the labels of the input datasets are combined:

    - ``"outer"``: the union of their labels;
    - ``"inner"``: the labels they all have;
    - ``"exact"``: they must all have the same labels (error otherwise);
    - ``"reindex"``: the fixed :py:attr:`labels`, the same for every
      conversion; an input label not in that list is an error.

    Where an input has no value for a label of the output, its values are
    missing (NaN), which turns an integer variable into a floating point one.
    """

    labels: list[str] | list[int] | list[float] | None = None
    """Labels of the dimension in the output (``join="reindex"`` only), e.g. the
    detectors of an instrument."""

    @model_validator(mode="after")
    def validate_labels(self) -> ExtraDimensionSettings:
        if self.join == "reindex":
            if not self.labels:
                raise ValueError("join='reindex' requires a non-empty list of labels")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"duplicate labels in {self.labels}")
        elif self.labels is not None:
            raise ValueError(
                f"labels are only used with join='reindex' (got join={self.join!r})"
            )

        return self


def check_codecs_available(
    value: list[CodecSettings] | None,
) -> list[CodecSettings] | None:
    """Check that every codec of a codec pipeline is registered with zarr, so
    that a missing package fails when the settings are created rather than
    mid-conversion."""
    if value is None:
        return value

    from zarr.registry import get_codec_class

    for codec in value:
        try:
            get_codec_class(codec["name"])
        except (KeyError, ValueError):
            # zarr raises KeyError (zarr < 3.2) or UnknownCodecError, a ValueError
            raise ValueError(
                f"zarr codec {codec['name']!r} is not available: install the package "
                "that registers it with zarr"
            ) from None

    return value


class HealpixGroupSettings(BaseModel):
    """Settings for converting a single Zarr group onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    healpix: Healpix
    """Output HEALPix grid settings."""

    chunk: Annotated[ChunkSettings, Field(discriminator="method")]
    """Settings for chunking data along the HEALPix cell dimension."""

    resampler: Annotated[ResamplerSettings, Field(discriminator="name")]
    """Resampling method name and settings."""

    metadata: InputMetadataSettings = Field(default_factory=InputMetadataSettings)
    """Settings for the metadata read from the input dataset(s) and propagated
    to the output group.
    """

    healpix_coordinate_codecs: list[CodecSettings] | None = Field(default=None)
    """Zarr codecs of the HEALPix cell ids coordinate (zarr's defaults if None).

    The cell ids of a chunk are sorted, and one contiguous range with
    ``healpix_cell_dense`` chunks, which zstd alone keeps at about one byte per
    cell. A delta filter before the compressor brings them to almost nothing,
    e.g. ``[{"name": "numcodecs.delta", "configuration": {"dtype": "<u8"}},
    {"name": "bytes"}, {"name": "zstd"}]``; readers then need an implementation
    of that codec. The delta filter is meant for plain cell ids: with
    ``healpix.compression`` set to ``"compacted"`` or ``"ranges"`` the ids are
    already compressed and it brings no further gain.
    """

    extra_dimensions: dict[str, ExtraDimensionSettings] = Field(default_factory=dict)
    """How the input datasets are aligned along their non-spatial dimensions
    (dimension name -> settings), when the group is converted from several
    input datasets.

    The input datasets must otherwise agree on every non-spatial dimension
    (same size and, where there is a coordinate, same labels): a dimension
    along which they differ and that has no entry here is an error.
    """

    codecs: list[CodecSettings] | None = Field(default=None)
    """Zarr codecs of the group's resampled data variables (other arrays use
    zarr's defaults).

    Each codec is looked up by name in zarr's codec registry, so the package
    that provides it must be installed where the settings are created, e.g.
    ``healpix-compress`` for ``hcc-sentinel2-pyramid``.
    """

    output_dtype: str | None = Field(default=None)
    """Data type of the group's resampled data variables, e.g.
    ``"float16"`` or ``"float32"``.

    If None, the variables are written as ``float64`` with the ``"bilinear"``
    and ``"psf"`` resamplers, and with the data type of the input variables
    with the other resamplers (note: data type is coerced after resampling,
    just before writing data). The derived levels of a pyramid have the data
    type of the converted level they are aggregated from.
    """

    @field_validator("output_dtype")
    @classmethod
    def validate_output_dtype(cls, value: str | None) -> str | None:
        if value is None:
            return value

        try:
            dtype = np.dtype(value)
        except TypeError:
            raise ValueError(f"{value!r} is not a data type") from None

        if dtype.kind != "f":
            raise ValueError(
                f"output_dtype must be a floating-point data type, got {value!r}"
            )

        # one spelling per data type, e.g. "f4" -> "float32"
        return dtype.name

    @field_validator("codecs")
    @classmethod
    def validate_codecs(
        cls, value: list[CodecSettings] | None
    ) -> list[CodecSettings] | None:
        return check_codecs_available(value)

    @model_validator(mode="after")
    def validate_extra_dimensions(self) -> HealpixGroupSettings:
        if self.healpix.spatial_dimension in self.extra_dimensions:
            raise ValueError(
                f"extra_dimensions: {self.healpix.spatial_dimension!r} is the HEALPix "
                "cell dimension of the output, not an extra dimension"
            )

        return self

    @model_validator(mode="after")
    def validate_cell_point_resampler_level(self) -> HealpixGroupSettings:
        if self.resampler.name == "cell-point" and self.healpix.refinement_level != 29:
            raise ValueError(
                "HEALPix refinement level must be equal to 29 when the 'cell-point' resampler "
                "is used."
            )

        return self

    @model_validator(mode="after")
    def validate_healpix_chunk_vs_data(self) -> HealpixGroupSettings:
        # validate healpix settings given for output chunks vs. output resampled data
        # TODO: add test
        chunk_healpix: Healpix | None = getattr(self.chunk, "healpix", None)
        if chunk_healpix is None:
            return self

        if self.healpix.indexing_scheme != "nested":
            raise ValueError(
                f"chunk method {self.chunk.method!r} only supports the 'nested' indexing scheme."
            )

        chunk_level = chunk_healpix.refinement_level
        level = self.healpix.refinement_level

        if level is not None and chunk_level is not None and level < chunk_level:
            raise ValueError(
                f"found output HEALPix refinement level {level} "
                f"that is lower than the chunk refinement level {chunk_level}."
            )

        return self


class PyramidSettings(BaseModel):
    """Settings for generating a pyramid of coarser HEALPix refinement levels
    in a multiscale group.

    The derived levels given by :py:attr:`derived_levels` are aggregated from
    the levels converted from the input data (the native levels of the
    multiscale group).

    The derived levels are written in the multiscale group, next to the native
    levels or in a child group of its own (see :py:attr:`subgroup_name`).

    Every child group of the multiscale group must be converted with the
    ``healpix_cell_dense`` chunk method: the cells of such a chunk are all the
    children of one HEALPix cell, which is what lets the derived levels be
    aggregated by position (see
    :py:func:`~healpix_convert.generate_pyramids`).

    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    derived_levels: Annotated[
        list[Annotated[int, Field(ge=0, le=29)]], Field(min_length=1)
    ]
    """HEALPix refinement levels to derive, e.g. ``[18, 16]``.

    Each of them must be coarser than the finest native level of the multiscale
    group and must not be one of its native levels (which are converted from
    the input data).
    """

    subgroup_name: str | None = None
    """Name of a new group, direct child of the multiscale group, where the
    derived levels are written, e.g. "pyramids"::

        reflectance          (multiscale group)
        ├── 20               (native level)
        ├── 19               (native level)
        ├── 17               (native level)
        └── pyramids
            ├── 18           (derived level)
            └── 16           (derived level)

    If None (default), the derived levels are written next to the native
    levels, as direct children of the multiscale group::

        reflectance          (multiscale group)
        ├── 20               (native level)
        ├── 19               (native level)
        ├── 18               (derived level)
        ├── 17               (native level)
        └── 16               (derived level)

    Either way the multiscale group holds every level of the pyramid, and its
    multiscales metadata describes them all.
    """

    reduce: ReduceMethod = "mean"
    """How the values of the children cells are aggregated.

    Missing (NaN) values are skipped. The mean does not suit categorical data
    (masks, classifications): aggregate those with "mode", the most frequent
    value of the cells of a block (the smallest of them in case of a tie).

    Bit flags (integer variables where each bit is a flag of its own, e.g.
    quality flags) are aggregated with "bitwise_or": a flag is set in a coarser
    cell if it is set in any of its cells (missing cells are skipped). It needs
    integer variables only, the data type of which is preserved.
    """

    codecs: list[CodecSettings] | None = Field(default=None)
    """Zarr codecs of the aggregated data variables of every derived level
    (Zarr's defaults if None), e.g. a serializer that codes the cells of one
    HEALPix cell per chunk, which the chunks of the derived levels are.

    As for the ``codecs`` group setting, each codec is looked up by name in
    Zarr's codec registry where the settings are created. The converted levels
    keep the codecs they were written with.
    """

    @field_validator("codecs")
    @classmethod
    def validate_codecs(
        cls, value: list[CodecSettings] | None
    ) -> list[CodecSettings] | None:
        return check_codecs_available(value)

    coarsen_from: Literal["nearest", "finest"] = "nearest"
    """Which native level a variable converted at several levels is aggregated
    from: the nearest level at or finer than the derived level (default),
    i.e., the copies resampled by the data provider are kept and reused, or
    always its finest native level, i.e., those copies are ignored.
    """


class MultiscaleSettings(BaseModel):
    """Settings for a multiscale Zarr groups."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    groups: list[PurePath] = Field(default_factory=list)
    """List of all multiscale groups in input dataset(s).

    Each Zarr group must contain children groups with defined
    HEALPix conversion settings.

    The metadata that will be added to the corresponding ouput group
    is compliant with the "multiscales" Zarr convention.
    """

    add_healpix_positioning: bool = False
    """If True, add HEALPix grid settings as absolute positioning
    information in each multiscale layout entry.

    False by default: it repeats the whole grid of a level in every entry of
    that level, which one entry per array (see :py:attr:`asset_type`) makes
    particularly verbose. The HEALPix grid info of a level can be read from the
    attributes of its group (easily accessible if metadata is consolidated for
    the multiscale group), and the refinement level corresponds to the name of
    a level group.
    """

    asset_type: Literal["group", "array", "auto"] = "auto"
    """What a layout item of the multiscales metadata refers to:

    - "group": one item per level, its asset the path of the level group. A
      pyramid whose derived levels cannot be described that way is rejected,
      i.e. one whose derived level holds variables aggregated from different
      native levels.
    - "array": one item per array of a level, its asset the path of the array
      (e.g. "7/band1"), which describes what every array is aggregated
      from.
    - "auto" (default): "group" when every level can be described by a single
      item, "array" otherwise.
    """

    pyramid_groups: dict[PurePath, PyramidSettings] = Field(default_factory=dict)
    """Settings of the pyramids derived for (some of) the multiscale groups.

    Each key must also be present in :py:attr:`groups`. The pyramids are
    generated by :py:func:`~healpix_convert.generate_pyramids`, after data
    conversion.
    """

    @field_serializer("groups")
    def serialize_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]

    @field_serializer("pyramid_groups")
    def serialize_pyramid_groups(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @model_validator(mode="after")
    def validate_pyramid_groups(self) -> MultiscaleSettings:
        not_multiscale = [
            path.as_posix() for path in self.pyramid_groups if path not in self.groups
        ]
        if not_multiscale:
            raise ValueError(
                f"pyramid groups {not_multiscale} are not multiscale groups "
                "(they must also be listed in the 'groups' field)"
            )

        return self


class ConsolidateMetadataSettings(BaseModel):
    """Settings for the consolidated Zarr metadata written in the output dataset.

    Consolidating the metadata of a Zarr group copies the metadata of all its
    descendants into its own metadata document, so that a reader opens the whole
    hierarchy below it with a single read instead of one read per node. It is
    written by the last stage of the conversion (see
    :py:func:`~healpix_convert.finalize_healpix_dataset`), since such a copy is
    stale as soon as a node is written next to it.

    The settings below select the groups it is written for: the root group (the
    whole output dataset) and, by default, every group a reader is expected to
    open on its own, i.e., a group converted to HEALPix, a multiscale group and
    a derived level group of a pyramid. The consolidated metadata of the root
    group may describe much more than what such a reader needs.

    """

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    root_group: bool = True
    """If True (the default), consolidate all the metadata of the output
    dataset in its root group.
    """

    healpix_groups: bool = True
    """If True (the default), consolidate the metadata of each group converted
    to HEALPix.
    """

    multiscale_groups: bool = True
    """If True (the default), consolidate the metadata of each multiscale
    group, i.e., of all the levels it holds.
    """

    derived_level_groups: bool = True
    """If True (the default), consolidate the metadata of each group holding a
    derived level of a pyramid.
    """

    groups: list[str] = Field(default_factory=list)
    """Paths of other groups of the output dataset to consolidate the metadata
    of (none by default).

    The paths are those of the output dataset, not of the input dataset(s); the
    path of the root group is ``"/"``. A group that the output dataset does not
    hold is reported when the staging cache is created and then skipped.
    """


def _default_zarr_conventions() -> list[Any]:
    return [
        DGGSConventionSelector(),
        MultiscalesConventionSelector(),
        CFConventionSelector(),
    ]


class MetadataSettings(BaseModel):
    """Settings for the metadata written in the output Zarr dataset."""

    model_config = ConfigDict(
        frozen=True, use_attribute_docstrings=True, extra="forbid"
    )

    zarr_conventions: list[ConventionSelector] = Field(
        default_factory=_default_zarr_conventions
    )
    """Conventions followed by the output Zarr dataset.

    Each item is either a convention name (e.g., ``"dggs"``) or a mapping with
    a ``name`` and, optionally, a ``version`` and a ``configuration``
    (see :py:obj:`~healpix_convert.settings.conventions.ConventionSelector`).
    """

    stac: STACSettings = Field(default_factory=STACSettings)
    """Settings for the STAC metadata written in the root group of the output
    dataset (see :py:class:`~healpix_convert.settings.stac.STACSettings`).
    """

    consolidate: ConsolidateMetadataSettings = Field(
        default_factory=ConsolidateMetadataSettings
    )
    """Settings for the consolidated Zarr metadata of the output dataset, i.e.,
    which of its groups it is written for (see
    :py:class:`ConsolidateMetadataSettings`).
    """

    @model_validator(mode="after")
    def validate_unique_conventions(self) -> MetadataSettings:
        names = [selector.name for selector in self.zarr_conventions]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate convention(s) {sorted(duplicates)}")

        return self

    def get(self, name: str) -> ConventionSelector | None:
        """Return the selected convention with the given name, or None if that
        convention is not selected.
        """
        for selector in self.zarr_conventions:
            if selector.name == name.lower():
                return selector

        return None

    def has(self, name: str) -> bool:
        """Returns True if the convention with the given name is selected."""
        return self.get(name) is not None


class ConvertSettings(BaseModel):
    """General settings for converting input Zarr dataset(s) onto HEALPix."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    exclude_groups: list[PurePath] = Field(default_factory=list)
    """Groups to exclude from the conversion (won't be included in output)."""

    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    """Settings for the metadata (conventions) written in the output dataset."""

    multiscale_settings: MultiscaleSettings = Field(default_factory=MultiscaleSettings)
    """Settings for multiscale Zarr groups."""

    group_settings: dict[PurePath, HealpixGroupSettings]
    """Conversion settings per group in the input Zarr dataset(s)."""

    def filter_and_replace(self, func: Callable[[str, dict], dict]) -> ConvertSettings:
        """Filter and replace conversion settings by applying a specified
        function over the settings of each group.

        Parameters
        ----------
        func: callable
            A function which accepts a string (group path) and a dictionary
            (group settings) as arguments and that returns a dictionary (new
            group settings).

        Returns
        -------
        :py:class:`ConvertSettings`
            New conversion settings (validated).

        """
        settings = self.model_dump()

        new_group_settings = {}

        for group_name, group_settings in settings["group_settings"].items():
            new_group_settings[group_name] = func(group_name, group_settings)

        settings["group_settings"] = new_group_settings

        return ConvertSettings.model_validate(settings)

    @field_serializer("exclude_groups")
    def serialize_exclude_groups(self, value):
        # always convert paths to strings (unix-style)
        return [v.as_posix() for v in value]

    @field_serializer("group_settings")
    def serialize_group_settings(self, value):
        # always convert paths to strings (unix-style)
        return {k.as_posix(): v for k, v in value.items()}

    @model_validator(mode="after")
    def validate_multiscales(self) -> ConvertSettings:
        # TODO: check multiscale settings children groups
        # - there exists >1 (or >=1?) direct child paths with HealpixGroupSetting instance
        # - all those instances have the same Healpix settings except refinement level
        # - all those instances have the same resampler settings (at least same method)

        for path, pyramid in self.multiscale_settings.pyramid_groups.items():
            chunk_levels = set()

            for child, child_settings in self.group_settings.items():
                if child.parent != path:
                    continue
                # the dense chunk method also ensures the 'nested' indexing
                # scheme (and thus a defined refinement level)
                if not isinstance(child_settings.chunk, HealpixDenseChunkSettings):
                    raise ValueError(
                        f"pyramid multiscale group {path.as_posix()!r}: group "
                        f"{child.as_posix()!r} is converted with the "
                        f"{child_settings.chunk.method!r} chunk method, but "
                        "generating a pyramid requires the 'healpix_cell_dense' "
                        "method"
                    )
                if not isinstance(child_settings.healpix.coordinate, str):
                    raise ValueError(
                        f"pyramid multiscale group {path.as_posix()!r}: group "
                        f"{child.as_posix()!r} is converted without a cell ids "
                        "coordinate, which generating a pyramid requires"
                    )
                chunk_levels.add(child_settings.chunk.healpix.refinement_level)

            if len(chunk_levels) > 1:
                raise ValueError(
                    f"pyramid multiscale group {path.as_posix()!r}: its groups "
                    f"are chunked with HEALPix cells of different refinement "
                    f"levels {sorted(chunk_levels)}, so the derived levels would "
                    "not hold the same cells"
                )

            levels = sorted(self.native_refinement_levels(path))
            if not levels:
                raise ValueError(
                    f"pyramid multiscale group {path.as_posix()!r} has no child "
                    "group to convert to HEALPix"
                )
            if len(set(pyramid.derived_levels)) != len(pyramid.derived_levels):
                raise ValueError(
                    f"pyramid multiscale group {path.as_posix()!r}: duplicate "
                    f"levels in {pyramid.derived_levels}"
                )

            for level in pyramid.derived_levels:
                if level in levels:
                    raise ValueError(
                        f"pyramid multiscale group {path.as_posix()!r}: the "
                        f"derived level {level} is one of the levels converted "
                        f"from the input data {levels}"
                    )
                if level >= levels[-1]:
                    raise ValueError(
                        f"pyramid multiscale group {path.as_posix()!r}: the "
                        f"derived level {level} must be coarser than the finest "
                        f"native level {levels[-1]} of the group"
                    )

        return self

    def native_refinement_levels(self, group: PurePath) -> dict[int, PurePath]:
        """Return the HEALPix refinement levels at which the children groups of
        a multiscale group are converted (level -> input group path).

        Parameters
        ----------
        group : :py:class:`pathlib.PurePath`
            Path of the multiscale group in the input dataset(s).

        """
        levels = {}

        for path, settings in self.group_settings.items():
            if path.parent != group:
                continue
            level = settings.healpix.refinement_level
            if level is None:
                raise ValueError(
                    f"group {path.as_posix()!r} has no HEALPix refinement level: "
                    f"the {settings.healpix.indexing_scheme!r} indexing scheme is not "
                    "supported in a multiscale group"
                )
            if level in levels:
                raise ValueError(
                    f"multiscale group {group.as_posix()!r} has more than one child "
                    f"group converted at the refinement level {level}"
                )
            levels[level] = path

        return levels

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConvertSettings:
        """Create a new settings object from a dictionary."""
        return cls.model_validate(value)

    def to_dict(self) -> dict[str, Any]:
        """Return settings as a Python dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, value: str) -> ConvertSettings:
        """Create a new settings object from a JSON-formatted string."""
        return cls.model_validate_json(value)

    def to_json(self) -> str:
        """Return settings as a JSON-formatted string."""
        return self.model_dump_json()


def get_settings(name: str) -> ConvertSettings:
    """Get default conversion settings.

    Parameters
    ----------
    name : str
        Name of the product.

    Returns
    -------
    :py:class:`ConvertSettings`

    """
    if name == "era5":
        return ConvertSettings.model_validate(ERA5_CONVERT_SETTINGS)
    elif name == "cams-eac4":
        return ConvertSettings.model_validate(CAMS_CONVERT_SETTINGS)
    elif name == "climatedt-sfc":
        return ConvertSettings.model_validate(CDT_SFC_CONVERT_SETTINGS)
    elif name == "dem-glo30":
        return ConvertSettings.model_validate(DEM_GLO30_CONVERT_SETTINGS)

    else:
        raise ValueError(f"settings not found for {name!r}")


def validate_convert_settings(settings: dict | ConvertSettings) -> ConvertSettings:
    if not isinstance(settings, ConvertSettings):
        try:
            settings = ConvertSettings.model_validate(settings)
        except ValidationError as err:
            log.error(
                "error while validating conversion settings:\n"
                + "\n".join(e["msg"] for e in err.errors())
            )
            raise

    return settings

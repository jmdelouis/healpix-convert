"""Main conversion functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePath
from typing import Any

import dask
import distributed
import shapely
import structlog
import xarray as xr
import zarr.api.synchronous as zarr

from healpix_convert.cache import create_staging_cache
from healpix_convert.core.conversion_models import ConvertStagingCache
from healpix_convert.core.metadata import (
    get_multiscale_group_attrs,
    write_root_conventions,
)
from healpix_convert.core.stac import (
    STAC_PROCESSING_SCHEMA_URL,
    _extract_stac_metadata,
    format_stac_derived_links,
    format_stac_processing,
    merge_stac_items,
)
from healpix_convert.finalize import consolidate_output_metadata
from healpix_convert.healpix_converters import (
    HealpixGroupConverter,
    init_converter,
)
from healpix_convert.pyramid import (
    assemble_chunk_local_levels,
    create_derived_level_arrays,
    create_derived_level_groups,
    write_derived_levels,
)
from healpix_convert.settings.common import (
    ConvertSettings,
    validate_convert_settings,
)

log = structlog.get_logger()


def _convert_group_to_healpix(
    converter: HealpixGroupConverter,
    *,
    chunk_indices: Sequence[int] | None = None,
    resources: dict | None = None,
    load_input_data: bool = False,
):
    log.info(f"••• converting group {converter.output_path} to HEALPix...")

    log.info(
        f"resampling data on HEALPix using {converter.settings.resampler.name} method"
    )

    if converter.settings.chunk.method == "no_chunk":
        converter.convert()
        return

    log.info(f"chunking output data using {converter.settings.chunk.method!r} method")

    # read once here what the chunks need to aggregate the derived levels of
    # the chunk-local pyramids, if any
    converter.setup_chunk_local_pyramid()

    try:
        client = distributed.Client.current()
    except ValueError:
        client = None

    n_chunks = int(math.ceil(converter.cell_dim_size / converter.cell_dim_chunk_size))

    if chunk_indices is None:
        chunk_indices = list(range(n_chunks))

    if any(idx < 0 or idx > n_chunks - 1 for idx in chunk_indices):
        raise ValueError("out of bounds chunk indices")

    if client is not None:
        log.info(
            f"••• converting {len(chunk_indices)} chunks in parallel using dask (distributed)..."
        )

        if resources is None:
            resources = {}

        if load_input_data:
            log.info("persist input dataset(s) on distributed memory with dask")
            converter.load_datasets(client=client)

        # use a wrapper func as passing _convert_one_chunk kwargs to client.map doesn't work
        # (dask/distributed does not like pickling `group_datasets`)
        func = lambda idx: converter.convert(idx)
        with dask.config.set({"optimization.fuse.active": False}):
            futures = client.map(func, chunk_indices, resources=resources)
        client.gather(futures)

    else:
        log.info(
            f"••• converting {len(chunk_indices)} chunks serially (it may take a while)..."
        )

        if load_input_data:
            log.info("pre-load input datasets in memory!")
            converter.load_datasets()

        for chunk_index in chunk_indices:
            converter.convert(chunk_index)

    log.info(f"••• finished converting {len(chunk_indices)} chunks.")


def _create_and_process_output(
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
    convert_data: bool = True,
    dry_run: bool = False,
) -> xr.DataTree:
    if dry_run:
        convert_data = False

    # --- create output Zarr dataset
    # TODO: check for any existing output Zarr dataset
    log.info(f"••• creating {output_path} Zarr dataset...")

    if dry_run:
        root_group = zarr.create_group(store=None)
    else:
        root_group = zarr.open_group(
            output_path, mode="w", storage_options=output_storage_options
        )

    output_store = root_group.store

    log.info("••• propagating stac metadata...")
    stac_settings = settings.metadata.stac
    stac_metadata = _extract_stac_metadata(cache.input_datatrees)
    merged_stac_metadata = merge_stac_items(
        stac_metadata, cache.output_spatial, output_path, stac_settings
    )

    # applied on the propagated metadata only: what is written below describes
    # the conversion itself and must survive the settings
    merged_stac_metadata.properties = stac_settings.filter_properties(
        merged_stac_metadata.properties
    )
    merged_stac_metadata.stac_extensions = stac_settings.filter_extensions(
        merged_stac_metadata.stac_extensions
    )
    merged_stac_metadata.links = stac_settings.filter_links(merged_stac_metadata.links)
    merged_stac_metadata.assets = stac_settings.filter_assets(
        merged_stac_metadata.assets
    )

    # record how this output was produced: without the settings, two outputs
    # resampled from the same input by different methods are indistinguishable
    input_paths = sorted(str(path) for path in cache.input_datatrees)
    merged_stac_metadata.properties.update(
        format_stac_processing(
            settings,
            input_ids=sorted(item.id for item in stac_metadata.values()),
            input_paths=input_paths,
            include_conversion_settings=stac_settings.include_conversion_settings,
        )
    )
    merged_stac_metadata.links.extend(format_stac_derived_links(input_paths))
    if STAC_PROCESSING_SCHEMA_URL not in merged_stac_metadata.stac_extensions:
        merged_stac_metadata.stac_extensions.append(STAC_PROCESSING_SCHEMA_URL)

    root_group.attrs["stac_discovery"] = merged_stac_metadata.model_dump()
    log.info("••• finished propagating stac metadata.")

    # declare the conventions followed by the whole output dataset (root group only)
    write_root_conventions(root_group, settings.metadata)

    # --- process groups
    log.info("••• processing groups...")

    dt0 = next(iter(cache.input_datatrees.values()))

    # the groups are converted once all the arrays of the output dataset are
    # created, the ones of the derived levels included: the chunks of a group
    # may aggregate the derived levels of its pyramid (chunk-local pyramids)
    spatial_paths: list[PurePath] = []

    for path in dt0.groups:
        path = PurePath(path.lstrip("/"))

        if path in cache.output_groups.excluded_groups:
            continue
        if str(path) == ".":
            # root path (skip it here if attributes are written above)
            continue

        group_path_rel = str(cache.output_groups.io_path_map[path])

        if path in cache.output_groups.multiscale_groups:
            log.info(f"••• writing group '{group_path_rel}' to zarr")
            multiscales_obj = cache.output_groups.multiscale_groups[path]
            zarr.create_group(
                output_store,
                path=group_path_rel,
                attributes=get_multiscale_group_attrs(
                    multiscales_obj, settings.metadata
                ),
            )
        elif path in cache.input_spatial_groups:
            log.info(f"••• writing group '{group_path_rel}' to zarr")
            spatial_paths.append(path)
        elif not dt0[str(path)].has_data:
            zarr.create_group(output_store, path=group_path_rel)
        else:
            # non-spatial group (skip for now)
            log.warning(
                f"••• skip writing non-spatial group '{group_path_rel}' to zarr "
                "(not yet supported)"
            )

    # creating the arrays of a group is a few store round trips per array
    # (about 0.5 s per array on object storage): the converters of the groups
    # are initialised in threads, so that these round trips overlap
    converters = []
    if spatial_paths:
        with ThreadPoolExecutor(max_workers=min(16, len(spatial_paths))) as pool:
            converters = list(
                pool.map(
                    lambda path: init_converter(
                        path, cache=cache, settings=settings, output_store=output_store
                    ),
                    spatial_paths,
                )
            )

    # --- derived levels (the coarser levels of the pyramid multiscale groups)
    if settings.multiscale_settings.pyramid_groups:
        log.info("••• creating the derived level groups...")
        create_derived_level_groups(output_store, cache, settings)

        if not dry_run:
            create_derived_level_arrays(
                output_store,
                cache,
                settings,
                output_storage_options=output_storage_options,
            )

    if convert_data:
        for converter in converters:
            _convert_group_to_healpix(converter)

    if convert_data and settings.multiscale_settings.pyramid_groups:
        assembled = assemble_chunk_local_levels(output_store, cache, settings)
        write_derived_levels(
            output_store,
            cache,
            settings,
            output_storage_options=output_storage_options,
            skip=assembled,
        )

    # --- finalize the output dataset
    if convert_data:
        # the whole conversion runs in one call: the output dataset is complete
        # here, so its last stage (finalize_healpix_dataset) runs here too
        consolidate_output_metadata(output_store, cache, settings)

    log.info("••• done!")

    # the metadata of a prepared (or dry run) output dataset is not
    # consolidated yet: the last stage of the conversion writes it
    consolidated = convert_data and settings.metadata.consolidate.root_group

    if dry_run:
        dt_out = xr.open_datatree(
            output_store,  # type: ignore
            engine="zarr",
            chunks="auto",
            consolidated=consolidated,
        )
    else:
        dt_out = xr.open_datatree(
            output_path,
            storage_options=output_storage_options,
            engine="zarr",
            chunks="auto",
            consolidated=consolidated,
        )

    return dt_out


def prepare_healpix_dataset(
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> xr.DataTree:
    """Create a new Zarr dataset without any data resampled on the HEALPix grid (yet).

    All output groups and arrays are created along with their metadata. Arrays
    having a HEALPix cell dimension are still empty.

    The groups of the derived levels of the pyramids and the multiscales
    metadata describing them are created as well; their arrays are written by
    :py:func:`generate_pyramids`. The arrays of the chunk-local pyramids
    (see :py:mod:`healpix_convert.pyramid`) are created here, with the staging
    arrays that the chunk conversion fills.

    Useful for multi-stage conversion workflows:

    - Use :py:func:`create_staging_cache` first to create the conversion
      staging cache.
    - Use :py:func:`convert_group_to_healpix` later (maybe multiple times)
      to resample data on HEALPix and write it to the output dataset
      (fill arrays with data values).
    - Use :py:func:`generate_pyramids` then to aggregate the converted data
      onto the coarser levels of the pyramids.
    - Use :py:func:`finalize_healpix_dataset` last to consolidate the Zarr
      metadata of the output dataset.

    Parameters
    ----------
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache`.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    dry_run : bool, optional
        If True, only display what would have been done. The output
        dataset also won't be written to a persistent storage but
        instead will be created in-memory. No data conversion will
        happen (`convert_data` set to False).
        Default: False.

    Returns
    -------
    :py:class:`xarray.DataTree`
        The output Zarr dataset with HEALPix metadata and empty data, returned
        as an Xarray DataTree object.

    See Also
    --------
    create_staging_cache
    convert_group_to_healpix
    finalize_healpix_dataset

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    return _create_and_process_output(
        cache,
        settings,
        output_path,
        output_storage_options=output_storage_options,
        convert_data=False,
        dry_run=dry_run,
    )


def convert_group_to_healpix(
    group: str,
    *,
    cache: ConvertStagingCache,
    settings: ConvertSettings,
    output_path: str,
    chunk_indices: Sequence[int] | None = None,
    output_storage_options: dict[str, Any] | None = None,
    resources: dict | None = None,
    load_input_data: bool = False,
):
    """Convert data in one group to HEALPix and write the output resampled data to
    the output zarr dataset.

    Parameters
    ----------
    group : str
        Path to the input Zarr group to convert to HEALPix.
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache`.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    chunk_indices : iterable of ints, optional
        Resample and write data only for the given chunk indices. If None (default),
        all chunks will be processed and written.
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    load_input_data : bool, optional
        If True, pre-load all input datasets into memory or presist them
        in distributed memory when using dask (default: False).
        This could speed-up the conversion process but this could also
        consume a lot of memory! Use it carefully.

    See Also
    --------
    create_staging_cache
    prepare_healpix_dataset

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    group_path = PurePath(group.lstrip("/"))

    root_group = zarr.open_group(
        output_path, mode="a", storage_options=output_storage_options
    )
    output_store = root_group.store

    converter = init_converter(
        group_path, cache=cache, settings=settings, output_store=output_store
    )
    _convert_group_to_healpix(
        converter,
        chunk_indices=chunk_indices,
        resources=resources,
        load_input_data=load_input_data,
    )


def generate_pyramids(
    cache: ConvertStagingCache,
    settings: dict | ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
) -> xr.DataTree:
    """Aggregate the converted data onto the coarser HEALPix refinement levels
    of the pyramids.

    Run it once the data has been converted. It writes the arrays of the
    derived level groups that :py:func:`prepare_healpix_dataset` has created,
    one group per derived refinement level of each multiscale group with
    pyramid settings (``pyramid_groups``, see
    :py:attr:`~healpix_convert.ConvertSettings.multiscale_settings`). It does
    nothing if no group has pyramid settings.

    The derived levels written here are not in the consolidated Zarr metadata
    of the output dataset: :py:func:`finalize_healpix_dataset`, the last stage
    of the conversion, writes it.

    Parameters
    ----------
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache` for the same ``settings``.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the converted HEALPix output Zarr dataset, updated in place.
    output_storage_options : dict, optional
        Storage options passed to Zarr for reading the converted dataset and
        writing the pyramids (useful for, e.g., writing on S3 object store).

    Returns
    -------
    :py:class:`xarray.DataTree`
        The output Zarr dataset with the pyramids, returned as an Xarray
        DataTree object.

    See Also
    --------
    create_staging_cache
    prepare_healpix_dataset
    convert_group_to_healpix
    finalize_healpix_dataset

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    if not settings.multiscale_settings.pyramid_groups:
        log.info("••• no group with pyramid settings: nothing to generate.")
    else:
        root_group = zarr.open_group(
            output_path, mode="a", storage_options=output_storage_options
        )
        # the derived levels the chunk conversion tasks have aggregated
        # (chunk-local pyramids), then the others, from the converted levels
        assembled = assemble_chunk_local_levels(root_group.store, cache, settings)
        write_derived_levels(
            root_group.store,
            cache,
            settings,
            output_storage_options=output_storage_options,
            skip=assembled,
        )

        log.info("••• done!")

    # the consolidated metadata of the output dataset, if it has any, does not
    # describe the derived levels written here: the last stage of the
    # conversion (finalize_healpix_dataset) writes it
    return xr.open_datatree(
        output_path,
        storage_options=output_storage_options,
        engine="zarr",
        chunks="auto",
        consolidated=False,
    )


def finalize_healpix_dataset(
    cache: ConvertStagingCache,
    settings: dict | ConvertSettings,
    output_path: str,
    *,
    output_storage_options: dict[str, Any] | None = None,
) -> xr.DataTree:
    """Finalize the converted output dataset.

    The last stage of the conversion, run once every group of the output
    dataset has been written (the pyramids included). It consolidates the Zarr
    metadata of the output dataset: the metadata of all the descendants of a
    group is copied into its own metadata document, so that a reader opens the
    whole hierarchy below it with a single read. The ``consolidate`` metadata
    settings select the groups it is written for (see
    :py:attr:`~healpix_convert.ConvertSettings.metadata`).

    No other stage writes consolidated metadata, since such a copy is stale as
    soon as a node is written next to it. Converting the data does not make it
    stale, though: it only writes array chunks.

    Parameters
    ----------
    cache : :py:class:`ConvertStagingCache`
        A data cache containing information extracted from the input dataset(s)
        and/or pre-computed for the output dataset, as returned by
        :py:func:`create_staging_cache` for the same ``settings``.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the converted HEALPix output Zarr dataset, updated in place.
    output_storage_options : dict, optional
        Storage options passed to Zarr for reading the converted dataset and
        writing its consolidated metadata (useful for, e.g., writing on S3
        object store).

    Returns
    -------
    :py:class:`xarray.DataTree`
        The finalized output Zarr dataset, returned as an Xarray DataTree
        object.

    See Also
    --------
    create_staging_cache
    prepare_healpix_dataset
    convert_group_to_healpix
    generate_pyramids

    """
    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    root_group = zarr.open_group(
        output_path, mode="a", storage_options=output_storage_options
    )
    consolidate_output_metadata(root_group.store, cache, settings)

    log.info("••• done!")

    return xr.open_datatree(
        output_path,
        storage_options=output_storage_options,
        engine="zarr",
        chunks="auto",
        consolidated=settings.metadata.consolidate.root_group,
    )


def create_healpix_dataset(
    input_paths: Sequence[str],
    settings: dict | ConvertSettings,
    output_path: str,
    *,
    groups: str | Sequence[str] | None = None,
    output_extent: dict | shapely.Polygon | None = None,
    output_storage_options: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> xr.DataTree:
    """Create a new Zarr dataset with data resampled on the HEALPix grid.

    Parameters
    ----------
    input_paths : list of paths
        Path(s) to the input Zarr dataset(s) to convert to HEALPix.
    settings : dict or object
        Healpix conversion settings.
    output_path : str
        Path to the HEALPix output Zarr dataset.
    groups : str or list, optional
        Groups (paths) in the input datasets to process. It can be
        either a string (single group or unix-like pattern) or a
        list of groups. By default, all groups found in input datasets
        are processed.
    output_extent : dict or :py:class:`shapely.Polygon`, optional
        Spatial extent of the output HEALPix dataset, given as a
        (GEOJSON-like or object) polygon with lat-lon coordinates.
        If not given (default), the spatial extent will be the union
        of the spatial extents of the input datasets (single polygon).
    output_storage_options : dict, optional
        Storage options passed to Zarr for the dataset creation
        (useful for, e.g., writing on S3 object store).
    dry_run : bool, optional
        If True, only display what would have been done. The output
        dataset also won't be written to a persistent storage but
        instead will be created in-memory. No data conversion will
        happen (`convert_data` set to False).
        Default: False.

    Returns
    -------
    :py:class:`xarray.DataTree`
        The output Zarr dataset with HEALPix data and/or metadata, returned
        as an Xarray DataTree object.

    Notes
    -----
    This function is meant for quick conversions of small inputs and for test
    cases. For production conversions of large datasets, prefer the staged API
    (:py:func:`create_staging_cache`, :py:func:`prepare_healpix_dataset`,
    :py:func:`convert_group_to_healpix`, :py:func:`generate_pyramids` and
    :py:func:`finalize_healpix_dataset`) and run the stages as the tasks of a
    workflow orchestration tool (Prefect, Airflow, Argo Workflows, etc.), which
    handles resource management, task failures and retries.

    """
    log.info(f"••• start converting {len(input_paths)} input datasets into HEALPix...")

    log.info("••• reading conversion settings...")
    settings = validate_convert_settings(settings)

    cache = create_staging_cache(
        input_paths, settings, groups=groups, output_extent=output_extent
    )

    return _create_and_process_output(
        cache,
        settings,
        output_path,
        output_storage_options=output_storage_options,
        convert_data=True,
        dry_run=dry_run,
    )

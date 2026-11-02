"""Reuse of the resamplers between groups.

The groups on a same input grid (e.g. the Sentinel-2 10 m masks) build, for a
given output chunk, the same resampler: same input points, same output cells,
same method and parameters. Building it (the neighbour search) is most of the
cost of a "nearest" group.

With the environment variable ``HEALPIX_CONVERT_RESAMPLER_CACHE`` set to a
directory, the resamplers of the methods listed in
``HEALPIX_CONVERT_RESAMPLER_CACHE_METHODS`` (comma-separated, default
``nearest``) are pickled there, one file per resampler, and the next group
that needs the same resampler loads it instead of building it again. The
directory is shared by the worker processes of a machine; the caller creates
it and removes it when the conversion is done. Without the variable, the
resamplers are built as before.

The neighbour search of a resampler depends only on the geometry, and the
same input grids are resampled onto the same chunks at every revisit. With
``HEALPIX_CONVERT_NEIGHBOUR_STORE`` set to a URL (e.g. ``s3://bucket/links``,
storage options as JSON in ``HEALPIX_CONVERT_NEIGHBOUR_STORE_OPTIONS``), the
links of the neighbour search of the resamplers that have one (healpix-resample
``NeighbourLinks``) are kept there, compressed, under the key of the resampler,
and a resampler built on the same geometry is built from them: the search is
skipped, the result is the same, bit for bit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import tempfile
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()

CACHE_DIR_ENV = "HEALPIX_CONVERT_RESAMPLER_CACHE"
CACHE_METHODS_ENV = "HEALPIX_CONVERT_RESAMPLER_CACHE_METHODS"
NEIGHBOUR_STORE_ENV = "HEALPIX_CONVERT_NEIGHBOUR_STORE"
NEIGHBOUR_STORE_OPTIONS_ENV = "HEALPIX_CONVERT_NEIGHBOUR_STORE_OPTIONS"
LINKS_ONLY_ENV = "HEALPIX_CONVERT_LINKS_ONLY"
DEVICE_ENV = "HEALPIX_CONVERT_DEVICE"


def links_only() -> bool:
    """``HEALPIX_CONVERT_LINKS_ONLY=1``: a conversion only fills the neighbour
    links store (the chunks build their resamplers, nothing is resampled):
    a pass that searches the neighbours of every geometry ahead of the
    conversions."""
    return os.environ.get(LINKS_ONLY_ENV, "") not in ("", "0")


def default_device() -> str | None:
    """``HEALPIX_CONVERT_DEVICE`` (e.g. ``cuda``): the torch device of the
    resamplers that take one and whose settings set none. Unset: the defaults
    of healpix-resample."""
    return os.environ.get(DEVICE_ENV) or None


def _with_device(resampler_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    device = default_device()
    if device is None or "device" in kwargs:
        return kwargs
    import inspect

    takes_device = any(
        "device" in inspect.signature(c.__init__).parameters
        for c in resampler_cls.__mro__
        if "__init__" in c.__dict__
    )
    return {**kwargs, "device": device} if takes_device else kwargs


def _cache_dir(method: str) -> str | None:
    directory = os.environ.get(CACHE_DIR_ENV)
    if not directory:
        return None
    methods = os.environ.get(CACHE_METHODS_ENV, "nearest")
    if method not in {m.strip() for m in methods.split(",")}:
        return None
    return directory


def resampler_key(resampler_cls: type, kwargs: dict[str, Any]) -> str:
    """Digest of everything the resampler is built from: its class, the
    healpix-resample version, the input points, the output cells and the
    other arguments, except the device (the neighbour links do not depend on
    it)."""
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{resampler_cls.__module__}.{resampler_cls.__qualname__}".encode())
    h.update(importlib.metadata.version("healpix-resample").encode())
    for name in sorted(kwargs):
        if name == "device":
            continue
        value = kwargs[name]
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            h.update(f"{value.dtype.str}{value.shape}".encode())
            h.update(value.data)
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def pack_links(links: Any) -> bytes:
    """The neighbour links of a resampler as compressed bytes: the cells as
    differences, the first neighbour of each sample as differences along the
    samples and the others relative to it (small integers), each array saved
    with numpy and compressed with zstd."""
    import numcodecs

    hi = links.hi.cpu().numpy().astype(np.int64)
    first = hi[:, :1]
    arrays = {
        "cell_ids": np.diff(links.cell_ids.cpu().numpy().astype(np.int64), prepend=0),
        "first": np.diff(first[:, 0], prepend=0),
        "others": hi[:, 1:] - first,
        "tie_rows": links.tie_rows.cpu().numpy().astype(np.int64),
        "tie_hi": links.tie_hi.cpu().numpy().astype(np.int64),
    }
    zstd = numcodecs.Zstd(level=3)
    parts, header = [], {}
    for name, arr in arrays.items():
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
        data = zstd.encode(buf.getvalue())
        header[name] = len(data)
        parts.append(data)
    head = json.dumps(header).encode()
    return len(head).to_bytes(4, "little") + head + b"".join(parts)


def unpack_links(data: bytes) -> Any:
    """The neighbour links packed by :func:`pack_links`."""
    import numcodecs
    import torch
    from healpix_resample import NeighbourLinks

    n = int.from_bytes(data[:4], "little")
    header = json.loads(data[4 : 4 + n])
    zstd = numcodecs.Zstd()
    arrays, pos = {}, 4 + n
    for name, size in header.items():
        arrays[name] = np.load(io.BytesIO(zstd.decode(data[pos : pos + size])), allow_pickle=False)
        pos += size
    first = np.cumsum(arrays["first"])[:, None]
    hi = np.concatenate([first, arrays["others"] + first], axis=1)
    return NeighbourLinks(
        cell_ids=torch.from_numpy(np.cumsum(arrays["cell_ids"])),
        hi=torch.from_numpy(hi),
        tie_rows=torch.from_numpy(arrays["tie_rows"]),
        tie_hi=torch.from_numpy(arrays["tie_hi"]),
    )


def _neighbour_store(resampler_cls: type) -> tuple[Any, str] | None:
    """The file system and root of the neighbour links store, if it is on and
    ``resampler_cls`` takes neighbour links."""
    url = os.environ.get(NEIGHBOUR_STORE_ENV)
    if not url:
        return None
    try:
        from healpix_resample import KNeighborsResampler
    except ImportError:  # healpix-resample without neighbour links
        return None
    if not issubclass(resampler_cls, KNeighborsResampler):
        return None
    import fsspec

    options = json.loads(os.environ.get(NEIGHBOUR_STORE_OPTIONS_ENV, "{}"))
    fs, root = fsspec.core.url_to_fs(url, **options)
    return fs, root.rstrip("/")


def _build_with_links(resampler_cls: type, kwargs: dict[str, Any]) -> Any:
    """``resampler_cls(**kwargs)``, from the neighbour links store when it is
    on (see the module docstring)."""
    store = _neighbour_store(resampler_cls)
    if store is None:
        return resampler_cls(**kwargs)
    fs, root = store
    path = f"{root}/{resampler_key(resampler_cls, kwargs)}.links"
    try:
        links = unpack_links(fs.cat_file(path))
    except FileNotFoundError:
        links = None
    except Exception as e:  # unreadable links: search again
        log.warning(f"neighbour store: cannot read {path} ({e!r}), searching")
        links = None
    if links is not None:
        return resampler_cls(**kwargs, neighbours=links)

    resampler = resampler_cls(**kwargs)
    found = getattr(resampler, "neighbours", None)
    if found is not None:
        try:
            fs.makedirs(root, exist_ok=True)
            fs.pipe_file(path, pack_links(found))
        except Exception as e:
            log.warning(f"neighbour store: cannot write {path} ({e!r})")
    return resampler


def build_resampler(resampler_cls: type, method: str, **kwargs: Any) -> Any:
    """``resampler_cls(**kwargs)``, from the resampler cache when it is on
    for ``method``, else from the neighbour links store when it is on (see
    the module docstring). The device is ``HEALPIX_CONVERT_DEVICE`` when it is
    set and ``kwargs`` has none."""
    kwargs = _with_device(resampler_cls, kwargs)
    if links_only():
        # the neighbour links of the resamplers that search them, stored
        if _neighbour_store(resampler_cls) is not None and kwargs.get("search") != "kdtree":
            fs, root = _neighbour_store(resampler_cls)
            if not fs.exists(f"{root}/{resampler_key(resampler_cls, kwargs)}.links"):
                _build_with_links(resampler_cls, kwargs)
        return None

    directory = _cache_dir(method)
    if directory is None:
        return _build_with_links(resampler_cls, kwargs)

    path = os.path.join(directory, resampler_key(resampler_cls, kwargs) + ".pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:  # a partial or unreadable file: build it again
        log.warning(f"resampler cache: cannot read {path} ({e!r}), building the resampler")

    resampler = _build_with_links(resampler_cls, kwargs)
    try:
        os.makedirs(directory, exist_ok=True)
        # write then rename, so another process never reads a partial file
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(resampler, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as e:
        log.warning(f"resampler cache: cannot write {path} ({e!r})")
    return resampler

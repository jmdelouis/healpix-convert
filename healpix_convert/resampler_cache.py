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
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import pickle
import tempfile
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()

CACHE_DIR_ENV = "HEALPIX_CONVERT_RESAMPLER_CACHE"
CACHE_METHODS_ENV = "HEALPIX_CONVERT_RESAMPLER_CACHE_METHODS"


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
    other arguments."""
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{resampler_cls.__module__}.{resampler_cls.__qualname__}".encode())
    h.update(importlib.metadata.version("healpix-resample").encode())
    for name in sorted(kwargs):
        value = kwargs[name]
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            h.update(f"{value.dtype.str}{value.shape}".encode())
            h.update(value.data)
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def build_resampler(resampler_cls: type, method: str, **kwargs: Any) -> Any:
    """``resampler_cls(**kwargs)``, from the resampler cache when it is on
    for ``method`` (see the module docstring)."""
    directory = _cache_dir(method)
    if directory is None:
        return resampler_cls(**kwargs)

    path = os.path.join(directory, resampler_key(resampler_cls, kwargs) + ".pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:  # a partial or unreadable file: build it again
        log.warning(f"resampler cache: cannot read {path} ({e!r}), building the resampler")

    resampler = resampler_cls(**kwargs)
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

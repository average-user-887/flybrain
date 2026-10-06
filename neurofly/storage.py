"""Explicit persistent directory selection; no brain migration or compatibility claim."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class StorageError(ValueError):
    """Selection is unusable; refuse before invoking the daemon."""


def config_path() -> Path:
    try:
        override = os.environ.get("NEUROFLY_CONFIG_HOME")
        if override:
            base = Path(override).expanduser()
            if not base.is_absolute():
                raise StorageError("NEUROFLY_CONFIG_HOME must be absolute; refusing cwd-dependent configuration")
            return base.resolve() / "storage.json"
        raw = os.environ.get("XDG_CONFIG_HOME")
        base = Path(raw or Path.home() / ".config").expanduser()
        if not base.is_absolute():
            raise StorageError("XDG_CONFIG_HOME must be absolute; refusing cwd-dependent configuration")
        return base.resolve() / "neurofly" / "storage.json"
    except (OSError, RuntimeError) as exc:
        raise StorageError(f"cannot resolve storage configuration location: {exc}") from exc



def preflight(raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or not Path(raw).is_absolute():
        raise StorageError(f"{label} must be a nonempty absolute canonical path")
    path = Path(raw)
    if str(path.resolve()) != raw:
        raise StorageError(f"{label} must be canonical: {raw}")
    try:
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise OSError("not an existing readable directory")
        # Open the directory read-only; never create files in adopted stores.
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError as exc:
        raise StorageError(f"{label} unavailable: {raw}: {exc}") from exc
    return path


def read_selection(path: Path | None = None) -> dict | None:
    path = config_path() if path is None else path
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Broken symlinks are damaged configuration, not an absent selection.
        if path.is_symlink():
            raise StorageError(f"storage configuration is a broken symlink: {path}")
        return None
    except (OSError, UnicodeError) as exc:
        raise StorageError(f"cannot read storage configuration {path}: {exc}") from exc
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key}")
                result[key] = value
            return result
        doc = json.loads(raw, object_pairs_hook=unique)
        if (not isinstance(doc, dict) or set(doc) != {"schema", "output_dir", "data_dir"}
                or type(doc["schema"]) is not int or doc["schema"] != 1):
            raise ValueError("unsupported storage schema; expected schema 1 and both directory fields")
        for key in ("output_dir", "data_dir"):
            value = doc[key]
            if (not isinstance(value, str) or not value.strip() or not Path(value).is_absolute()
                    or str(Path(value).resolve()) != value):
                raise ValueError(f"{key} must be a nonempty absolute canonical path")
        return doc
    except (ValueError, OSError, RuntimeError) as exc:
        raise StorageError(f"malformed storage configuration {path}: {exc}") from exc


def adopt(output_dir: str, data_dir: str) -> dict:
    """Remember an explicit selection atomically; never touch store contents."""
    try:
        import fcntl
    except ImportError as exc:
        raise StorageError("safe storage configuration locking is unsupported on this platform") from exc
    selected = {}
    for key, raw in (("output_dir", output_dir), ("data_dir", data_dir)):
        if not raw.strip() or not Path(raw).expanduser().is_absolute():
            raise StorageError(f"{key} adoption requires an absolute existing directory")
        selected[key] = str(preflight(str(Path(raw).expanduser().resolve()), key))
    doc = {"schema": 1, **selected}
    path = config_path()
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stable inode: do not unlink the lock after replacement.
        with (path.parent / ".storage.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageError("storage configuration update already in progress; retry later") from exc
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".storage-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(doc, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as exc:
        raise StorageError(f"cannot update storage configuration {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return doc


def show(checkout: Path) -> dict:
    doc = read_selection()
    if doc is None:
        return {"config": str(config_path()), "configured": False,
                "policy": "unchanged legacy defaults; no store discovery or migration"}
    relationships = {}
    for key in ("output_dir", "data_dir"):
        path = preflight(doc[key], key)
        relationships[key] = {"inside_current_checkout": path.is_relative_to(checkout.resolve()),
                              "retention": "selected directory must survive checkout replacement"}
    return {"config": str(config_path()), "configured": True, **doc,
            "paths": relationships, "preflight": "existing readable directories only; model compatibility is registry-owned"}


def run_arguments(args: list[str], parsed, path: Path | None = None) -> list[str]:
    for label in ("output_dir", "data_dir"):
        value = getattr(parsed, label)
        if value is not None and not value.strip():
            raise StorageError(f"explicit {label} must not be empty or whitespace")
    if parsed.output_dir is not None and parsed.data_dir is not None:
        return args  # Explicit per-run selection does not depend on saved config.
    doc = read_selection(path)
    if doc is None:
        return args
    result = list(args)
    if parsed.output_dir is None:
        result += ["--output-dir", str(preflight(doc["output_dir"], "configured output_dir"))]
    if parsed.data_dir is None and not os.environ.get("NEUROFLY_DATA_DIR", "").strip():
        result += ["--data-dir", str(preflight(doc["data_dir"], "configured data_dir"))]
    return result

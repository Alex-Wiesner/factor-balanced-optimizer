import json
import pandas as pd
import pathlib
import threading
import datetime
import time
import gzip
import pickle
from typing import Any, Dict, Callable

CACHE_ROOT = pathlib.Path.home() / ".cache" / "fetch_data"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

INDEX_PATH = CACHE_ROOT / "cache_index.json"
_INDEX_LOCK = threading.Lock()

CACHE_TTL_SECONDS = 86400


def _load_index() -> Dict[str, Dict[str, Any]]:
    if INDEX_PATH.is_file():
        try:
            with INDEX_PATH.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return {}
    return {}


def _save_index(index: Dict[str, Dict[str, Any]]) -> None:
    tmp_path = INDEX_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(index, fp, separators=(",", ":"))
    tmp_path.replace(INDEX_PATH)


def make_key(prefix: str, **kwargs: Any) -> str:
    parts = [prefix]
    for k in sorted(kwargs):
        v = kwargs[k]
        if isinstance(v, (list, tuple)):
            v = ",".join(map(str, v))
        elif isinstance(v, (datetime.date, datetime.datetime)):
            v = v.isoformat()
        parts.append(f"{k}={v}")
    return "|".join(parts)


def _write_to_disk(data: Any, path: pathlib.Path) -> None:
    if isinstance(data, pd.DataFrame):
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))


def _read_from_disk(path: pathlib.Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


def cached_fetch(
    key: str,
    fetch_fn: Callable[..., Any],
    *,
    ttl: int = CACHE_TTL_SECONDS,
    **fetch_kwards: Any,
) -> Any:
    now = time.time()
    with _INDEX_LOCK:
        idx = _load_index()
        entry = idx.get(key)

        if entry and (ttl == 0 or now - entry["timestamp"] < ttl):
            try:
                cached_path = CACHE_ROOT / entry["path"]
                return _read_from_disk(cached_path)
            except Exception:
                pass

    result = fetch_fn(**fetch_kwards)

    safe_name = key.replace("|", "_").replace("=", "-")
    if isinstance(result, pd.DataFrame):
        filename = f"{safe_name}.pkl.gz"
    else:
        filename = f"{safe_name}.json"

    cache_path = CACHE_ROOT / filename
    _write_to_disk(result, cache_path)

    with _INDEX_LOCK:
        idx = _load_index()
        idx[key] = {"path": filename, "timestamp": now}
        _save_index(idx)

    return result


def clear_cache() -> None:
    if CACHE_ROOT.is_dir():
        for child in CACHE_ROOT.iterdir():
            try:
                child.unlink()
            except Exception:
                pass
        try:
            CACHE_ROOT.rmdir()
        except Exception:
            pass
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    _save_index({})

from __future__ import annotations

import functools
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import torch
from tqdm.auto import tqdm as _tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar("T")
log = logging.getLogger(__name__)


def setup_logging(log_path: Path | str | None = None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    for h in handlers:
        h.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def with_retry_once(stage_name: str) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            for attempt in (1, 2):
                try:
                    log.info("Stage [%s] attempt %d starting", stage_name, attempt)
                    t0 = time.time()
                    result = fn(*args, **kwargs)
                    log.info("Stage [%s] attempt %d completed in %.1fs",
                             stage_name, attempt, time.time() - t0)
                    return result
                except Exception as e:
                    log.error("Stage [%s] attempt %d FAILED: %s", stage_name, attempt, e)
                    log.error("Traceback:\n%s", traceback.format_exc())
                    if attempt == 1:
                        log.warning("Retrying stage [%s] once...", stage_name)
                        free_cuda_memory()
                        time.sleep(5)
                    else:
                        log.error("Stage [%s] failed twice; skipping", stage_name)
                        return None
            return None
        return wrapper
    return decorator


def checkpoint_exists(path: Path | str) -> bool:
    return Path(path).exists()


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def free_cuda_memory() -> None:
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "(no CUDA)"
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / 1024**3
    total_gb = total / 1024**3
    return f"VRAM used: {used_gb:.1f}/{total_gb:.1f} GB"


def safe_model_slug(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def disk_free_gb(path: Path | str) -> float:
    import shutil
    return shutil.disk_usage(str(path)).free / 1024**3


def progress_bar(iterable: Iterable, *, desc: str, total: int | None = None, **kwargs):
    return _tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=not sys.stdout.isatty(),
        leave=True,
        **kwargs,
    )


redirect_logging_to_tqdm = logging_redirect_tqdm

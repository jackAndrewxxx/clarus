"""Persistent kernel tuning cache.

`triton.autotune` re-benchmarks every config in every new process. The engine
instead benchmarks once per (kernel, shape key), stores the winner in a JSON
file per GPU model and triton version, and later processes launch the cached
config directly. Combined with Triton's own on-disk binary cache this makes warm
starts fast.
"""
from __future__ import annotations

import json
import os
import re
import threading

import torch
import triton
from triton.testing import do_bench

_CACHE_DIR = os.environ.get(
    "CLARUS_TUNE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                 "cache"),
)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")


def default_cache_path() -> str:
    """One file per (GPU model, triton version): a config tuned on one card is never
    silently reused on another. CLARUS_TUNE_CACHE overrides the full path."""
    if os.environ.get("CLARUS_TUNE_CACHE"):
        return os.environ["CLARUS_TUNE_CACHE"]
    dev = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    return os.path.join(_CACHE_DIR, f"tune_{_slug(dev)}_triton{triton.__version__}.json")


class Tuner:
    def __init__(self, path: str | None = None):
        self._path = path
        self.lock = threading.Lock()
        self.cache: dict[str, dict] | None = None
        self.frozen = False  # set while capturing graphs: a miss is then a bug

    @property
    def path(self) -> str:
        if self._path is None:
            self._path = default_cache_path()
        return self._path

    def _load(self):
        if self.cache is None:
            self.cache = {}
            if os.path.exists(self.path):
                with open(self.path) as f:
                    self.cache = json.load(f)
        return self.cache

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f, indent=0, sort_keys=True)
        os.replace(tmp, self.path)

    def pick(self, name: str, key: tuple, candidates: list[dict], launch) -> dict:
        k = f"{name}|{','.join(map(str, key))}"
        cfg = self._load().get(k)
        if cfg is not None:
            return cfg
        if self.frozen:
            raise RuntimeError(f"untuned kernel shape during graph capture: {k}")
        if len(candidates) == 1:
            best = candidates[0]
        else:
            best, best_t = None, float("inf")
            for c in candidates:
                try:
                    t = do_bench(lambda: launch(c), warmup=2, rep=8)
                except (triton.runtime.errors.OutOfResources, RuntimeError):
                    continue
                if t < best_t:
                    best, best_t = c, t
            if best is None:
                raise RuntimeError(f"no valid config for {k}")
        with self.lock:
            self.cache[k] = best
            self._save()
        return best


TUNER = Tuner()


def cfg(bm, bn, bk, w, s, **extra):
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, **extra)


def p2(x: int) -> int:
    return triton.next_power_of_2(max(1, int(x)))

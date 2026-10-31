"""Persistent kernel tuning cache.

`triton.autotune` re-benchmarks every config in every new process. The engine
instead benchmarks once per (kernel, shape key), stores the winner in a JSON
file per GPU model and triton version, and later processes launch the cached
config directly. Combined with Triton's own on-disk binary cache this makes warm
starts fast.

State is partitioned by tuning target (GPU model + triton version): a process that
builds models on two different GPU models keeps two caches / files. Saves merge with
the file on disk under an exclusive lock, so concurrent processes keep each other's
entries.
"""
from __future__ import annotations

import json
import os
import re
import threading
import warnings

import torch
import triton
from triton.testing import do_bench

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

_CACHE_DIR = os.environ.get(
    "CLARUS_TUNE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                 "cache"),
)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")


def _device_name(device=None) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if device is None:
        idx = torch.cuda.current_device()
    else:
        device = torch.device(device)
        if device.type != "cuda":
            return device.type
        idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_name(idx)


def target_of(device=None) -> str:
    """Tuning target of a device: '<GPU model slug>_triton<version>'."""
    return f"{_slug(_device_name(device))}_triton{triton.__version__}"


def default_cache_path(device=None) -> str:
    """One file per (GPU model, triton version): a config tuned on one card is never
    silently reused on another. CLARUS_TUNE_CACHE overrides the full path; it may contain
    `{target}`, which expands to the target (otherwise see Tuner.path_for)."""
    target = target_of(device)
    env = os.environ.get("CLARUS_TUNE_CACHE")
    if env:
        return os.path.abspath(env.replace("{target}", target))
    return os.path.abspath(os.path.join(_CACHE_DIR, f"tune_{target}.json"))


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Conservative: do the memory spans of two tensors intersect?"""
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False

    def span(t):
        lo = t.data_ptr()
        hi = lo + (sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride())) + 1) * t.element_size()
        return lo, hi
    (a0, a1), (b0, b1) = span(a), span(b)
    return a0 < b1 and b0 < a1


def aliased(out: torch.Tensor, *inputs) -> bool:
    """True if `out` may share memory with any of the (non-None) inputs."""
    return any(t is not None and _overlaps(out, t) for t in inputs)


class _State:
    def __init__(self, path: str):
        self.path = path
        self.cache: dict[str, dict] | None = None
        self.new: dict[str, dict] = {}   # entries tuned by this process (merged into the file)


class Tuner:
    def __init__(self, path: str | None = None):
        self._path = os.path.abspath(path) if path else None
        self.lock = threading.Lock()
        self.frozen = False  # set while capturing graphs: a miss is then a bug
        self._states: dict[str, _State] = {}
        self._pinned_owner: str | None = None  # target owning a pinned (non-templated) path

    # ------------------------------------------------------------- per-target state
    def path_for(self, device=None) -> str:
        """Cache file of the device's target. A pinned path (constructor argument or
        CLARUS_TUNE_CACHE without `{target}`) belongs to the first target that uses it;
        other targets in the same process get a sibling file '<stem>.<target><ext>'."""
        return self._state(target_of(device)).path

    def _state(self, target: str) -> _State:
        st = self._states.get(target)
        if st is not None:
            return st
        env = os.environ.get("CLARUS_TUNE_CACHE")
        pinned = self._path or (os.path.abspath(env) if env and "{target}" not in env else None)
        if pinned is None:
            path = (os.path.abspath(env.replace("{target}", target)) if env
                    else os.path.abspath(os.path.join(_CACHE_DIR, f"tune_{target}.json")))
        elif self._pinned_owner in (None, target):
            self._pinned_owner = target
            path = pinned
        else:
            stem, ext = os.path.splitext(pinned)
            path = f"{stem}.{target}{ext or '.json'}"
            warnings.warn(f"clarus tuner: {pinned} holds configs for {self._pinned_owner}; "
                          f"configs for {target} go to {path}")
        st = self._states[target] = _State(path)
        return st

    @property
    def path(self) -> str:
        return self.path_for(None)

    @property
    def cache(self) -> dict:
        """Entries of the current device's target (loaded on first use)."""
        return self._load(self._state(target_of(None)))

    @staticmethod
    def _read(path: str) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load(self, st: _State) -> dict:
        if st.cache is None:
            st.cache = self._read(st.path)
        return st.cache

    def _save(self, st: _State):
        """Lock, re-read, merge this process's new entries, atomically replace."""
        d = os.path.dirname(st.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(st.path + ".lock", "a+") as lk:
            if fcntl is not None:
                fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                disk = self._read(st.path)
                for k, v in disk.items():  # pick up other processes' entries
                    st.cache.setdefault(k, v)
                disk.update(st.new)
                tmp = f"{st.path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(disk, f, indent=0, sort_keys=True)
                os.replace(tmp, st.path)
            finally:
                if fcntl is not None:
                    fcntl.flock(lk, fcntl.LOCK_UN)

    # ------------------------------------------------------------------- tuning
    def pick(self, name: str, key: tuple, candidates: list[dict], launch, *, device=None,
             restore=()) -> dict:
        """Cached best config for (name, key) on `device`'s target (default: current device).

        `restore`: tensors the launch both reads and writes (e.g. `out` aliasing the
        residual). They are snapshotted before benchmarking and restored afterwards, so the
        caller's following launch sees the original values.
        """
        k = f"{name}|{','.join(map(str, key))}"
        st = self._state(target_of(device))
        cfg = self._load(st).get(k)
        if cfg is not None:
            return cfg
        if self.frozen:
            raise RuntimeError(f"untuned kernel shape during graph capture: {k}")
        if len(candidates) == 1:
            best = candidates[0]
        else:
            saved = [t.clone() for t in restore if t is not None]
            best, best_t = None, float("inf")
            try:
                for c in candidates:
                    try:
                        t = do_bench(lambda: launch(c), warmup=5, rep=25)
                    except (triton.runtime.errors.OutOfResources, RuntimeError):
                        continue
                    if t < best_t:
                        best, best_t = c, t
            finally:
                for dst, src in zip([t for t in restore if t is not None], saved):
                    dst.copy_(src)
            if best is None:
                raise RuntimeError(f"no valid config for {k}")
        with self.lock:
            st.cache[k] = best
            st.new[k] = best
            self._save(st)
        return best


TUNER = Tuner()


def cfg(bm, bn, bk, w, s, **extra):
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=s, **extra)


def p2(x: int) -> int:
    return triton.next_power_of_2(max(1, int(x)))

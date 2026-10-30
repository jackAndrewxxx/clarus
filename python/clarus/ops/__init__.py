"""Operator dispatch.

A model calls primitives on one `Ops` object (``ops.linear(...)``, ``ops.dwconv(...)``).
`Ops` holds an ordered list of backends and, per call, uses the first backend that
implements the op and accepts the arguments. A backend declines a call by raising
`Unsupported` (e.g. a kernel only written for 3x3 windows), and the next backend runs it.
The last backend should be the PyTorch reference (`torch`), which implements every op.

Contract shared by all backends (so they can be mixed op by op):
  * activations are NHWC, contiguous; the compute dtype is `Ops.dtype`
  * weights are passed in the *canonical* layout produced by the model's weight folding
    (dense convs [Cout, Cin, kh, kw], depthwise [C, 1, kh, kw], linear/1x1 as [K, N]);
    a backend that needs another layout converts lazily and caches (`Backend.cached`)
  * return types / shapes are identical (see torch_ref.py docstrings)

Adding a backend: subclass `Backend`, implement any subset of the op methods, register it
with `@register_backend("name")`, and list it in `runtime.backends` of the engine config.
"""
from __future__ import annotations

import collections
import importlib

import torch


class Unsupported(Exception):
    """Raised by a backend op to hand the call to the next backend."""


class BackendUnavailable(Exception):
    """Raised by a backend constructor when it cannot run on this device / dtype."""


_BACKENDS: dict[str, str] = {
    "torch": "clarus.ops.torch_ref:TorchOps",
    "triton": "clarus.ops.triton_cuda:TritonOps",
}


def register_backend(name: str, path: str | None = None):
    """Register a backend class, either as decorator or by 'module:Class' path."""
    if path is not None:
        _BACKENDS[name] = path
        return None

    def deco(cls):
        _BACKENDS[name] = cls
        cls.name = name
        return cls
    return deco


def _resolve(name: str):
    ref = _BACKENDS.get(name)
    if ref is None:
        raise ValueError(f"unknown op backend {name!r}; known: {sorted(_BACKENDS)}")
    if isinstance(ref, str):
        mod, cls = ref.split(":")
        ref = getattr(importlib.import_module(mod), cls)
    return ref


class Backend:
    """Base class. `dtype` is the activation dtype, `device` a torch.device."""

    name = "base"

    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        self._cache: dict = {}

    def cached(self, t: torch.Tensor, tag: str, fn):
        """Per-backend weight re-layout, computed once per (tensor, tag)."""
        key = (t.data_ptr(), tuple(t.shape), tag)
        v = self._cache.get(key)
        if v is None:
            v = self._cache[key] = fn(t)
        return v


class Ops:
    def __init__(self, backends: list[str], dtype: torch.dtype, device: torch.device,
                 overrides: dict[str, str] | None = None, log=print):
        self.dtype = dtype
        self.device = device
        self.backends: list[Backend] = []
        self.skipped: dict[str, str] = {}
        for name in backends:
            try:
                self.backends.append(_resolve(name)(dtype, device))
            except BackendUnavailable as e:
                self.skipped[name] = str(e)
                log(f"[clarus] op backend {name!r} unavailable: {e}")
        if not self.backends:
            raise RuntimeError(f"no usable op backend among {backends}: {self.skipped}")
        self.by_name = {b.name: b for b in self.backends}
        self.overrides = dict(overrides or {})
        for op, b in self.overrides.items():
            if b not in self.by_name:
                raise ValueError(f"op_overrides: {op} -> {b!r}, but backend {b!r} is not active")
        self.served = collections.Counter()  # (op, backend) -> eager calls

    def _make(self, op: str):
        chain = [self.by_name[self.overrides[op]]] if op in self.overrides else self.backends
        impls = [(b.name, getattr(b, op)) for b in chain if callable(getattr(b, op, None))]
        if not impls:
            raise AttributeError(f"no backend implements op {op!r}")
        served = self.served

        def call(*args, **kwargs):
            for i, (bname, fn) in enumerate(impls):
                try:
                    out = fn(*args, **kwargs)
                except Unsupported:
                    if i == len(impls) - 1:
                        raise
                    continue
                served[(op, bname)] += 1
                return out
        call.__name__ = op
        return call

    def __getattr__(self, op: str):
        if op.startswith("_"):
            raise AttributeError(op)
        fn = self._make(op)
        setattr(self, op, fn)  # resolve the chain once per op
        return fn

    def report(self) -> dict:
        """{op: {backend: calls}} for the eager calls seen so far."""
        out: dict = collections.defaultdict(dict)
        for (op, b), n in sorted(self.served.items()):
            out[op][b] = n
        return {"backends": [b.name for b in self.backends], "skipped": self.skipped,
                "overrides": self.overrides, "ops": dict(out)}

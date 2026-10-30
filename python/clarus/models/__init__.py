"""Model registry and the interface every served network implements.

A model is a Python class registered under a name; the engine config selects it with
`model.name` and passes the whole `model:` section to its constructor. External models do
not have to live in this package: list their modules in `model.plugins` and they are
imported (and so registered) before the model is built.

    @register_model("my_net")
    class MyNet(RestorationModel):
        def __init__(self, cfg, ops, device, dtype): ...
        def forward(self, x): ...          # [N, C_in, H, W] fp32 in [0, 1] -> [N, C_out, sH, sW] fp32

See models/template.py for a complete, commented skeleton.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import torch

_MODELS: dict[str, type] = {}


@dataclass
class IOSpec:
    """What the engine needs to know to feed the model and check its outputs."""

    in_channels: int = 3       # the RGB8 IO path supports 3
    out_channels: int = 3
    scale: int = 1             # output H, W = input H, W * scale (super-resolution > 1)
    pad_multiple: int = 1      # informational: the model pads internally to this multiple


class RestorationModel(ABC):
    """Base class of served networks.

    `forward` is called once eagerly per new (batch, H, W) to compile / tune kernels and then
    captured into a device graph when the device supports it, so it must be capture-safe:
    no host synchronization (`.item()`, `.cpu()`, `print(tensor)`), no data-dependent Python
    control flow, and fixed output shapes for a fixed input shape.
    """

    name: str = "?"
    io: IOSpec = IOSpec()

    def __init__(self, cfg: dict, ops, device: torch.device, dtype: torch.dtype):
        self.cfg, self.ops, self.device, self.dtype = cfg, ops, device, dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def describe(self) -> dict:
        return {"name": self.name, "io": asdict(self.io)}


def register_model(name: str, *aliases: str):
    def deco(cls):
        for n in (name, *aliases):
            if n in _MODELS and _MODELS[n] is not cls:
                raise ValueError(f"model name {n!r} registered twice")
            _MODELS[n] = cls
        cls.name = name
        return cls
    return deco


def load_plugins(modules) -> None:
    for m in modules or ():
        importlib.import_module(m)


def available_models() -> list[str]:
    _builtin()
    return sorted(_MODELS)


def _builtin():
    from . import nafnet  # noqa: F401
    from . import restormer  # noqa: F401
    from . import rrdbnet  # noqa: F401
    from . import fftformer  # noqa: F401
    from . import evssm  # noqa: F401


def create_model(cfg: dict, ops, device, dtype) -> RestorationModel:
    _builtin()
    load_plugins(cfg.get("plugins"))
    name = cfg.get("name")
    if name not in _MODELS:
        raise ValueError(f"unknown model.name {name!r}; registered: {sorted(_MODELS)} "
                         f"(external models: add their module to model.plugins)")
    return _MODELS[name](cfg, ops, device, dtype)

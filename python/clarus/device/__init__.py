"""Device abstraction used by the model worker.

The worker needs: a compute stream, events to report completion, page-locked host memory
for asynchronous copies, and (optionally) graph capture / replay. `Device` wraps these for
one torch device type; `get_device("cuda:0" | "npu:0" | "cpu" | ...)` picks the class.

  * CudaDevice      - CUDA (and ROCm builds of torch): streams, pinned memory, CUDA graphs.
  * AcceleratorDevice - any other torch accelerator exposing the usual `torch.<type>` module
                      API (Stream, Event, synchronize, set_device), e.g. torch_npu's
                      `torch.npu`. Graph capture is used only if the config enables it and
                      the module exposes a graph class (`graph_class` below).
  * CpuDevice       - synchronous, no streams / graphs; useful for porting and CI.

Porting to another accelerator usually means: make sure its torch plugin is imported
(`runtime.device_imports: [torch_npu]`), then, if the generic class is not enough,
subclass `AcceleratorDevice` and `register_device("<type>", cls)`.
"""
from __future__ import annotations

import contextlib
import importlib

import torch


class _DoneEvent:
    def query(self):
        return True

    def synchronize(self):
        pass


class Device:
    type = "?"
    supports_graphs = False

    def __init__(self, spec: str):
        self.torch_device = torch.device(spec)
        self.index = self.torch_device.index or 0

    # -- identity
    def name(self) -> str:
        return str(self.torch_device)

    def set_current(self):
        pass

    # -- execution
    def new_stream(self):
        return None

    def stream_ctx(self, stream):
        return contextlib.nullcontext()

    def record_event(self, stream):
        return _DoneEvent()

    def synchronize(self, stream=None):
        pass

    # -- memory
    def host_buffer(self, shape, dtype=torch.uint8) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)

    def max_memory_gib(self) -> float | None:
        return None

    # -- graphs
    def graph_pool(self):
        return None

    def capture(self, fn, stream, pool):
        """Capture `fn()` and return a zero-argument replay callable."""
        raise NotImplementedError


class CudaDevice(Device):
    type = "cuda"
    supports_graphs = True

    def name(self):
        return torch.cuda.get_device_name(self.torch_device)

    def set_current(self):
        torch.cuda.set_device(self.torch_device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def new_stream(self):
        return torch.cuda.Stream(self.torch_device)

    def stream_ctx(self, stream):
        return torch.cuda.stream(stream)

    def record_event(self, stream):
        ev = torch.cuda.Event()
        ev.record(stream)
        return ev

    def synchronize(self, stream=None):
        (stream.synchronize if stream is not None else torch.cuda.synchronize)()

    def host_buffer(self, shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    def max_memory_gib(self):
        return torch.cuda.max_memory_allocated(self.torch_device) / 2 ** 30

    def graph_pool(self):
        return torch.cuda.graph_pool_handle()

    def capture(self, fn, stream, pool):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, stream=stream):
            fn()
        return g.replay


class AcceleratorDevice(Device):
    """Generic `torch.<type>` accelerator (e.g. torch_npu). Graphs off unless enabled."""

    graph_class_names = ("CUDAGraph", "NPUGraph", "XPUGraph")

    def __init__(self, spec):
        super().__init__(spec)
        self.type = self.torch_device.type
        self.mod = getattr(torch, self.type, None)
        if self.mod is None:
            raise RuntimeError(f"torch.{self.type} is not available; import the device plugin first "
                               f"(runtime.device_imports)")
        self.graph_cls = next((getattr(self.mod, n) for n in self.graph_class_names
                               if hasattr(self.mod, n)), None)
        self.supports_graphs = self.graph_cls is not None and hasattr(self.mod, "graph")

    def name(self):
        fn = getattr(self.mod, "get_device_name", None)
        return fn(self.index) if fn else str(self.torch_device)

    def set_current(self):
        self.mod.set_device(self.torch_device)

    def new_stream(self):
        return self.mod.Stream(self.torch_device) if hasattr(self.mod, "Stream") else None

    def stream_ctx(self, stream):
        return self.mod.stream(stream) if stream is not None else contextlib.nullcontext()

    def record_event(self, stream):
        if not hasattr(self.mod, "Event"):
            self.synchronize(stream)
            return _DoneEvent()
        ev = self.mod.Event()
        ev.record(stream)
        return ev

    def synchronize(self, stream=None):
        (stream.synchronize if stream is not None else self.mod.synchronize)()

    def host_buffer(self, shape, dtype=torch.uint8):
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, dtype=dtype)

    def max_memory_gib(self):
        fn = getattr(self.mod, "max_memory_allocated", None)
        return fn(self.torch_device) / 2 ** 30 if fn else None

    def graph_pool(self):
        fn = getattr(self.mod, "graph_pool_handle", None)
        return fn() if fn else None

    def capture(self, fn, stream, pool):
        g = self.graph_cls()
        with self.mod.graph(g, pool=pool, stream=stream):
            fn()
        return g.replay


class CpuDevice(Device):
    type = "cpu"

    def name(self):
        return "cpu"


_DEVICES: dict[str, type] = {"cuda": CudaDevice, "cpu": CpuDevice}


def register_device(type_name: str, cls: type) -> None:
    _DEVICES[type_name] = cls


def get_device(spec: str, imports=()) -> Device:
    for m in imports or ():
        importlib.import_module(m)
    kind = torch.device(spec).type
    return _DEVICES.get(kind, AcceleratorDevice)(spec)

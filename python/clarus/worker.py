"""Model worker: the Python side of the engine.

The Rust engine embeds CPython (PyO3) and drives exactly one `ModelWorker` from its
scheduler thread. Every call takes / returns one JSON string. Pixel data never crosses
the boundary as Python objects: the worker allocates (page-locked, when the device allows)
host "slots", reports their addresses in `info()`, Rust IO threads decode PNGs straight
into them, and the worker copies them to the device asynchronously.

Per batch, everything is enqueued on one device stream and, when the device supports it,
replayed from one captured graph per (batch, H, W, metrics):
  H2D(u8 HWC) -> u8 -> fp32 CHW (+ zero pad) -> model.forward -> clamp/round to u8
  -> [PSNR, SSIM] -> D2H(u8 HWC, metrics) -> event

Config (JSON from Rust): {"model": {...}, "runtime": {...}, "worker": {"slots", "max_pixels",
"batch_sizes", "inflight"}}. See configs/*.yaml for the documented schema.
"""
from __future__ import annotations

import collections
import json
import math
import os
import time

import torch

from .metrics import make_metrics
from .runtime import build, runtime_config

try:
    from .kernels.tuning import TUNER
except Exception:  # pragma: no cover - no triton on this platform
    TUNER = None


def _ceil(v: int, m: int) -> int:
    return (v + m - 1) // m * m


class ModelWorker:
    def __init__(self, cfg_json: str):
        cfg = json.loads(cfg_json)
        self.cfg = cfg
        t0 = time.perf_counter()
        self.rt = runtime_config(cfg.get("runtime"))
        self.model, self.ops, self.device = build(cfg["model"], self.rt)
        self.dev = self.device.torch_device
        self.io = self.model.io
        if self.io.in_channels != 3 or self.io.out_channels != 3:
            raise ValueError("the RGB8 IO path needs a 3-channel in / out model")
        wk = cfg.get("worker", {})
        self.batch_sizes = sorted({int(b) for b in wk.get("batch_sizes", [1, 2])})
        self.max_batch = self.batch_sizes[-1]
        g = self.rt["graphs"]
        self.use_graph = (self.device.supports_graphs if g == "auto" else bool(g)) and self.device.supports_graphs
        if g is True and not self.device.supports_graphs:
            print(f"[clarus] runtime.graphs=true but {self.device.name()} cannot capture graphs; running eagerly")
        # the eager pass before capture compiles / tunes kernels; pointless for a pure-torch eager run
        self.warm_run = self.use_graph or any(b.name != "torch" for b in self.ops.backends)
        self.stream = self.device.new_stream()
        self.metrics = make_metrics(self.dev, self.rt["metrics_backend"])
        # host slots: [lq (H*W*3) | gt (sH*sW*3) | out (sH*sW*3)] per slot
        s = self.io.scale
        n_slots = int(wk["slots"])
        self.cap_in = int(wk["max_pixels"]) * 3
        self.cap_out = self.cap_in * s * s
        self.slot_bytes = _ceil(self.cap_in + 2 * self.cap_out, 4096)
        self.host = self.device.host_buffer((n_slots, self.slot_bytes))
        self.graphs: collections.OrderedDict = collections.OrderedDict()
        self.max_graphs = int(self.rt["max_graphs"])  # LRU size; must cover every shape in a run
        self.pool = self.device.graph_pool() if self.use_graph else None
        self.pending: collections.deque = collections.deque()
        self.res_ring = [self.device.host_buffer((self.max_batch, 3), torch.float64)
                         for _ in range(int(wk.get("inflight", 4)) + 2)]
        self.ring_pos = 0
        self.load_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ setup
    def _plane(self, slot: int, which: str, n: int) -> torch.Tensor:
        off = {"lq": 0, "gt": self.cap_in, "out": self.cap_in + self.cap_out}[which]
        return self.host[slot, off:off + n]

    def info(self, _arg: str = "") -> str:
        st = self.host.stride(0)
        return json.dumps({
            "slot_base": self.host.data_ptr(), "slot_stride": st,
            "planes": {"lq": [0, self.cap_in], "gt": [self.cap_in, self.cap_out],
                       "out": [self.cap_in + self.cap_out, self.cap_out]},
            "slots": self.host.shape[0], "scale": self.io.scale, "batch_sizes": self.batch_sizes,
            "device": self.device.name(), "graphs": self.use_graph, "model": self.model.describe(),
            "ops": self.ops.report(), "load_s": round(self.load_s, 3),
        })

    def _pipeline(self, e):
        """The whole per-batch device pipeline (captured into one graph when supported)."""
        h, w, s = e["h"], e["w"], self.io.scale
        inp = e["inp"]
        # same arithmetic as basicsr: float32(u8) / 255; zero padding to the model's multiple
        inp[:, :, :h, :w].copy_(e["u8_in"].permute(0, 3, 1, 2)).div_(255.0)
        y = self.model.forward(inp)[:, :, :h * s, :w * s]
        # tensor2img: clamp(0, 1) -> * 255 -> round (half to even) -> uint8
        e["u8_out"].copy_(y.clamp(0.0, 1.0).mul(255.0).round_().permute(0, 2, 3, 1))
        res = e["res"]
        res[:, 2].copy_((~torch.isfinite(y)).flatten(1).sum(1))
        if e["metrics"]:
            for i in range(e["b"]):
                self.metrics.compute(e["u8_out"][i], e["u8_gt"][i], res[i])

    def _entry(self, b: int, h: int, w: int, metrics: bool):
        key = (b, h, w, metrics)
        e = self.graphs.get(key)
        if e is not None:
            self.graphs.move_to_end(key)
            return e
        t0 = time.perf_counter()
        m, s, d = max(1, self.io.pad_multiple), self.io.scale, self.dev
        e = {"b": b, "h": h, "w": w, "metrics": metrics, "graph": None,
             "inp": torch.zeros((b, 3, _ceil(h, m), _ceil(w, m)), device=d, dtype=torch.float32),
             "u8_in": torch.zeros((b, h, w, 3), device=d, dtype=torch.uint8),
             "u8_gt": torch.zeros((b, h * s, w * s, 3), device=d, dtype=torch.uint8),
             "u8_out": torch.empty((b, h * s, w * s, 3), device=d, dtype=torch.uint8),
             "res": torch.zeros((b, 3), device=d, dtype=torch.float64)}
        with self.device.stream_ctx(self.stream):
            if self.warm_run:
                self._pipeline(e)  # eager: compiles / tunes every kernel for this shape
            if self.use_graph:
                self._pipeline(e)
                self.device.synchronize(self.stream)
                if TUNER is not None:
                    TUNER.frozen = True
                try:
                    e["graph"] = self.device.capture(lambda: self._pipeline(e), self.stream, self.pool)
                finally:
                    if TUNER is not None:
                        TUNER.frozen = False
        self.device.synchronize(self.stream)
        self.graphs[key] = e
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        e["prepare_s"] = time.perf_counter() - t0
        return e

    def warmup(self, req_json: str) -> str:
        """req: {"shapes": [[h, w], ...], "metrics": bool} -> prepares every (batch size, shape)."""
        req = json.loads(req_json)
        out = []
        for h, w in req["shapes"]:
            for b in self.batch_sizes:
                e = self._entry(b, int(h), int(w), bool(req.get("metrics", True)))
                out.append({"batch": b, "h": h, "w": w, "prepare_s": round(e["prepare_s"], 3)})
        return json.dumps(out)

    # -------------------------------------------------------------- execution
    def submit(self, req_json: str) -> str:
        """req: {"id": int, "h": h, "w": w, "slots": [s, ...], "metrics": bool, "output": bool}.

        All images of one batch share (h, w); the scheduler buckets requests by shape.
        Returns immediately; completion is reported by poll() / wait().
        """
        req = json.loads(req_json)
        slots, h, w = req["slots"], int(req["h"]), int(req["w"])
        want_m, want_o = bool(req.get("metrics")), bool(req.get("output"))
        b = len(slots)
        e = self._entry(b, h, w, want_m)
        s = self.io.scale
        n_in, n_out = h * w * 3, h * w * 3 * s * s
        t_submit = time.perf_counter()
        with self.device.stream_ctx(self.stream):
            for i, sl in enumerate(slots):
                e["u8_in"][i].view(-1).copy_(self._plane(sl, "lq", n_in), non_blocking=True)
                if want_m:
                    e["u8_gt"][i].view(-1).copy_(self._plane(sl, "gt", n_out), non_blocking=True)
            if e["graph"] is not None:
                e["graph"]()
            else:
                self._pipeline(e)
            if want_o:
                for i, sl in enumerate(slots):
                    self._plane(sl, "out", n_out).copy_(e["u8_out"][i].view(-1), non_blocking=True)
            res_host = self.res_ring[self.ring_pos % len(self.res_ring)][:b]
            self.ring_pos += 1
            res_host.copy_(e["res"], non_blocking=True)
            ev = self.device.record_event(self.stream)
        self.pending.append((req["id"], ev, res_host, slots, t_submit))
        return "ok"

    def _finish(self, entry) -> dict:
        bid, ev, res_host, slots, t_submit = entry
        ev.synchronize()
        vals = res_host.tolist()

        def fin(x):  # JSON has no inf / nan
            return x if math.isfinite(x) else None
        return {"id": bid, "results": [{"slot": sl, "psnr": fin(v[0]), "ssim": fin(v[1]), "nonfinite": int(v[2])}
                                       for sl, v in zip(slots, vals)],
                "latency_ms": round((time.perf_counter() - t_submit) * 1000, 3)}

    def wait(self, _arg: str = "") -> str:
        """Block until the oldest in-flight batch completes; return its results."""
        if not self.pending:
            return "null"
        return json.dumps(self._finish(self.pending.popleft()))

    def poll(self, _arg: str = "") -> str:
        done = []
        while self.pending and self.pending[0][1].query():
            done.append(self._finish(self.pending.popleft()))
        return json.dumps(done)

    def inflight(self, _arg: str = "") -> str:
        return str(len(self.pending))

    def shutdown(self, _arg: str = "") -> str:
        while self.pending:
            self._finish(self.pending.popleft())
        self.device.synchronize()
        mem = self.device.max_memory_gib()
        return json.dumps({"peak_mem_gib": None if mem is None else round(mem, 3),
                           "graphs": [list(k) for k in self.graphs], "ops": self.ops.report()})


def create(cfg_json: str) -> ModelWorker:
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    return ModelWorker(cfg_json)

"""Fast self-tests of the validation tools (CPU, synthetic files under /tmp).

  python tools/test_tools.py [-v]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common  # noqa: E402
import check_zoo_variants  # noqa: E402
import compare_reports  # noqa: E402
import debug_diverge  # noqa: E402
import parity  # noqa: E402
import score_outputs  # noqa: E402
from clarus.ops import Ops  # noqa: E402

PY = sys.executable


def run_tool(script, *args, env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run([PY, os.path.join(HERE, script), *args], capture_output=True, text=True, env=e)


def rows(*names):
    return [{"name": n, "psnr": 30.0 + i, "ssim": 0.9} for i, n in enumerate(names)]


class CompareReports(unittest.TestCase):
    def test_identical_sets(self):
        names, problems, excluded = compare_reports.match_names(rows("a", "b"), rows("b", "a"))
        self.assertEqual((names, problems, excluded), (["a", "b"], [], []))

    def test_missing_extra_duplicates(self):
        _, problems, _ = compare_reports.match_names(rows("a", "b", "b"), rows("a", "c"))
        text = "\n".join(problems)
        self.assertIn("duplicate names in engine report: ['b']", text)
        self.assertIn("missing from the engine report: ['c']", text)
        self.assertIn("not in the reference report: ['b']", text)

    def test_subset_lists_excluded(self):
        names, problems, excluded = compare_reports.match_names(rows("a", "b"), rows("a", "c"), subset=True)
        self.assertEqual((names, problems), (["a"], []))
        self.assertEqual(len(excluded), 2)
        _, problems, _ = compare_reports.match_names(rows("a", "a"), rows("a"), subset=True)
        self.assertTrue(problems)                     # duplicates stay an error with --subset
        _, problems, _ = compare_reports.match_names(rows("a"), rows("b"), subset=True)
        self.assertIn("no images in common", problems)

    def test_cli_exit_codes(self):
        d = tempfile.mkdtemp(prefix="clarus_tt_cmp_")
        e, r = os.path.join(d, "e.json"), os.path.join(d, "r.json")
        for path, names in ((e, ("a", "b")), (r, ("a", "b", "c"))):
            with open(path, "w") as f:
                json.dump({"images": rows(*names)}, f)
        p = run_tool("compare_reports.py", e, r)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("['c']", p.stderr)
        p = run_tool("compare_reports.py", e, r, "--subset")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("excluded (--subset)", p.stdout)
        self.assertIn("images compared      2", p.stdout)
        p = run_tool("compare_reports.py", e, e)
        self.assertEqual(p.returncode, 0, p.stderr)


class ScoreOutputs(unittest.TestCase):
    def setUp(self):
        import cv2
        self.d = tempfile.mkdtemp(prefix="clarus_tt_score_")
        self.out, self.gt, self.other = (os.path.join(self.d, s) for s in ("out", "gt", "other"))
        img = np.zeros((8, 8, 3), np.uint8)
        for sub, names in ((self.gt, "abc"), (self.out, "ab"), (self.other, "abc")):
            os.makedirs(sub)
            for n in names:
                cv2.imwrite(os.path.join(sub, f"{n}.png"), img)
        cv2.imwrite(os.path.join(self.out, "zzz.png"), img)          # output without GT

    def test_match(self):
        pairs, problems, _ = score_outputs.match_files(self.out, self.gt)
        text = "\n".join(problems)
        self.assertIn("without output", text)
        self.assertIn("['c.png']", text)
        self.assertIn("['zzz.png']", text)
        pairs, problems, excluded = score_outputs.match_files(self.out, self.gt, subset=True)
        self.assertEqual((pairs, problems), ([("a.png", "a.png"), ("b.png", "b.png")], []))
        self.assertEqual(len(excluded), 2)

    def test_other_and_suffix(self):
        os.remove(os.path.join(self.other, "b.png"))
        for n in "abc":
            os.rename(os.path.join(self.out, f"{n}.png") if n != "c" else os.path.join(self.out, "zzz.png"),
                      os.path.join(self.out, f"{n}_out.png"))
        pairs, problems, _ = score_outputs.match_files(self.out, self.gt, "_out", self.other)
        self.assertEqual(len(problems), 1)
        self.assertIn(f"missing from {self.other}: ['b.png']", problems[0])
        pairs, problems, _ = score_outputs.match_files(self.out, self.gt, "_out", self.other, subset=True)
        self.assertEqual([p[0] for p in pairs], ["a.png", "c.png"])

    def test_duplicate_stems(self):
        import cv2
        cv2.imwrite(os.path.join(self.gt, "a.jpg"), np.zeros((8, 8, 3), np.uint8))
        _, problems, _ = score_outputs.match_files(self.out, self.gt, subset=True)
        self.assertIn("['a']", "\n".join(problems))

    def test_cli_fails_before_scoring(self):
        p = run_tool("score_outputs.py", self.out, self.gt)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("c.png", p.stderr)
        self.assertNotIn('"psnr"', p.stdout)


class DebugDiverge(unittest.TestCase):
    def test_snapshot_survives_buffer_reuse(self):
        class Net:
            ops = Ops(["torch"], torch.float32, torch.device("cpu"), log=lambda *_: None)
        log = []
        debug_diverge.record(Net, log)
        a, buf = torch.ones(1, 2, 2, 4), torch.empty(1, 2, 2, 4)
        Net.ops.add_scaled(a, a, 1.0, out=buf)                # buf = 2
        Net.ops.add_scaled(a, a, 2.0, out=buf)                # buf = 3, same buffer
        self.assertEqual([n for n, _ in log], ["add_scaled", "add_scaled"])
        self.assertEqual(log[0][1].unique().tolist(), [2.0])
        self.assertEqual(log[1][1].unique().tolist(), [3.0])

    def test_sequence_checks(self):
        t = torch.ones(3)
        quiet = []
        self.assertEqual(debug_diverge.compare([("x", t), ("y", t)], [("x", t), ("y", t)], quiet.append), 0)
        out = []
        self.assertEqual(debug_diverge.compare([("x", t), ("y", t), ("z", t)], [("x", t), ("w", t)], out.append), 1)
        self.assertIn("MISMATCH at #1: a=y b=w", out[-1])
        self.assertEqual(sum(" x " in o for o in out), 1)   # only the matching prefix is compared
        out = []
        self.assertEqual(debug_diverge.compare([("x", t)], [("x", t), ("y", t)], out.append), 1)
        self.assertIn("MISMATCH at #1: a=<end> b=y", out[-1])


class Parity(unittest.TestCase):
    def test_sampling_uses_actual_count(self):
        names = [f"{i:03d}.png" for i in range(12)]
        self.assertEqual(parity.sample_names(names[::-1], 3), ["000.png", "004.png", "008.png"])
        self.assertEqual(parity.sample_names(names, 12), names)
        big = [f"{i:04d}.png" for i in range(1111)]
        self.assertEqual(parity.sample_names(big, 3), ["0000.png", "0370.png", "0740.png"])
        with self.assertRaises(SystemExit):
            parity.sample_names(names, 13)

    def test_psnr_is_the_worker_rule(self):
        from clarus import metrics
        self.assertIs(parity.psnr, metrics.psnr)
        self.assertAlmostEqual(parity.psnr(torch.zeros(4, 4, 3), torch.full((4, 4, 3), 2.0)), -6.0206, places=3)


class ReferenceWrappers(unittest.TestCase):
    """Padding / cropping of the official test scripts, with a shape-recording identity network."""

    def setUp(self):
        self.seen = []

        def net(x):
            self.seen.append(tuple(x.shape[2:]))
            return x
        self.net = net

    def test_restormer_fftformer_pad_and_crop(self):
        x = torch.rand(1, 3, 719, 1279)
        self.assertTrue(torch.equal(_common._fwd_restormer(self.net, x), x))
        self.assertTrue(torch.equal(_common._fwd_fftformer(self.net, x), x))
        self.assertEqual(self.seen, [(720, 1280), (736, 1280)])
        _common._fwd_fftformer(self.net, torch.rand(1, 3, 720, 1280))
        self.assertEqual(self.seen[-1], (736, 1280))      # 720 is not a multiple of 32

    def test_evssm_official_path(self):
        y = _common._fwd_evssm(self.net, torch.rand(2, 3, 64, 96))
        self.assertEqual((y.shape[0], self.seen), (2, [(64, 96), (64, 96)]))   # batch 1 at a time
        with self.assertRaises(ValueError):
            _common._fwd_evssm(self.net, torch.rand(1, 3, 721, 1280))

    def test_rrdbnet_mod_pad(self):
        def up2(x):
            self.seen.append(tuple(x.shape[2:]))
            return torch.nn.functional.interpolate(x, scale_factor=2)
        y = _common._fwd_rrdbnet(up2, torch.rand(1, 3, 35, 51), 2)
        self.assertEqual((self.seen, tuple(y.shape[2:])), ([(36, 52)], (70, 102)))


class ZooVariants(unittest.TestCase):
    def test_no_references_is_failure(self):
        env = {k: "" for k in ("CLARUS_NAFNET_REPO", "CLARUS_RESTORMER_REPO", "CLARUS_BASICSR_PIP",
                               "CLARUS_FFTFORMER_REPO", "CLARUS_EVSSM_REPO")}
        p = run_tool("check_zoo_variants.py", "--device", "cpu", env=env)
        self.assertNotEqual(p.returncode, 0, p.stdout)
        self.assertIn("0 passed, 0 failed, 22 skipped", p.stdout)
        self.assertNotIn("OK", p.stdout.splitlines()[-1])

    def test_triton_case_without_triton_ops_fails(self):
        class Model:
            def forward(self, x):
                self.ops.add_scaled(x, x, 0.0)
                return x
        ops = Ops(["torch"], torch.float32, torch.device("cpu"), log=lambda *_: None)
        ops.skipped["triton"] = "not available on cpu"
        m = Model()
        m.ops = ops
        orig = check_zoo_variants.build_from_config
        check_zoo_variants.build_from_config = lambda cfg, **kw: (m, ops, None)
        try:
            x = torch.rand(1, 4, 4, 3)
            ok, msg = check_zoo_variants.run_case("triton,torch/fp16", {}, "triton", 2e-2, "", x, x)
            self.assertFalse(ok)
            self.assertIn("NO OP RAN ON triton (not available on cpu)", msg)
            ok, msg = check_zoo_variants.run_case("torch/fp32", {}, "torch", 1e-4, "", x, x)
            self.assertTrue(ok, msg)
            self.assertIn("torch: 1 ops", msg)
        finally:
            check_zoo_variants.build_from_config = orig


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Integration test for the ScrapGpu ORT plugin EP (runs inference in a subprocess)."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ep_lib() -> Path:
    for sub in ("_w/bt/adapters/onnxruntime", "_w/bort/adapters/onnxruntime"):
        p = _repo_root() / sub / "libscrapgpu_ort_ep.dylib"
        if p.is_file():
            return p
    raise FileNotFoundError("build libscrapgpu_ort_ep.dylib with SCRAPGPU_BUILD_ORT_EP=ON")


_RUNNER = r"""
import ctypes
import os, sys
import numpy as np

np.random.seed(0)
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

lib = sys.argv[1]
ort.register_execution_provider_library("ScrapGpuExecutionProvider", lib)
devices = [d for d in ort.get_ep_devices() if d.ep_name == "ScrapGpuExecutionProvider"]
if not devices:
    print("SKIP:no_devices")
    sys.exit(0)

k, m, n, n2 = 32, 4, 16, 8
w0 = numpy_helper.from_array(np.random.randn(k, n).astype(np.float32), name="W0")
w1 = numpy_helper.from_array(np.random.randn(n, n2).astype(np.float32), name="W1")
x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])
y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n2])
graph = helper.make_graph(
    [
        helper.make_node("MatMul", ["X", "W0"], ["T0"]),
        helper.make_node("Add", ["T0", "T0"], ["T1"]),
        helper.make_node("MatMul", ["T1", "W1"], ["Y"]),
    ],
    "scrapgpu_ep_test",
    [x_info],
    [y_info],
    [w0, w1],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10  # onnx 1.23 writes IR 14 by default; ONNX Runtime 1.31 reads up to 13
onnx.checker.check_model(model)
x = np.random.randn(m, k).astype(np.float32)

cpu = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
y_ref = cpu.run(None, {"X": x})[0]

so = ort.SessionOptions()
so.add_session_config_entry("ep.scrapgpu.weight_type", os.environ.get("SCRAPGPU_ORT_WEIGHT_TYPE", "F32"))
so.add_provider_for_devices(devices[:1], {})
ep = ort.InferenceSession(model.SerializeToString(), so)
y_ep = ep.run(None, {"X": x})[0]

def _claimed_nodes() -> int:
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    raw = libc.getenv(b"SCRAPGPU_ORT_EP_CLAIMED_NODES")
    return int(raw.decode()) if raw else 0

claimed = _claimed_nodes()
diff = y_ep - y_ref
nmse = float(np.sum(diff * diff) / max(float(np.sum(y_ref * y_ref)), 1e-30))
print(f"OK:claimed={claimed}:nmse={nmse}:maxabs={float(np.max(np.abs(diff)))}")
"""


class TestScrapGpuEp(unittest.TestCase):
    def _run_child(self, weight_type: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["SCRAPGPU_ORT_WEIGHT_TYPE"] = weight_type
        return subprocess.run(
            [sys.executable, "-c", _RUNNER, str(_ep_lib())],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

    def test_f32_matches_cpu(self) -> None:
        proc = self._run_child("F32")
        if proc.returncode != 0:
            self.fail(f"EP subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
        if line.startswith("SKIP:"):
            self.skipTest(line[5:])
        self.assertTrue(line.startswith("OK:"), line)
        claimed = int(line.split("claimed=")[1].split(":")[0])
        nmse = float(line.split("nmse=")[1].split(":")[0])
        maxabs = float(line.split("maxabs=")[1])
        self.assertGreaterEqual(claimed, 3)
        self.assertLess(nmse, 1e-6)
        self.assertLess(maxabs, 1e-4)

    def test_quantized_weights_report(self) -> None:
        for wt in ("Q8_0", "Q4_0"):
            proc = self._run_child(wt)
            if proc.returncode != 0:
                self.fail(f"{wt} subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
            line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
            if line.startswith("SKIP:"):
                self.skipTest(line[5:])
            self.assertTrue(line.startswith("OK:"), line)
            nmse = float(line.split("nmse=")[1].split(":")[0])
            maxabs = float(line.split("maxabs=")[1])
            print(f"{wt}: max_abs={maxabs:.6g} nmse={nmse:.6g}")
            if wt == "Q8_0":
                self.assertLess(nmse, 1e-3)
            else:
                self.assertLess(nmse, 1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

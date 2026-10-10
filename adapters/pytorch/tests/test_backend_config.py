#!/usr/bin/env python3
"""Process-isolated registration/configuration checks, without any GPU requirement."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (
    "SCRAPGPU_TORCH_BACKEND", "SCRAPGPU_TORCH_DEVICE",
    "SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK", "SCRAPGPU_TORCH_MAX_STAGING_BYTES",
)


def run(code, settings=None):
    env = os.environ.copy()
    for key in SETTINGS:
        env.pop(key, None)
    env["SCRAPGPU_TORCH_BACKEND"] = "cpu"
    env["PYTHONPATH"] = str(ROOT)
    env.update(settings or {})
    result = subprocess.run([sys.executable, "-c", "import os, torch, scrapgpu_torch as sgt\n" + code], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def main():
    run("""sgt.load()
initial = torch.ops.scrapgpu.device_count()
assert initial == 1
name = torch.ops.scrapgpu.device_name(0)
for i in range(8):
    sgt.load()
    assert torch.ops.scrapgpu.device_count() == initial
os.environ['SCRAPGPU_TORCH_BACKEND'] = 'opencl11'
os.environ['SCRAPGPU_TORCH_DEVICE'] = '9999'
sgt.load()
assert torch.ops.scrapgpu.device_name(0) == name
assert torch.equal(torch.ops.scrapgpu.add(torch.ones(2), torch.ones(2)), torch.full((2,), 2.0))
""")
    invalid = [
        ({"SCRAPGPU_TORCH_BACKEND": "cuda"}, "must be cpu or opencl11"),
        ({"SCRAPGPU_TORCH_DEVICE": "-1"}, "nonnegative integer"),
        ({"SCRAPGPU_TORCH_DEVICE": "abc"}, "nonnegative integer"),
        ({"SCRAPGPU_TORCH_DEVICE": "1"}, "no ScrapGpu cpu device"),
        ({"SCRAPGPU_TORCH_DEVICE": "18446744073709551616"}, "too large"),
        ({"SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK": "2"}, "must be 0 or 1"),
        ({"SCRAPGPU_TORCH_MAX_STAGING_BYTES": "-1"}, "nonnegative integer"),
    ]
    for settings, expected in invalid:
        run(f"""try:
    sgt.load()
except RuntimeError as exc:
    assert {expected!r} in str(exc), str(exc)
else:
    raise AssertionError('invalid configuration was accepted')
""", settings)
    print("PASS: repeated registration, frozen backend/device, 7 invalid configurations")


if __name__ == "__main__":
    main()

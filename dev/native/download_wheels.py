# -*- coding: utf-8 -*-
"""download_wheels.py - fetch the onnxruntime + Pillow wheels bundled with the
extension (ARP-style: Blender installs the matching one automatically when
the addon is enabled — no pip, no internet, no install button for the user).

Only the two LEAF packages are bundled. Their PyPI-declared transitive deps
(sympy, protobuf, coloredlogs, mpmath, flatbuffers, humanfriendly, packaging)
are NOT imported at runtime by `import onnxruntime` or a real
InferenceSession().run() call - verified empirically - so they are skipped.
numpy is deliberately NOT bundled: Blender's own numpy must be the one that
loads (see the wheels= comment in blender_manifest.toml for why).

Run whenever ONNX_VERSION or PILLOW_VERSION changes:
    py -3.11 dev/native/download_wheels.py
Then update the `wheels = [...]` list in blender_manifest.toml to match the
printed filenames (they're deterministic from the versions below).
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_OUT = os.path.join(_ROOT, "wheels")

ONNX_VERSION = "1.20.1"
PILLOW_VERSION = "12.3.0"

# (python_version, abi, platform) triples to cover.
# cp311 = Blender 4.2-4.5 (Python 3.11); cp313 = Blender 5.x (Python 3.13).
_PY_TAGS = [("311", "cp311"), ("313", "cp313")]

_ONNX_PLATFORMS = ["win_amd64", "manylinux_2_28_x86_64", "macosx_13_0_universal2"]
# Pillow has no macOS universal2 wheel — arm64 and x86_64 ship separately.
_PILLOW_PLATFORMS = ["win_amd64", "manylinux_2_28_x86_64",
                     "macosx_11_0_arm64", "macosx_10_10_x86_64"]


def _download(pkg, version, pyver, abi, platform_tag):
    print(f"=== {pkg}=={version}  py{pyver}  {platform_tag} ===")
    subprocess.check_call([
        sys.executable, "-m", "pip", "download", f"{pkg}=={version}",
        "--no-deps", "--only-binary=:all:",
        "--python-version", pyver, "--implementation", "cp", "--abi", abi,
        "--platform", platform_tag,
        "-d", _OUT,
    ])


def main():
    os.makedirs(_OUT, exist_ok=True)
    for pyver, abi in _PY_TAGS:
        for plat in _ONNX_PLATFORMS:
            _download("onnxruntime", ONNX_VERSION, pyver, abi, plat)
    for pyver, abi in _PY_TAGS:
        for plat in _PILLOW_PLATFORMS:
            _download("Pillow", PILLOW_VERSION, pyver, abi, plat)

    print("\n=== downloaded ===")
    total = 0
    for f in sorted(os.listdir(_OUT)):
        if f.endswith(".whl"):
            size = os.path.getsize(os.path.join(_OUT, f))
            total += size
            print("  %7.1f MB  %s" % (size / 1024 / 1024, f))
    print("TOTAL: %.1f MB" % (total / 1024 / 1024))
    print("\nUpdate the wheels = [...] list in blender_manifest.toml to match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

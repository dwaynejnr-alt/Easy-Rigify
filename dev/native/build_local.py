# -*- coding: utf-8 -*-
"""build_local.py - compile the abi3 _ercrypto extension for THIS platform and
drop it into <addon>/native/ under a platform-tagged name the runtime loader
recognises.

    py -3.11 build_local.py

Cross-platform note: this only builds the current OS/arch. macOS + Linux
binaries come from the GitHub Actions workflow (.github/workflows/build-crypto.yml),
which runs this same setup.py on each runner. All produced binaries are abi3,
so they load in every Blender Python.
"""
import glob
import os
import platform
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # dev/native
_ROOT = os.path.dirname(os.path.dirname(_HERE))             # addon root
_NATIVE = os.path.join(_ROOT, "native")


def platform_tag():
    sysname = platform.system().lower()
    mach = platform.machine().lower()
    if sysname == "windows":
        return "win_amd64" if mach in ("amd64", "x86_64") else "win_" + mach
    if sysname == "darwin":
        # A universal2 build reports the host arch; the CI builds universal2 and
        # tags it macos_universal2. A single-arch local build is tagged by arch.
        return "macos_" + ("arm64" if mach == "arm64" else "x86_64")
    if sysname == "linux":
        return "linux_" + ("x86_64" if mach in ("x86_64", "amd64") else mach)
    return sysname + "_" + mach


def ext_for(tag):
    return ".pyd" if tag.startswith("win") else ".so"


def main():
    build_lib = os.path.join(_HERE, "_build")
    if os.path.isdir(build_lib):
        shutil.rmtree(build_lib, ignore_errors=True)
    # Build with the interpreter running THIS script.
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--build-lib", build_lib],
        cwd=_HERE,
    )
    # setuptools names an abi3 module *.abi3.pyd / *.abi3.so (or plain on some
    # toolchains). Grab whatever it produced.
    cands = (glob.glob(os.path.join(build_lib, "_ercrypto*.pyd"))
             + glob.glob(os.path.join(build_lib, "_ercrypto*.so")))
    if not cands:
        print("ERROR: no built extension found in", build_lib)
        return 1
    src = cands[0]
    tag = platform_tag()
    os.makedirs(_NATIVE, exist_ok=True)
    dst = os.path.join(_NATIVE, "_ercrypto_%s%s" % (tag, ext_for(tag)))
    shutil.copyfile(src, dst)
    print("built:", src)
    print("installed:", dst, "(%.1f KB)" % (os.path.getsize(dst) / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

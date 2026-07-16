# -*- coding: utf-8 -*-
"""model_crypto.py - load the bundled encrypted model files.

The keyed transform lives in a compiled, platform-specific binary under
native/ (_ercrypto_<platform>.<pyd|so>), built abi3 so one file per OS/arch
loads in every Blender Python. The decryption key exists ONLY inside that
binary - there is no Python-side key and no pure-Python fallback: if the
native module for the current platform is missing, model loading fails loudly
rather than silently exposing anything.

Build the binaries with dev/native/build_local.py (current OS) or the
.github/workflows/build-crypto.yml CI matrix (all OSes).
"""

import importlib.util
import os
import platform

_MAGIC = b"RAMK\x00"
_VERSION = 2
_HEADER_LEN = len(_MAGIC) + 1 + 8

_NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
_native_mod = None


def _candidate_files():
    """Filenames to try for the current platform, in priority order."""
    s = platform.system().lower()
    m = platform.machine().lower()
    if s == "windows":
        tags, ext = (["win_amd64"] if m in ("amd64", "x86_64") else ["win_" + m]), ".pyd"
    elif s == "darwin":
        arch = "arm64" if m == "arm64" else "x86_64"
        tags, ext = ["macos_universal2", "macos_" + arch], ".so"
    elif s == "linux":
        arch = "x86_64" if m in ("x86_64", "amd64") else m
        tags, ext = ["linux_" + arch, "manylinux_" + arch], ".so"
    else:
        tags, ext = [s + "_" + m], ".so"
    return ["_ercrypto_%s%s" % (t, ext) for t in tags]


def _load_native():
    global _native_mod
    if _native_mod is not None:
        return _native_mod
    tried = []
    for fname in _candidate_files():
        path = os.path.join(_NATIVE_DIR, fname)
        tried.append(fname)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_ercrypto", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _native_mod = mod
            return mod
    raise RuntimeError(
        "Easy Rigify: the model-crypto binary for this platform "
        "(%s/%s) is missing from native/. Expected one of: %s. "
        "Reinstall the addon build for your OS."
        % (platform.system(), platform.machine(), ", ".join(tried)))


def encrypt_bytes(plaintext):
    body = _load_native().transform(plaintext)
    return _MAGIC + bytes([_VERSION]) + len(plaintext).to_bytes(8, "little") + body


def decrypt_bytes(blob):
    if blob[:len(_MAGIC)] != _MAGIC:
        raise ValueError("unrecognized model file")
    ver = blob[len(_MAGIC)]
    if ver != _VERSION:
        raise ValueError("unsupported model file version: %d" % ver)
    n = int.from_bytes(blob[len(_MAGIC) + 1:_HEADER_LEN], "little")
    body = blob[_HEADER_LEN:]
    plain = _load_native().transform(body)
    if len(plain) != n:
        raise ValueError("corrupt model file")
    return plain


def load_model_bytes(path):
    """Read a model file from disk and return its decrypted bytes."""
    with open(path, "rb") as f:
        return decrypt_bytes(f.read())


def encrypt_file(src_path, dst_path):
    with open(src_path, "rb") as f:
        plain = f.read()
    blob = encrypt_bytes(plain)
    with open(dst_path, "wb") as f:
        f.write(blob)
    if decrypt_bytes(blob) != plain:
        raise RuntimeError("round-trip verification failed for %s" % dst_path)
    return len(plain), len(blob)

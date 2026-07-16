# -*- coding: utf-8 -*-
"""Validate the compiled _ercrypto against the pure-Python reference."""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import _refgen  # noqa: E402


def load_native():
    import platform
    tag = "win_amd64" if platform.system() == "Windows" else None
    cands = [f for f in os.listdir(os.path.join(_ROOT, "native"))
             if f.startswith("_ercrypto_")]
    # prefer the one for this platform
    path = None
    for f in cands:
        if tag and tag in f:
            path = os.path.join(_ROOT, "native", f)
    if path is None and cands:
        path = os.path.join(_ROOT, "native", cands[0])
    spec = importlib.util.spec_from_file_location("_ercrypto", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def main():
    mod, path = load_native()
    print("loaded:", path)
    import random
    random.seed(7)
    for n in (0, 1, 15, 31, 32, 33, 64, 100, 1000, 100003):
        data = bytes(random.randrange(256) for _ in range(n))
        c = mod.transform(data)
        r = _refgen.transform(data)
        assert c == r, f"MISMATCH at n={n}: C != reference"
        assert mod.transform(c) == data, f"round-trip failed at n={n}"
    print("PASS: C transform matches reference + symmetric round-trip (all sizes)")
    # confirm the key is NOT the old plaintext key
    import hashlib
    old_parts = [b"r1g", b"_addon", b"_pr0", b"::neural", b"::2026"]
    old_salt = bytes([0x9E,0x37,0x79,0xB9,0x4A,0xC1,0x5E,0x2D])
    old_key = hashlib.sha256(b"|".join(old_parts) + b"#" + old_salt).digest()
    # a 32-zero block transformed reveals the first keystream block = f(key)
    ks_new = mod.transform(bytes(32))
    ks_old = bytearray(32)
    blk = hashlib.sha256(old_key + (0).to_bytes(8, "little")).digest()
    ks_old[:] = blk
    assert ks_new != bytes(ks_old), "new key equals OLD key — rotation failed!"
    print("PASS: key rotated (differs from the old plaintext key)")


if __name__ == "__main__":
    main()

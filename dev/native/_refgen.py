# -*- coding: utf-8 -*-
"""Generate the masked secret table for ercrypto.c AND a pure-Python reference
transform (used only to VALIDATE the compiled binary — never shipped).

Run:  py -3.11 _refgen.py --emit-c     # prints the C array to paste
      imported by test_ercrypto.py for reference vectors
"""
import hashlib
import sys

# The one source of truth for the NEW key. Changing this rotates the key.
_SECRET = b"EasyRigify::ercrypto::v2::model-key::2026"


def _mask(i):
    return (0x5B + 7 * i) & 0xFF


def masked_table():
    return bytes((b ^ _mask(i)) & 0xFF for i, b in enumerate(_SECRET))


def key():
    return hashlib.sha256(_SECRET).digest()


def transform(data):
    """Reference: XOR data with sha256-CTR keystream (little-endian counter)."""
    k = key()
    out = bytearray(data)
    counter = 0
    off = 0
    n = len(out)
    while off < n:
        block = hashlib.sha256(k + counter.to_bytes(8, "little")).digest()
        take = min(32, n - off)
        for j in range(take):
            out[off + j] ^= block[j]
        off += take
        counter += 1
    return bytes(out)


def emit_c():
    tbl = masked_table()
    lines = []
    for i in range(0, len(tbl), 16):
        chunk = ",".join("0x%02X" % b for b in tbl[i:i+16])
        lines.append("    " + chunk + ",")
    body = "\n".join(lines).rstrip(",")
    print("static const uint8_t SECRET_MASKED[] = {")
    print(body)
    print("};")
    print("// len =", len(tbl))


if __name__ == "__main__":
    if "--emit-c" in sys.argv:
        emit_c()
    else:
        print("key =", key().hex())
        print("masked len =", len(masked_table()))

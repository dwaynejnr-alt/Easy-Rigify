# -*- coding: utf-8 -*-
"""rotate_models.py - one-shot key rotation.

Decrypts each shipped models/*.rmodel with the OLD (plaintext-Python) key and
re-encrypts it with the NEW native key (model_crypto -> _ercrypto). The exact
model bytes are preserved, so inference is unchanged; only the key changes.

Backs up the current .rmodel files to dev/native/_rmodel_backup_v1/ first.
Idempotent: files already at version 2 are decrypted with the new module and
re-written (a no-op round-trip).

    py -3.11 rotate_models.py
"""
import hashlib
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_MODELS = os.path.join(_ROOT, "models")
_BACKUP = os.path.join(_HERE, "_rmodel_backup_v1")

sys.path.insert(0, _ROOT)
import model_crypto  # NEW (native-backed)  # noqa: E402

_MAGIC = b"RAMK\x00"
_HDR = len(_MAGIC) + 1 + 8


def _old_key():
    parts = [b"r1g", b"_addon", b"_pr0", b"::neural", b"::2026"]
    salt = bytes([0x9E, 0x37, 0x79, 0xB9, 0x4A, 0xC1, 0x5E, 0x2D])
    return hashlib.sha256(b"|".join(parts) + b"#" + salt).digest()


def _old_transform(data, key):
    out = bytearray(data)
    counter, off, n = 0, 0, len(out)
    while off < n:
        blk = hashlib.sha256(key + counter.to_bytes(8, "little")).digest()
        take = min(32, n - off)
        for j in range(take):
            out[off + j] ^= blk[j]
        off += take
        counter += 1
    return bytes(out)


def _decrypt_any(blob):
    """Decrypt a v1 (old key) or v2 (new native key) blob -> plaintext."""
    if blob[:len(_MAGIC)] != _MAGIC:
        raise ValueError("unrecognized model file")
    ver = blob[len(_MAGIC)]
    n = int.from_bytes(blob[len(_MAGIC) + 1:_HDR], "little")
    body = blob[_HDR:]
    if ver == 1:
        plain = _old_transform(body, _old_key())
    elif ver == 2:
        plain = model_crypto._load_native().transform(body)
    else:
        raise ValueError("unsupported version %d" % ver)
    if len(plain) != n:
        raise ValueError("corrupt/wrong-key model file (len mismatch)")
    return plain


def main():
    files = [f for f in os.listdir(_MODELS) if f.endswith(".rmodel")]
    if not files:
        print("no .rmodel files in", _MODELS)
        return 1
    os.makedirs(_BACKUP, exist_ok=True)
    for name in sorted(files):
        path = os.path.join(_MODELS, name)
        with open(path, "rb") as f:
            blob = f.read()
        ver = blob[len(_MAGIC)]
        # back up the original once
        bpath = os.path.join(_BACKUP, name)
        if not os.path.isfile(bpath):
            shutil.copyfile(path, bpath)
        plain = _decrypt_any(blob)
        new_blob = model_crypto.encrypt_bytes(plain)
        # verify the NEW file decrypts back to the same bytes via the shipped path
        assert model_crypto.decrypt_bytes(new_blob) == plain, "round-trip failed: " + name
        with open(path, "wb") as f:
            f.write(new_blob)
        print("  rotated %-18s v%d -> v2  (%.2f MB, sha256 %s)"
              % (name, ver, len(plain) / 1024 / 1024, hashlib.sha256(plain).hexdigest()[:12]))
    print("Done. Backups in", _BACKUP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

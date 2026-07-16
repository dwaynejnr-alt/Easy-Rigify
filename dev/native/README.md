# Model-crypto native core

The decryption key for the bundled models lives ONLY inside a compiled binary
(`_ercrypto`), not in any readable Python. This raises model extraction from
"read `model_crypto.py` and copy the key" to "reverse-engineer a native
binary" (a determined attacker with a debugger can still recover it — this is a
deterrent, not DRM; see the design notes in chat).

## Files (none of this ships — `dev/` is build-excluded)

- `ercrypto.c` — the abi3 C extension: SHA-256 counter-mode keystream + XOR,
  with the key rebuilt at runtime from a **masked** byte table (so `strings`
  reveals nothing). **Contains the secret** — never ship or publish it.
- `_refgen.py` — single source of truth for the secret; emits the masked C
  table (`--emit-c`) and a pure-Python reference used only for testing.
- `setup.py` — builds the abi3 extension (`Py_LIMITED_API 0x03070000`).
- `build_local.py` — builds for the CURRENT OS and installs into `../../native/`.
- `test_ercrypto.py` — checks the binary matches the reference + round-trips.
- `rotate_models.py` — one-shot: re-encrypt `models/*.rmodel` under the new key.

## What DOES ship

Only `native/_ercrypto_<platform>.<pyd|so>` (the compiled binaries) and the
thin loader `model_crypto.py`. One abi3 binary per OS/arch loads in every
Blender Python (4.2 = 3.11 … 5.x = 3.13).

## Workflow

### Rotate the key / change the algorithm
1. Edit the secret in `_refgen.py`, run `py -3.11 _refgen.py --emit-c`, paste the
   table into `ercrypto.c` (or edit the C directly).
2. `py -3.11 build_local.py`         # rebuild THIS platform's binary
3. `py -3.11 test_ercrypto.py`       # verify
4. `py -3.11 rotate_models.py`       # re-encrypt the shipped models
5. Push → the **build-crypto** GitHub Action rebuilds macOS + Linux binaries
   and commits them into `native/`.

### Just rebuild all-platform binaries
Push a change to `ercrypto.c`, or run the **build-crypto** workflow manually
(Actions tab). It builds win/mac/linux abi3 binaries and commits them to
`native/`.

## Important
- After ANY key change you MUST re-run `rotate_models.py` or the shipped models
  won't decrypt.
- `native/` must contain a binary for every OS you ship to, or that OS fails
  loudly at model-load time (no silent fallback, by design).

# -*- coding: utf-8 -*-
"""Build the _ercrypto abi3 extension.

    python setup.py build_ext --build-lib <out>

Produces a Py_LIMITED_API (abi3) module so ONE binary per OS/arch loads in
every CPython >= 3.7 — i.e. every Blender (4.2 = 3.11 ... 5.x = 3.13). The
local wrapper (build_local.py) and the CI workflow both call this, then rename
the artifact to native/_ercrypto_<platform>.<ext>.
"""
from setuptools import setup, Extension

ext = Extension(
    "_ercrypto",
    sources=["ercrypto.c"],
    define_macros=[("Py_LIMITED_API", "0x03070000")],
    py_limited_api=True,
)

setup(
    name="_ercrypto",
    version="2.0",
    ext_modules=[ext],
    options={"bdist_wheel": {"py_limited_api": "cp37"}},
)

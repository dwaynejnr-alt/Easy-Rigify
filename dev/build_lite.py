#!/usr/bin/env python3
"""Build the Easy Rigify LITE extension zip.

Lite is the same addon with the neural side left out: no models/*.rmodel, no
onnxruntime/Pillow wheels, no model-crypto binaries. That drops the package from
~181 MB to ~2 MB. Only the geometric engines remain (Auto Detect Body, geometric
fingers, geometric face) — the addon detects this itself at import time via
constants.LITE_BUILD (models/*.rmodel absent) and hides the AI options, so there
is no flag here to keep in sync with the code.

The Python modules are NOT stripped: ai_detect_geo imports ai_detect_lvt, which
imports ai_detect at module level, so the AI modules must ship for the geometric
engine to import at all. They are inert without models — every onnxruntime import
in them is lazy.

Why a temp tree instead of editing blender_manifest.toml in place: the extension
builder requires that exact filename at the source root, and a crash mid-build
would otherwise leave the repo holding the Lite manifest and silently ship a
Lite-branded Full build.

Usage:
    python dev/build_lite.py [--blender PATH] [--output-dir PATH]

BLENDER_EXE env var is honoured (same convention as dev/regression/).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that define the Full edition — everything Lite exists to omit.
HEAVY_DIRS = {"models", "wheels", "native"}

# Never copy these into the build tree (mirrors blender_manifest.toml's
# paths_exclude_pattern; kept here too so the temp tree stays small and the
# builder never even sees them).
SKIP_DIRS = HEAVY_DIRS | {
    ".git", ".github", ".claude", "dev", "Backup", "__pycache__",
    "face_dataset", "debug_face_infer",
}

LITE_MANIFEST = '''\
schema_version = "1.0.0"

id = "easy_rigify"
version = "{version}"
name = "Easy Rigify Lite"
tagline = "Rigify markers, auto-align and smart skinning (no AI)"
maintainer = "Dwayne Jones"
type = "add-on"

tags = ["Rigging", "Animation"]

blender_version_min = "4.2.0"

license = [
  "SPDX:GPL-3.0-or-later",
]
copyright = [
  "2024-2026 Dwayne Jones",
]

# No `wheels` list: Lite ships no onnxruntime/Pillow. The addon's AI imports are
# all lazy, so the modules still import; they simply have no models to run and
# constants.LITE_BUILD hides every AI control.

[build]
paths_exclude_pattern = [
  "__pycache__/",
  "/.git/",
  "/.github/",
  "/.gitignore",
  "/*.zip",
  "/dev/",
  "/.claude/",
  "/Backup/",
  "/face_dataset/",
  "/debug_face_infer/",
  # The Lite edition: no models, no AI wheels, no model-crypto binaries.
  "/models/",
  "/wheels/",
  "/native/",
  "/*.onnx",
  "*.rmodel.bak*",
]
'''


def find_blender(explicit=None):
    if explicit:
        return explicit
    if os.environ.get("BLENDER_EXE"):
        return os.environ["BLENDER_EXE"]
    candidates = []
    for base in (r"C:\Program Files\Blender Foundation",
                 "/usr/share/blender", "/Applications"):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), reverse=True):
            for exe in ("blender.exe", "blender",
                        "Blender.app/Contents/MacOS/Blender"):
                p = os.path.join(base, entry, exe)
                if os.path.isfile(p):
                    candidates.append(p)
    if not candidates:
        sys.exit("Blender not found — pass --blender PATH or set BLENDER_EXE.")
    return candidates[0]


def read_version():
    src = os.path.join(REPO, "blender_manifest.toml")
    with open(src, encoding="utf-8") as fh:
        m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
    if not m:
        sys.exit("Could not read version from blender_manifest.toml")
    return m.group(1)


def stage(tmp):
    """Copy the addon into tmp, minus the heavy dirs, and write the Lite manifest."""
    def ignore(dirpath, names):
        drop = {n for n in names if n in SKIP_DIRS}
        drop |= {n for n in names if n.endswith((".zip", ".onnx"))
                 or ".rmodel.bak" in n}
        return drop

    shutil.copytree(REPO, tmp, ignore=ignore, dirs_exist_ok=True)

    version = read_version()
    with open(os.path.join(tmp, "blender_manifest.toml"), "w",
              encoding="utf-8") as fh:
        fh.write(LITE_MANIFEST.format(version=version))
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender")
    ap.add_argument("--output-dir", default=REPO)
    args = ap.parse_args()

    blender = find_blender(args.blender)
    tmp = os.path.join(tempfile.mkdtemp(prefix="er_lite_"), "easy_rigify")

    try:
        version = stage(tmp)

        # Guard: a stray models/ dir in the build tree would silently produce a
        # Full-behaviour addon wearing the Lite name (LITE_BUILD is computed
        # from what is on disk).
        for d in HEAVY_DIRS:
            if os.path.isdir(os.path.join(tmp, d)):
                sys.exit(f"BUG: {d}/ leaked into the Lite build tree")

        print(f"Building Easy Rigify Lite {version} ...")
        # Build into a temp dir, NOT the output dir. The builder names the zip
        # from the manifest id, and Lite deliberately shares Full's id (so the
        # two editions can never be installed side by side and collide
        # registering the same autorig.* operators) — building straight into the
        # output dir therefore OVERWRITES the Full easy_rigify-<ver>.zip sitting
        # there. Stage the artifact, then move it to the Lite name.
        stage_out = os.path.join(os.path.dirname(tmp), "out")
        os.makedirs(stage_out, exist_ok=True)
        subprocess.run(
            [blender, "--command", "extension", "build",
             "--source-dir", tmp, "--output-dir", stage_out],
            check=True,
        )

        built = os.path.join(stage_out, f"easy_rigify-{version}.zip")
        if not os.path.isfile(built):
            sys.exit(f"Expected {built} — build produced nothing")

        final = os.path.join(os.path.abspath(args.output_dir),
                             f"easy_rigify_lite-{version}.zip")
        shutil.move(built, final)
        size = os.path.getsize(final) / 1024 / 1024
        print(f"\ncreated: {final}  ({size:.1f} MB)")
    finally:
        shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)


if __name__ == "__main__":
    main()

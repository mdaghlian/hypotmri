#!/usr/bin/env python3
"""
Set up conda/mamba environments for the hypot pipeline.

Usage:
    python s00_python_setup.py [--env ENV] [--clean-env | --update] [--prefix PATH]

Environments: b14, autoflat, pctx, preproc, prf

Reads PYPACKAGE_MANAGER, PIPELINE_DIR, FSL_VERSION from the shell environment
(set by sourcing config/config_pipeline.sh).

HPC note: use --prefix /scratch/$USER/envs to install outside the default conda dir.
Git installs (prfpy, dpu_mini) require outbound internet — run on a login node, not
a compute node.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CONDA = os.environ.get("PYPACKAGE_MANAGER", "mamba")
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", Path(__file__).resolve().parent.parent))
ENVS_DIR = Path(__file__).resolve().parent / "envs"
FSL_VERSION = os.environ.get("FSL_VERSION", "6.0.7.19")

VALID_ENVS = ["b14", "autoflat", "pctx", "preproc", "prf"]

_FSL_PLATFORMS = {
    ("Linux", "x86_64"): "linux-64",
    ("Linux", "aarch64"): "linux-aarch64",
    ("Darwin", "x86_64"): "macos-64",
    ("Darwin", "arm64"): "macos-M1",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _run(cmd: list) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def _env_exists(name: str, prefix: Path | None) -> bool:
    result = subprocess.run(
        [CONDA, "env", "list"], capture_output=True, text=True, check=True
    )
    target_path = str(prefix / name) if prefix else None
    for line in result.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if target_path and parts[-1] == target_path:
            return True
        if not target_path and parts[0] == name:
            return True
    return False


def _target(name: str, prefix: Path | None) -> list:
    """Return the --name / --prefix flags for a conda command."""
    return ["--prefix", str(prefix / name)] if prefix else ["--name", name]


def _conda_run(name: str, prefix: Path | None, cmd: list) -> None:
    _run([CONDA, "run"] + _target(name, prefix) + cmd)


def _install_from_yml(
    name: str, yml: Path, prefix: Path | None, clean: bool, update: bool
) -> bool:
    """
    Create or update an env from a yml file.
    Returns True if an install or update was performed.
    """
    exists = _env_exists(name, prefix)
    flags = _target(name, prefix)

    if exists:
        if clean:
            print(f"  Removing '{name}'...")
            _run([CONDA, "env", "remove", "-y"] + flags)
        elif update:
            print(f"  Updating '{name}'...")
            _run([CONDA, "env", "update", "-f", str(yml)] + flags)
            return True
        else:
            print(f"  '{name}' already exists — skipping (use --update or --clean-env to modify).")
            return False

    _run([CONDA, "env", "create", "-f", str(yml)] + flags)
    return True


# ─────────────────────────────────────────────
# Per-environment installers
# ─────────────────────────────────────────────

def _install_b14(prefix: Path | None, clean: bool, update: bool, suffix: str = "") -> None:
    name = "b14" + suffix
    print(f"\n=== {name} (Benson atlas) ===")
    _install_from_yml(name, ENVS_DIR / "b14.yml", prefix, clean, update)


def _install_autoflat(prefix: Path | None, clean: bool, update: bool, suffix: str = "") -> None:
    name = "autoflat" + suffix
    print(f"\n=== {name} ===")
    _install_from_yml(name, ENVS_DIR / "autoflat.yml", prefix, clean, update)


def _install_pctx(prefix: Path | None, clean: bool, update: bool, suffix: str = "") -> None:
    name = "pctx" + suffix
    print(f"\n=== {name} (pycortex) ===")
    _install_from_yml(name, ENVS_DIR / "pctx.yml", prefix, clean, update)


def _install_prf(prefix: Path | None, clean: bool, update: bool, suffix: str = "") -> None:
    name = "prf" + suffix
    print(f"\n=== {name} ===")
    acted = _install_from_yml(name, ENVS_DIR / "prf.yml", prefix, clean, update)
    if acted:
        # --no-build-isolation: pip uses the conda env directly instead of a fresh temp
        # environment, so it sees conda-installed packages (contourpy, matplotlib, etc.)
        # and won't try to compile them from source with the HPC's ICC.
        for pkg in [
            "git+https://github.com/mdaghlian/prfpy.git",
            "git+https://github.com/mdaghlian/dpu_mini.git",
        ]:
            _conda_run(name, prefix, ["pip", "install", "--no-build-isolation", pkg]) #"--no-deps", pkg])
        _conda_run(name, prefix, [
            "pip", "install", "-e", str(PIPELINE_DIR / "cvl_utils"), "--no-deps"
        ])


def _install_preproc(prefix: Path | None, clean: bool, update: bool, suffix: str = "") -> None:
    name = "preproc" + suffix
    print(f"\n=== {name} (FSL base + pipeline extras) ===")

    fsl_platform = _FSL_PLATFORMS.get((platform.system(), platform.machine()))
    if not fsl_platform:
        sys.exit(f"Unsupported platform: {platform.system()}-{platform.machine()}")

    exists = _env_exists(name, prefix)
    flags = _target(name, prefix)

    if exists and not (clean or update):
        print(f"  '{name}' already exists — skipping (use --update or --clean-env to modify).")
        return

    if exists and clean:
        print(f"  Removing '{name}'...")
        _run([CONDA, "env", "remove", "-y"] + flags)
        exists = False

    fsl_url = (
        f"https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/"
        f"fsl-{FSL_VERSION}_{fsl_platform}.yml"
    )
    print(f"  Downloading FSL yml ({fsl_platform}, v{FSL_VERSION})...")

    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
        fsl_tmp = Path(f.name)

    try:
        urllib.request.urlretrieve(fsl_url, fsl_tmp)

        if not exists:
            # FSL yml has its own name; --name/--prefix overrides it
            _run([CONDA, "env", "create", "-f", str(fsl_tmp)] + flags)
        else:
            _run([CONDA, "env", "update", "-f", str(fsl_tmp)] + flags)

        # Layer pipeline extras on top — no --prune so FSL packages are preserved
        _run([CONDA, "env", "update", "-f", str(ENVS_DIR / "preproc_extras.yml")] + flags)
        _conda_run(name, prefix, [
            "pip", "install", "-e", str(PIPELINE_DIR / "cvl_utils"), "--no-deps"
        ])
    finally:
        fsl_tmp.unlink(missing_ok=True)


_INSTALLERS = {
    "b14": _install_b14,
    "autoflat": _install_autoflat,
    "pctx": _install_pctx,
    "preproc": _install_preproc,
    "prf": _install_prf,
}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env", choices=VALID_ENVS, metavar="ENV",
        help=f"Install one environment. Choices: {', '.join(VALID_ENVS)}",
    )
    parser.add_argument(
        "--env-suffix", type=str, default="", metavar="SUFFIX",
        help="Append SUFFIX to environment names (e.g. --env-suffix '-test' creates 'b14-test', etc.)",
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--clean-env", action="store_true",
        help="Remove and recreate environment if it already exists",
    )
    group.add_argument(
        "--update", action="store_true",
        help="Update existing environment from yml (idempotent, non-destructive)",
    )
    parser.add_argument(
        "--prefix", type=Path, default=None, metavar="PATH",
        help="Install envs under PATH instead of the default conda envs dir "
             "(HPC: e.g. /scratch/$USER/envs)",
    )
    args = parser.parse_args()

    targets = [args.env] if args.env else VALID_ENVS
    for name in targets:
        _INSTALLERS[name](args.prefix, args.clean_env, args.update, args.env_suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Thin importlib loader for prebuilt .so kernels.

Replaces torch.utils.cpp_extension.load() — no JIT, no env-var lookups,
no ninja. Assumes the Makefile has produced release/build/lib<name>.so.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def load(name: str):
    """Load release/build/lib<name>.so as a Python module.

    Args:
        name: kernel base name (e.g. "dispatch_gemm").

    Returns:
        Python module exposing the kernel's pybind functions.

    Raises:
        RuntimeError if the .so doesn't exist (run `make` first).
    """
    so = BUILD / f"lib{name}.so"
    if not so.exists():
        raise RuntimeError(
            f"{so} does not exist. Run `make {name}` (or `make all`) "
            f"from {ROOT} first."
        )
    mod_name = f"mkernel_release_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import spec for {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

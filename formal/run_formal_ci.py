"""
Formal CI entrypoint (no external EasyCrypt binary required).

Runs:
  1) Python Unruh combinatorial checker
  2) Python QROM game-hop checker
  3) Lean build if `lake` is on PATH
  4) Static validation that EasyCrypt sources import in-repo QROM.ec

Exit 0 iff all available checks pass.
"""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_py_module(mod: str) -> None:
    print(f"==> python -m {mod}")
    subprocess.check_call([sys.executable, "-m", mod], cwd=ROOT)


def check_easycrypt_sources() -> None:
    ec_dir = os.path.join(ROOT, "formal", "easycrypt")
    qrom = os.path.join(ec_dir, "QROM.ec")
    unruh = os.path.join(ec_dir, "UnruhBinaryQROM.ec")
    counting = os.path.join(ec_dir, "UnruhBinaryCounting.ec")
    for p in (qrom, unruh, counting):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    text = open(unruh, encoding="utf-8").read()
    if "require import QROM" not in text:
        raise AssertionError("UnruhBinaryQROM.ec must import in-repo QROM")
    qtext = open(qrom, encoding="utf-8").read()
    if "soundness_bound" not in qtext or "qrom_term" not in qtext:
        raise AssertionError("QROM.ec missing soundness_bound / qrom_term")
    print("[ok] EasyCrypt sources link in-repo QROM.ec (stdlib-free)")


def try_lake_build() -> None:
    lean_dir = os.path.join(ROOT, "formal", "lean")
    lake = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake.exe")
    if os.name != "nt":
        lake = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake")
    if not os.path.isfile(lake):
        # PATH fallback
        from shutil import which

        lake = which("lake")
    if not lake:
        print("[skip] lake not found")
        return
    print(f"==> lake build ({lake})")
    subprocess.check_call([lake, "build"], cwd=lean_dir)


def try_easycrypt() -> None:
    from shutil import which

    ec = which("easycrypt")
    if not ec:
        print("[skip] easycrypt binary not installed (sources validated statically)")
        return
    ec_dir = os.path.join(ROOT, "formal", "easycrypt")
    print("==> easycrypt UnruhBinaryQROM.ec")
    subprocess.check_call([ec, "-I", ec_dir, "UnruhBinaryQROM.ec"], cwd=ec_dir)


def main() -> int:
    sys.path.insert(0, ROOT)
    run_py_module("formal.check_unruh_soundness")
    run_py_module("formal.check_unruh_qrom_games")
    check_easycrypt_sources()
    try_lake_build()
    try_easycrypt()
    print("FORMAL_CI_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

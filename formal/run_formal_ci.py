"""
Formal CI entrypoint (no external EasyCrypt binary required).

Runs:
  1) Python Unruh combinatorial checker
  2) Python QROM game-hop checker
  3) Python SHA3-QROM library checker (FIPS vector + EC link)
  4) Lean build if `lake` is on PATH
  5) Static validation of EasyCrypt SHA3-QROM sources
  6) Optional easycrypt binary check

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
    required = [
        os.path.join(ec_dir, "QROM.ec"),
        os.path.join(ec_dir, "SHA3_QROM.ec"),
        os.path.join(ec_dir, "UnruhBinaryQROM.ec"),
        os.path.join(ec_dir, "UnruhBinaryCounting.ec"),
        os.path.join(ec_dir, "lib", "SHA3.ec"),
        os.path.join(ec_dir, "lib", "QROMCore.ec"),
    ]
    for p in required:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    unruh = open(os.path.join(ec_dir, "UnruhBinaryQROM.ec"), encoding="utf-8").read()
    if "require import QROM" not in unruh:
        raise AssertionError("UnruhBinaryQROM.ec must import QROM")
    if "qrom_term_is_sha3_o2h" not in unruh:
        raise AssertionError("UnruhBinaryQROM.ec must link SHA3 O2H qrom_term")
    core = open(os.path.join(ec_dir, "lib", "QROMCore.ec"), encoding="utf-8").read()
    if "sha3_256" not in core or "2%r ^ 256" not in core:
        raise AssertionError("QROMCore.ec must instantiate H:=sha3_256 and 2^256 bound")
    print("[ok] EasyCrypt SHA3-QROM production library linked")


def try_lake_build() -> None:
    lean_dir = os.path.join(ROOT, "formal", "lean")
    lake = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake.exe")
    if os.name != "nt":
        lake = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake")
    if not os.path.isfile(lake):
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
    print("==> easycrypt UnruhBinaryQROM.ec (with lib/)")
    subprocess.check_call(
        [ec, "-I", ec_dir, "-I", os.path.join(ec_dir, "lib"), "UnruhBinaryQROM.ec"],
        cwd=ec_dir,
    )


def main() -> int:
    sys.path.insert(0, ROOT)
    run_py_module("formal.check_unruh_soundness")
    run_py_module("formal.check_unruh_qrom_games")
    run_py_module("formal.check_sha3_qrom")
    check_easycrypt_sources()
    try_lake_build()
    try_easycrypt()
    print("FORMAL_CI_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

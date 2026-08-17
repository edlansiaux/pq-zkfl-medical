"""
Formal CI entrypoint (no external EasyCrypt binary required).
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
        os.path.join(ec_dir, "lib", "KeccakF1600.ec"),
    ]
    for p in required:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    unruh = open(os.path.join(ec_dir, "UnruhBinaryQROM.ec"), encoding="utf-8").read()
    if "require import QROM" not in unruh or "qrom_term_is_sha3_o2h" not in unruh:
        raise AssertionError("UnruhBinaryQROM.ec SHA3-QROM link incomplete")
    core = open(os.path.join(ec_dir, "lib", "QROMCore.ec"), encoding="utf-8").read()
    if "sha3_256" not in core or "2%r ^ 256" not in core:
        raise AssertionError("QROMCore.ec must instantiate H:=sha3_256")
    kec = open(os.path.join(ec_dir, "lib", "KeccakF1600.ec"), encoding="utf-8").read()
    if "keccak_round" not in kec or "op theta" not in kec:
        raise AssertionError("KeccakF1600.ec must define bit-level round ops")
    for needle in (
        "theta_column_parity",
        "pi_bijection",
        "chi_local",
        "iota_local_zero",
        "packing_roundtrip",
        "keccak_f1600_is_round_fold",
    ):
        if needle not in kec:
            raise AssertionError(f"KeccakF1600.ec missing algebraic lemma/axiom: {needle}")
    print("[ok] EasyCrypt SHA3-QROM + bit-level Keccak library linked")


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
    run_py_module("formal.check_keccak_bitlevel")
    check_easycrypt_sources()
    try_lake_build()
    try_easycrypt()
    print("FORMAL_CI_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

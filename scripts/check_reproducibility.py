from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_PATH = PROJECT_ROOT / "data" / "raw_checksums.txt"
LOCK_PATH = PROJECT_ROOT / "requirements-lock.txt"
KEY_PACKAGES = ("scikit-learn", "pandas", "numpy", "scipy", "joblib")


def _run_git_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_checksum_file() -> list[tuple[str, Path]]:
    expected: list[tuple[str, Path]] = []
    for line in CHECKSUM_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected_hash, relative_path = line.split(maxsplit=1)
        expected.append((expected_hash.upper(), PROJECT_ROOT / relative_path))
    return expected


def check_data_hashes() -> bool:
    print("\nData checksums")
    print("-" * 80)

    ok = True
    for expected_hash, path in _parse_checksum_file():
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        if not path.exists():
            print(f"MISSING  {relative_path}")
            ok = False
            continue

        actual_hash = _sha256(path)
        status = "OK" if actual_hash == expected_hash else "MISMATCH"
        print(f"{status:<8} {relative_path}")
        if actual_hash != expected_hash:
            print(f"         expected {expected_hash}")
            print(f"         actual   {actual_hash}")
            ok = False
    return ok


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_lock_file() -> dict[str, str]:
    locked: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        locked[_normalize_package_name(name)] = version
    return locked


def check_locked_packages() -> bool:
    print("\nLocked package versions")
    print("-" * 80)

    ok = True
    locked = _parse_lock_file()
    for package_name in KEY_PACKAGES:
        normalized_name = _normalize_package_name(package_name)
        expected_version = locked.get(normalized_name)
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = "missing"

        status = "OK" if actual_version == expected_version else "MISMATCH"
        print(f"{status:<8} {package_name:<14} expected={expected_version} actual={actual_version}")
        if actual_version != expected_version:
            ok = False
    return ok


def print_environment_summary() -> None:
    print("Code and runtime")
    print("-" * 80)
    print(f"Git commit: { _run_git_command(['rev-parse', 'HEAD']) }")
    git_status = _run_git_command(["status", "--short"])
    print(f"Git status: {git_status if git_status else 'clean'}")
    print(f"Python:     {sys.version.split()[0]}")


def main() -> int:
    print_environment_summary()
    data_ok = check_data_hashes()
    package_ok = check_locked_packages()

    print("\nResult")
    print("-" * 80)
    if data_ok and package_ok:
        print("Reproducibility check passed.")
        return 0

    print("Reproducibility check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

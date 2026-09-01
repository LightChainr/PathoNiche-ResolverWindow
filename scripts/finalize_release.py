#!/usr/bin/env python3
"""Write deterministic file and SHA-256 manifests for the release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_MANIFEST = ROOT / "FILE_MANIFEST.txt"
SHA256SUMS = ROOT / "SHA256SUMS.txt"
EXCLUDED = {
    FILE_MANIFEST.name,
    SHA256SUMS.name,
    "release_validation_report.json",
}


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    )


def main() -> None:
    files = included_files()
    manifest_lines = ["bytes\tpath"]
    checksum_lines = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        manifest_lines.append(f"{path.stat().st_size}\t{relative}")
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    FILE_MANIFEST.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    SHA256SUMS.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"finalized {len(files)} files")


if __name__ == "__main__":
    main()

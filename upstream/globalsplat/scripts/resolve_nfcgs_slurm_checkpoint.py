#!/usr/bin/env python3
"""Resolve the final NFCGS checkpoint from a runner's SLURM stdout log."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex


_VERSION = re.compile(r"^version_(\d+)$")


def output_roots(log_path: Path) -> list[Path]:
    roots: list[Path] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        for token in tokens:
            if token.startswith("output_dir="):
                roots.append(Path(token.removeprefix("output_dir=")).expanduser())
    return roots


def checkpoint_sort_key(path: Path) -> tuple[int, int]:
    version = -1
    for parent in path.parents:
        match = _VERSION.match(parent.name)
        if match:
            version = int(match.group(1))
            break
    return version, path.stat().st_mtime_ns


def resolve_checkpoint(log_path: Path) -> Path:
    roots = output_roots(log_path)
    if not roots:
        raise FileNotFoundError(f"no output_dir=... command found in {log_path}")
    for root in reversed(roots):
        candidates = list(root.glob("*/version_*/last.ckpt"))
        if candidates:
            return max(candidates, key=checkpoint_sort_key).resolve()
    raise FileNotFoundError(f"no version_*/last.ckpt found below {roots[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(resolve_checkpoint(args.log.resolve()))


if __name__ == "__main__":
    main()

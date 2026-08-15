#!/usr/bin/env python3
"""Compile the canonical LaTeX manuscript."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main() -> None:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise SystemExit("tectonic is required: brew install tectonic")
    subprocess.run([tectonic, "--keep-logs", "--keep-intermediates", "main.tex"], cwd=PAPER, check=True)
    print(PAPER / "main.pdf")


if __name__ == "__main__":
    main()

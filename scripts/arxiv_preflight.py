#!/usr/bin/env python3
"""Audit the exact manuscript source archive before a human submission review."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "paper" / "arxiv-source.zip"
EXPECTED = {
    "main.tex",
    "references.bib",
    "tables/generated-main-results.tex",
    "figures/generated-method-overview.pdf",
    "figures/generated-finecops-difficulty.pdf",
    "figures/generated-efficiency-summary.pdf",
}
FORBIDDEN = re.compile(r"TODO|TBD|FIXME|PLACEHOLDER|/Users/|private[_ -]?key|secret[_ -]?key", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    archive = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ARCHIVE
    require(archive.is_file(), f"Missing archive: {archive}")
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        require(names == EXPECTED, f"Unexpected archive contents: {sorted(names ^ EXPECTED)}")
        source = "\n".join(package.read(name).decode("utf-8") for name in ("main.tex", "references.bib", "tables/generated-main-results.tex"))
        require(FORBIDDEN.search(source) is None, "Archive has a placeholder, local path, or sensitive marker")
        main_tex = package.read("main.tex").decode("utf-8")
        references = package.read("references.bib").decode("utf-8")
        require("\\title{" in main_tex and "\\author{" in main_tex, "Title or author missing")
        require("\\appendix" in main_tex and "Reproducibility Details" in main_tex, "Reproducibility appendix missing")
        cited = {key.strip() for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", main_tex) for key in match.group(1).split(",")}
        bibliography = set(re.findall(r"^@\w+\{([^,]+),", references, re.M))
        require(cited <= bibliography, f"Missing bibliography keys: {sorted(cited - bibliography)}")
        included = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_tex))
        included.update(re.findall(r"\\input\{([^}]+)\}", main_tex))
        require(included <= names, f"Archive misses included files: {sorted(included - names)}")
    print(f"arXiv source preflight passed: {len(EXPECTED)} files, {len(cited)} citations, source-only audit")


if __name__ == "__main__":
    main()

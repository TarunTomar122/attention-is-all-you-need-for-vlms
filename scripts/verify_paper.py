#!/usr/bin/env python3
"""Check that the manuscript package still agrees with frozen experiment outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_paper_assets.py")], check=True)
    data = json.loads((ROOT / "paper/data/paper-data.json").read_text())
    require(data["frozen_contract"] == {"tau": 0.8, "bootstrap_seed": 20260812, "bootstrap_replicates": 10_000}, "frozen evaluation contract changed")
    require(round(data["results"]["finecops"]["a4_minus_s4_pp"], 2) == -0.52, "FineCops point estimate changed")
    require(round(data["results"]["finecops"]["a8_minus_s4_pp"], 2) == 0.26, "A8 recovery estimate changed")
    require(round(data["results"]["refadv"]["a4_minus_s4_pp"], 2) == 0.79, "Ref-Adv result changed")
    for stem in ("method-overview", "finecops-difficulty", "efficiency-summary"):
        for suffix in ("png", "pdf"):
            path = ROOT / f"paper/figures/generated-{stem}.{suffix}"
            require(path.is_file() and path.stat().st_size > 1000, f"missing asset: {path}")
    for path in ("paper/main.tex", "paper/draft.md", "paper/references.bib", "paper/citation-audit.md", "paper/data-availability.md", "paper/claims-and-limitations.md", "paper/tables/generated-main-results.tex"):
        require((ROOT / path).is_file(), f"missing paper file: {path}")
    subprocess.run([sys.executable, str(ROOT / "scripts/render_paper_pdf.py")], check=True)
    pdf = ROOT / "paper/main.pdf"
    require(pdf.is_file() and pdf.stat().st_size > 50_000, "missing or empty rendered paper PDF")
    from pypdf import PdfReader

    reader = PdfReader(pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    require(len(reader.pages) == 4, "rendered paper page count changed")
    require(reader.metadata.author == "Tarun Tomar", "rendered paper author changed")
    for phrase in ("FineCops-Ref", "Qualitative-panel provenance", "References"):
        require(phrase in text, f"rendered paper omits {phrase}")
    print("Paper verification passed: frozen contract, result table, and 9 generated figure assets.")


if __name__ == "__main__":
    main()

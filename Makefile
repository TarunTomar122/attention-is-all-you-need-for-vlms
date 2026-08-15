.PHONY: paper-assets verify-paper submission paper-pdf overleaf-package arxiv-package arxiv-preflight clean-paper-assets

PYTHON ?= python3

paper-assets:
	$(PYTHON) scripts/generate_paper_assets.py

verify-paper:
	$(PYTHON) scripts/verify_paper.py

submission: paper-assets verify-paper

paper-pdf: paper-assets
	$(PYTHON) scripts/render_paper_pdf.py

overleaf-package: paper-assets
	cd paper && zip -FS -q overleaf-package.zip main.tex references.bib figures/generated-method-overview.pdf figures/generated-finecops-difficulty.pdf figures/generated-efficiency-summary.pdf tables/generated-main-results.tex

arxiv-package: paper-assets
	cd paper && zip -FS -q arxiv-source.zip main.tex references.bib figures/generated-method-overview.pdf figures/generated-finecops-difficulty.pdf figures/generated-efficiency-summary.pdf tables/generated-main-results.tex

arxiv-preflight: arxiv-package
	$(PYTHON) scripts/arxiv_preflight.py

clean-paper-assets:
	rm -f paper/data/paper-data.json paper/tables/generated-* paper/figures/generated-* docs/assets/generated-*

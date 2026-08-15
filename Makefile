.PHONY: paper-assets verify-paper paper-pdf

paper-assets:
	python3 scripts/generate_paper_assets.py

verify-paper:
	python3 scripts/verify_paper.py

paper-pdf:
	python3 scripts/render_paper_pdf.py

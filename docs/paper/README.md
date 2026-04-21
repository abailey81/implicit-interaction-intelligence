# Rendering the I³ Research Paper

This directory contains the academic paper for the I³ project
(*Implicit Interaction Intelligence: Cross-Attention Conditioning
for On-Device Adaptive Language Models from Behavioural Signals*).

## Files

- `I3_research_paper.md` — the paper itself, written as single-column
  Markdown targeting an IEEE/ACM two-column render via pandoc.
- `references.bib` — BibTeX bibliography (28 entries).
- `README.md` — this file.

## Quick render (PDF via pandoc + XeLaTeX)

The canonical render command, which produces a two-column IEEE-style PDF:

```bash
pandoc \
  -V documentclass=IEEEtran \
  -V twocolumn \
  --pdf-engine=xelatex \
  --bibliography=references.bib \
  --citeproc \
  -o paper.pdf \
  I3_research_paper.md
```

Requirements:

- `pandoc` ≥ 3.1 (for LaTeX math passthrough and biblatex support).
- XeLaTeX toolchain (`tlmgr install ieeetran`, or TeX Live `collection-latexextra`).
- The `IEEEtran.cls` document class (bundled with the `ieeetran` TeX Live package).

## Alternative renders

**Two-column ACM style (no IEEEtran):**

```bash
pandoc -V documentclass=sigconf -V twocolumn \
  --pdf-engine=xelatex --bibliography=references.bib --citeproc \
  -o paper-acm.pdf I3_research_paper.md
```

**Plain single-column for quick review:**

```bash
pandoc --pdf-engine=xelatex --bibliography=references.bib --citeproc \
  -o paper-review.pdf I3_research_paper.md
```

**HTML (for inline review / MathJax):**

```bash
pandoc --mathjax --bibliography=references.bib --citeproc \
  -s -o paper.html I3_research_paper.md
```

## Notes

- All equations are written in LaTeX and render under MathJax 3
  (HTML) or native LaTeX (PDF).
- Citations in the Markdown source use plain `[N]` numeric style; the
  BibTeX bibliography is provided for future conversion to a
  citeproc-compatible citation style if desired.
- The paper targets 8 pages two-column (≈ 6 500–8 500 words).
- Do not modify `I3_research_paper.md` or `references.bib` without
  updating the author / version stamps in the paper header.

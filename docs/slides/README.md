# Slide Deck — Implicit Interaction Intelligence (I³)

15-slide technical presentation for the Huawei London HMI Lab interview on
**29 April 2026, 12:00**, Meeting Room MR1, Gridiron Building, 1 Pancras
Square. Slides submitted to `matthew.riches@huawei.com` on **28 April 2026**
with subject `Technical Presentation — Implicit Interaction Intelligence (I³)`.

## Files

| File | Purpose |
|------|---------|
| `presentation.md`      | The 15-slide deck — Marp-compatible Markdown |
| `marp-theme.css`       | Custom Marp theme matching the project palette |
| `speaker_notes.md`     | Per-slide narration (150–200 words, 60–90 s) |
| `rehearsal_timings.md` | Cue-sheet with per-slide target + cumulative time |
| `qa_prep.md`           | 52 prepared Q&A pairs across 7 categories |
| `closing_lines.md`     | Verbatim closing line, candidate questions, honesty wording |

## Palette (kept consistent across all render paths)

| Token      | Hex        | Role                        |
|------------|------------|-----------------------------|
| `bg`       | `#1a1a2e`  | Slide background            |
| `panel`    | `#16213e`  | Panels, code blocks         |
| `accent`   | `#0f3460`  | Secondary accent            |
| `highlight`| `#e94560`  | Emphasis, highlight strokes |
| `muted`    | `#a0a0b0`  | Secondary text, captions    |
| `active`   | `#f0f0f0`  | Primary text                |

Mono font for numeric tables, code, and formulas. Sans-serif (system default)
for body copy. Slide size is 16:9.

---

## Option 1 — Marp CLI (primary)

Marp renders `presentation.md` straight to PDF and PPTX. This is the
submission path.

### Install

```
npm install -g @marp-team/marp-cli
```

### Render both PDF and PPTX (exact submission command)

```
marp presentation.md --pdf --pptx --theme marp-theme.css --allow-local-files
```

That produces `presentation.pdf` and `presentation.pptx` side-by-side. The
`--allow-local-files` flag is only needed if the deck later embeds images
from the `docs/assets/` tree.

### Live preview while editing

```
marp presentation.md --preview --theme marp-theme.css
```

### Speaker-notes export (HTML with `<aside class="notes">`)

Marp already picks up the `<!-- _notes: ... -->` comments inline in
`presentation.md`. For a standalone notes document, rely on
`speaker_notes.md` — it is the canonical rehearsal text.

---

## Option 2 — reveal.js

Useful if the deck needs to be presented from a browser rather than a PDF
viewer on the interview laptop. The deck already uses `---` as slide
separators, which reveal.js understands natively via `reveal-md`.

### Install

```
npm install -g reveal-md
```

### Serve locally

```
reveal-md presentation.md --theme black --css marp-theme.css
```

### Static export (self-contained HTML bundle)

```
reveal-md presentation.md --static _site --theme black --css marp-theme.css
```

Open `_site/index.html` — it is portable and works offline. This is a viable
fallback if Marp's PDF rendering breaks at the last minute.

---

## Option 3 — Pandoc → PPTX / PDF

Pandoc gives finer control when the Marp output needs post-editing in
PowerPoint or Keynote.

### PPTX (PowerPoint-editable)

```
pandoc presentation.md -o presentation.pptx \
  --slide-level=1 \
  --reference-doc=reference.pptx
```

The optional `--reference-doc` is a blank PPTX whose master slide defines
fonts and colours — drop one in the same folder if corporate branding
changes.

### PDF via LaTeX Beamer

```
pandoc presentation.md -o presentation.pdf \
  -t beamer \
  --slide-level=1 \
  -V theme:metropolis \
  -V colortheme:owl
```

This path does not honour `marp-theme.css` directly — colours must be set
via Beamer colour themes. Use it only as a last-resort PDF route.

---

## Rendering checklist (28 April, submission day)

1. `marp presentation.md --pdf --pptx --theme marp-theme.css`
2. Open both outputs. Confirm:
   - 15 slides, no orphan slides
   - All em dashes (`—`) render as em dashes, not `--`
   - Monospace numeric tables are aligned
   - No bullet wraps to a third line anywhere
   - Palette is applied everywhere (no default Marp white)
3. Attach both files to the email.
4. Copy both onto the two USB drives alongside the 5-minute backup demo MP4.
5. Print the PDF 6-per-page, landscape, for note-making during Q&A.

---

## House style

- Every slide title leads with an experience verb or noun, never a technology.
- ≤ 7 bullets per slide, ≤ 12 words per bullet, never wrap to a third line.
- No emojis. Mono only for code and numeric tables.
- Em dash (`—`) everywhere, never `--`.
- Every Huawei product claim is footnoted with a public source.
- No "revolutionary", "cutting-edge", or "state-of-the-art".

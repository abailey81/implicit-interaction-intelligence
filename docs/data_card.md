# Data Card

A datasheet in the style of Gebru *et al.*, "Datasheets for Datasets"
(CACM 2021). Describes the data used to train and evaluate the shipped
I³ checkpoints.

## Motivation { #motivation }

### Why was this dataset created?

To train the TCN encoder on *implicit interaction traces* without using
any real users' keystroke data, and to train the SLM on neutral,
multi-topic dialogue. Synthetic traces give us full ground-truth control
for the contrastive objective; public dialogue corpora give us a broad
language distribution.

### Who created the dataset, and on whose behalf?

The synthetic trace generator is part of the I³ repository
(`training/generate_synthetic.py`). The public dialogue sources are
DailyDialog (Li *et al.*, 2017) and EmpatheticDialogues
(Rashkin *et al.*, 2019), used under their published licences.

### Who funded it?

Unfunded academic / demonstration work.

## Composition { #composition }

### What do the instances represent?

Two independent datasets are used:

1. **Synthetic interaction traces** — tuples of
   `(keystroke_intervals, composition_time, edit_count, pause, user_state_label)`.
2. **Dialogue corpora** — short multi-turn exchanges in English.

### How many instances?

- Synthetic: 10,000 sessions × ~20 messages per session.
- DailyDialog: ~13k dialogues.
- EmpatheticDialogues: ~25k dialogues.

### Does the dataset contain all possible instances?

No. The synthetic generator has eight user states; real users inhabit a
continuum.

### What does each instance consist of?

- **Synthetic**: numeric keystroke features + a generated message string
  consistent with the user state's linguistic baseline.
- **Dialogue**: turn text, emotion label (where available).

### Is there a label or target?

- **Synthetic**: yes, the latent user-state index is recorded but not
  used for the contrastive objective.
- **Dialogue**: next-token target.

### Is any information missing?

No intentional redactions beyond the standard upstream corpora cleaning.

### Are relationships between instances made explicit?

Within-session ordering is preserved in both datasets.

### Splits

| Split | Synthetic | DailyDialog | EmpatheticDialogues |
|:------|----------:|------------:|--------------------:|
| Train | 80 % | 80 % | 80 % |
| Val   | 10 % | 10 % | 10 % |
| Test  | 10 % | 10 % | 10 % |

Splits are deterministic under `--seed`.

### Are there errors, noise, or redundancies?

- Dialogue corpora contain transcription artefacts and occasional NSFW
  vocabulary; we do **not** fine-tune beyond them, and the tokenizer's
  fixed vocabulary effectively filters many rare tokens as UNK.
- Synthetic traces are by construction noise-free.

### Is the dataset self-contained?

Yes once the generator is run; no external network fetches at training
time.

### Is the dataset confidential?

No. Everything used is either synthetic or from public datasets.

## Collection Process { #collection }

### How was data acquired?

- **Synthetic**: generated via the Markov + state-parametrised generator
  in `training/generate_synthetic.py`. See the module docstring for the
  generator's joint distribution.
- **Public**: downloaded from the corpora's canonical distribution points
  by `training/prepare_dialogue.py`.

### Who was involved?

- **Synthetic**: no humans.
- **Public**: original annotators of the upstream corpora.

### Over what timeframe?

Data preparation runs in seconds (synthetic) to minutes (corpora download
and tokenisation). No ongoing collection.

### Ethical review?

Synthetic generation requires none. Public corpora were released by
their original authors with their own IRB equivalents.

### Notifications and consent?

No identifiable individuals are in the synthetic data. The public corpora
are crowd-sourced under their published licences.

## Preprocessing / Cleaning / Labelling { #preprocessing }

- **PII sanitisation** — the same ten-pattern sanitiser used at inference
  (see [Privacy](architecture/privacy.md)) is run over every training
  string. Matches become typed placeholders (`[REDACTED:email]` etc.).
- **Lowercasing and punctuation splitting** — before tokenisation.
- **UTF-8 normalisation** — NFC.
- **Vocabulary construction** — top 8192 most frequent post-cleaning
  tokens + special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`).

The raw (unredacted) corpus is not retained; only the sanitised + tokenised
shards are written to `data/dialogue/`.

## Uses { #uses }

### Has the dataset been used for other tasks?

The public corpora have been used widely in dialogue research under their
licences. The synthetic traces are novel.

### What tasks is it suitable for?

- Pre-training the TCN encoder (contrastive).
- Pre-training the SLM (next-token).
- Evaluating conditioning-mode ablations.

### What tasks is it *not* suitable for?

- Real-user keystroke-dynamics research (synthetic).
- Production-grade open-domain dialogue (scale).
- Sensitive-topic fine-tuning (licensing).

## Distribution { #distribution }

### Is the dataset available to others?

- **Synthetic**: re-derivable from the shipped generator + seed.
- **Public**: downloadable from their canonical URLs via the prepare
  script.

### Any restrictions?

- DailyDialog: **CC BY-NC-SA 4.0** — non-commercial, share-alike.
- EmpatheticDialogues: **CC BY 4.0** — attribution.

## Maintenance { #maintenance }

- **Maintainer**: tamer.atesyakar@bk.ru
- **Updates**: versioned with the I³ release cycle. The synthetic generator
  is pinned to the commit recorded in the checkpoint's `manifest.json`.
- **Deprecation**: if either upstream corpus withdraws, we will switch to
  a licence-compatible replacement and issue a new data card version.

## Known limitations { #limitations }

- **Cultural narrowness** — both public corpora are English-centric and
  skew towards US/UK norms.
- **Generator coverage** — the eight synthetic states cannot encode
  fine-grained emotion or idiosyncratic typists.
- **Domain shift** — production users may differ markedly from the
  dialogue distribution; we mitigate with the router's cloud arm for
  factual queries.

"""Prepare dialogue datasets for SLM training.

Processes DailyDialog and EmpatheticDialogues datasets into tokenized,
conditioned training data for the Adaptive SLM. If raw datasets are not
available locally, generates a synthetic fallback corpus to enable
development and testing.

Pipeline::

    1. Load raw dialogue data (or generate synthetic fallback)
    2. Extract (history, response, emotion_label) triples
    3. Derive conditioning labels from response text:
       - cognitive_load from Flesch-Kincaid readability
       - formality from contraction/slang count
       - emotionality from emotion label
       - verbosity from response length
    4. Build tokenizer vocabulary from training corpus
    5. Tokenize and pad all sequences
    6. Create PyTorch Dataset with conditioning vectors
    7. Save train/val/test splits as .pt files

Usage::

    python -m training.prepare_dialogue
    python -m training.prepare_dialogue --output-dir data/processed/dialogue
    python -m training.prepare_dialogue --vocab-size 8000 --max-seq-len 128
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

# Ensure project root is on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.slm.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DailyDialog emotion labels
EMOTION_LABELS: dict[int, str] = {
    0: "no_emotion",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}

# Contraction patterns for formality detection
CONTRACTIONS: set[str] = {
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
    "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't",
    "don't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "mustn't", "can't", "shan't", "let's", "that's", "who's", "what's",
    "here's", "there's", "when's", "where's", "why's", "how's",
    "gonna", "wanna", "gotta", "kinda", "sorta", "ya", "yeah", "yep",
    "nah", "nope", "lol", "omg", "btw", "idk", "imo",
}

# Slang words that indicate informality
SLANG_WORDS: set[str] = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
    "gimme", "cuz", "cos", "yall", "ain't", "nah", "yeah", "yep",
    "nope", "lol", "lmao", "omg", "bruh", "dude", "bro", "fam",
    "lit", "vibe", "chill", "cool", "sick", "dope", "hella",
    "stuff", "thing", "like", "totally", "literally", "basically",
    "honestly", "seriously", "actually", "whatever", "ok", "okay",
}


# ---------------------------------------------------------------------------
# Data loading: DailyDialog format
# ---------------------------------------------------------------------------

def load_dailydialog(data_dir: Path) -> list[dict[str, Any]]:
    """Load DailyDialog dataset from the standard directory structure.

    Expected files::

        data_dir/dialogues_text.txt    -- one dialogue per line, utterances
                                          separated by __eou__
        data_dir/dialogues_emotion.txt -- emotion labels per utterance

    Parameters
    ----------
    data_dir : Path
        Directory containing DailyDialog files.

    Returns
    -------
    list[dict]
        List of dialogue dicts with keys:
        - ``"utterances"`` -- list of utterance strings
        - ``"emotions"`` -- list of emotion ints
    """
    text_file = data_dir / "dialogues_text.txt"
    emotion_file = data_dir / "dialogues_emotion.txt"

    if not text_file.exists() or not emotion_file.exists():
        logger.warning(
            "DailyDialog files not found in %s. Will use synthetic data.",
            data_dir,
        )
        return []

    dialogues: list[dict[str, Any]] = []

    with open(text_file, "r", encoding="utf-8") as ft, \
         open(emotion_file, "r", encoding="utf-8") as fe:
        for text_line, emo_line in zip(ft, fe):
            utterances = [
                u.strip() for u in text_line.strip().split("__eou__")
                if u.strip()
            ]
            emotions = [
                int(e) for e in emo_line.strip().split()
                if e.strip().isdigit()
            ]

            # Ensure alignment
            min_len = min(len(utterances), len(emotions))
            if min_len >= 2:
                dialogues.append({
                    "utterances": utterances[:min_len],
                    "emotions": emotions[:min_len],
                })

    logger.info("Loaded %d dialogues from DailyDialog", len(dialogues))
    return dialogues


def load_empathetic_dialogues(data_dir: Path) -> list[dict[str, Any]]:
    """Load EmpatheticDialogues dataset from CSV files.

    Expected file: ``data_dir/train.csv`` with columns:
    conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags

    Parameters
    ----------
    data_dir : Path
        Directory containing EmpatheticDialogues files.

    Returns
    -------
    list[dict]
        List of dialogue dicts.
    """
    csv_file = data_dir / "train.csv"

    if not csv_file.exists():
        logger.warning(
            "EmpatheticDialogues not found at %s. Will use synthetic data.",
            csv_file,
        )
        return []

    # Simple CSV parsing (no pandas dependency)
    conversations: dict[str, list[tuple[int, str, str]]] = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                conv_id = parts[0]
                utt_idx = int(parts[1]) if parts[1].isdigit() else 0
                context = parts[2]
                utterance = parts[5] if len(parts) > 5 else ""

                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append((utt_idx, context, utterance))

    dialogues: list[dict[str, Any]] = []
    for conv_id, turns in conversations.items():
        turns.sort(key=lambda x: x[0])
        utterances = [t[2] for t in turns if t[2].strip()]
        if len(utterances) >= 2:
            # Use context as emotion label proxy
            emotion_label = turns[0][1] if turns else "neutral"
            dialogues.append({
                "utterances": utterances,
                "emotions": [0] * len(utterances),  # Simplified
                "emotion_context": emotion_label,
            })

    logger.info("Loaded %d dialogues from EmpatheticDialogues", len(dialogues))
    return dialogues


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_dialogues(
    n_dialogues: int = 5000, seed: int = 42
) -> list[dict[str, Any]]:
    """Generate synthetic dialogue data as a fallback when real data is unavailable.

    Creates dialogues covering a range of emotional states, formality
    levels, and complexity levels to ensure diverse conditioning signals.

    Parameters
    ----------
    n_dialogues : int
        Number of dialogues to generate (default 5000).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    list[dict]
        List of dialogue dicts.
    """
    # SEC: seed must be passed in (was hard-coded to 42) so that the CLI
    # ``--seed`` flag actually controls the synthetic-fallback corpus.
    import random
    random.seed(seed)

    # Templates organized by style
    greetings = [
        "Hello! How are you doing today?",
        "Hi there, what's going on?",
        "Hey, how's it going?",
        "Good morning! How can I help you?",
        "Good afternoon, what can I do for you?",
    ]

    casual_responses = [
        "Not much, just hanging out.",
        "Pretty good, thanks for asking!",
        "I'm doing alright, how about you?",
        "Yeah, I'm okay, just a bit tired.",
        "Doing great! Had a good day so far.",
        "Not bad at all, thanks!",
        "Could be better, but I'm managing.",
        "Just taking it easy today.",
        "I've been pretty busy lately.",
        "Things are going well, can't complain.",
    ]

    formal_responses = [
        "I am doing well, thank you for inquiring.",
        "I appreciate your concern. I have been well.",
        "Thank you for asking. Everything is going smoothly.",
        "I have been quite productive today, thank you.",
        "I am in good spirits. How may I assist you?",
        "Everything is proceeding according to plan.",
        "I have been engaged in several important tasks.",
        "I must say, today has been rather productive.",
        "I am pleased to report that all is well.",
        "Thank you for your consideration.",
    ]

    emotional_responses = [
        "I'm really happy today, everything feels wonderful!",
        "I'm feeling a bit down, to be honest.",
        "I'm so excited about what happened today!",
        "I've been feeling stressed and overwhelmed lately.",
        "I'm grateful for all the good things in my life.",
        "I'm worried about what might happen tomorrow.",
        "I feel peaceful and content right now.",
        "I'm frustrated with how things have been going.",
        "I'm thrilled about the news I just received!",
        "I'm feeling nostalgic, thinking about the old days.",
    ]

    complex_responses = [
        "The situation is multifaceted and requires careful consideration of several interconnected variables before any meaningful conclusion can be drawn.",
        "I've been contemplating the philosophical implications of our daily routines and how they shape our understanding of purpose and meaning.",
        "The correlation between environmental factors and behavioral outcomes suggests a more nuanced approach to problem-solving is warranted.",
        "When we examine the underlying assumptions of conventional wisdom, we often discover that reality is far more complex than initially anticipated.",
        "The interplay between cognitive biases and rational decision-making creates fascinating dynamics in everyday interactions.",
    ]

    simple_responses = [
        "Sure!",
        "Okay.",
        "Got it.",
        "Yes, thanks.",
        "Sounds good.",
        "That works.",
        "No problem.",
        "I see.",
        "Right.",
        "Makes sense.",
    ]

    follow_ups = [
        "That's interesting. Tell me more about that.",
        "I understand. What happened next?",
        "How did that make you feel?",
        "What do you think about it now?",
        "Is there anything I can help with?",
        "That sounds like quite an experience.",
        "I appreciate you sharing that with me.",
        "Would you like to talk more about it?",
        "That makes a lot of sense.",
        "I can see why you would feel that way.",
    ]

    all_pools = [
        casual_responses, formal_responses, emotional_responses,
        complex_responses, simple_responses, follow_ups,
    ]

    emotion_map = {
        "casual": 0,
        "formal": 0,
        "emotional_positive": 4,
        "emotional_negative": 5,
        "complex": 0,
        "simple": 0,
    }

    dialogues: list[dict[str, Any]] = []

    for i in range(n_dialogues):
        n_turns = random.randint(3, 8)
        utterances: list[str] = []
        emotions: list[int] = []

        # Start with a greeting
        utterances.append(random.choice(greetings))
        emotions.append(0)

        for _ in range(n_turns - 1):
            pool_idx = random.randint(0, len(all_pools) - 1)
            pool = all_pools[pool_idx]
            utterances.append(random.choice(pool))

            # Assign emotion based on pool
            if pool is emotional_responses:
                emotions.append(random.choice([4, 5, 6]))
            elif pool is casual_responses:
                emotions.append(0)
            elif pool is formal_responses:
                emotions.append(0)
            else:
                emotions.append(random.randint(0, 6))

        dialogues.append({
            "utterances": utterances,
            "emotions": emotions,
        })

    logger.info("Generated %d synthetic dialogues", len(dialogues))
    return dialogues


# ---------------------------------------------------------------------------
# Extract (history, response, emotion) triples
# ---------------------------------------------------------------------------

def extract_triples(
    dialogues: list[dict[str, Any]],
    max_history_turns: int = 3,
) -> list[dict[str, Any]]:
    """Extract (history, response, emotion_label) triples from dialogues.

    For each response utterance (starting from index 1), the preceding
    utterances form the history context. The emotion label of the
    response utterance is preserved for conditioning.

    Parameters
    ----------
    dialogues : list[dict]
        Raw dialogues with ``"utterances"`` and ``"emotions"`` keys.
    max_history_turns : int
        Maximum number of preceding turns to include in history.

    Returns
    -------
    list[dict]
        List of dicts with keys:
        - ``"history"`` -- concatenated history utterances
        - ``"response"`` -- response utterance text
        - ``"emotion"`` -- integer emotion label
    """
    triples: list[dict[str, Any]] = []

    for dialogue in dialogues:
        utterances = dialogue["utterances"]
        emotions = dialogue.get("emotions", [0] * len(utterances))

        for i in range(1, len(utterances)):
            # History: preceding turns (limited to max_history_turns)
            start = max(0, i - max_history_turns)
            history_turns = utterances[start:i]
            history = " [SEP] ".join(history_turns)

            response = utterances[i]
            emotion = emotions[i] if i < len(emotions) else 0

            triples.append({
                "history": history,
                "response": response,
                "emotion": emotion,
            })

    logger.info("Extracted %d (history, response, emotion) triples", len(triples))
    return triples


# ---------------------------------------------------------------------------
# Conditioning derivation from text heuristics
# ---------------------------------------------------------------------------

def compute_flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level for a text.

    Uses a simplified syllable counter. The formula is:

    .. math::

        FK = 0.39 \\cdot \\frac{\\text{words}}{\\text{sentences}}
           + 11.8 \\cdot \\frac{\\text{syllables}}{\\text{words}} - 15.59

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    float
        Grade level estimate (0-20 typical range).
    """
    words = text.split()
    n_words = max(len(words), 1)

    # Count sentences (split on . ! ?)
    sentences = re.split(r'[.!?]+', text)
    n_sentences = max(len([s for s in sentences if s.strip()]), 1)

    # Simplified syllable count
    def count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"()-")
        if not word:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        # Handle silent e
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    total_syllables = sum(count_syllables(w) for w in words)

    grade = (
        0.39 * (n_words / n_sentences)
        + 11.8 * (total_syllables / n_words)
        - 15.59
    )

    return grade


def derive_conditioning(
    response: str,
    emotion: int,
) -> dict[str, float]:
    """Derive 8-dim conditioning vector from response text heuristics.

    Maps text characteristics to AdaptationVector dimensions:

    - [0] cognitive_load: Flesch-Kincaid grade normalized to [0, 1]
    - [1] formality: based on contraction/slang usage (fewer = more formal)
    - [2] verbosity: based on response word count
    - [3] emotionality: based on emotion label intensity
    - [4] directness: based on sentence structure (shorter = more direct)
    - [5] emotional_tone: based on emotion valence
    - [6] accessibility: inverse of cognitive load
    - [7] reserved: 0.0

    Parameters
    ----------
    response : str
        The response text.
    emotion : int
        DailyDialog emotion label (0-6).

    Returns
    -------
    dict[str, float]
        Dictionary with keys matching the AdaptationVector dimensions.
    """
    words = response.lower().split()
    n_words = max(len(words), 1)

    # --- Cognitive load from Flesch-Kincaid ---
    fk_grade = compute_flesch_kincaid_grade(response)
    # Normalize: grade 0-2 -> 0.0, grade 12+ -> 1.0
    cognitive_load = max(0.0, min(1.0, (fk_grade - 2.0) / 10.0))

    # --- Formality from contraction/slang count ---
    contraction_count = sum(1 for w in words if w in CONTRACTIONS)
    slang_count = sum(1 for w in words if w in SLANG_WORDS)
    informal_ratio = (contraction_count + slang_count) / n_words
    # Higher ratio = less formal. Invert so formality=1 is most formal.
    formality = max(0.0, min(1.0, 1.0 - informal_ratio * 5.0))

    # --- Verbosity from response length ---
    # 1-5 words -> 0.0, 50+ words -> 1.0
    verbosity = max(0.0, min(1.0, (n_words - 5.0) / 45.0))

    # --- Emotionality from emotion label ---
    # 0=no_emotion -> 0.0, others -> higher
    emotion_intensity: dict[int, float] = {
        0: 0.1,   # no_emotion
        1: 0.8,   # anger
        2: 0.7,   # disgust
        3: 0.9,   # fear
        4: 0.7,   # happiness
        5: 0.8,   # sadness
        6: 0.6,   # surprise
    }
    emotionality = emotion_intensity.get(emotion, 0.3)

    # --- Directness from sentence structure ---
    sentences = re.split(r'[.!?]+', response)
    active_sentences = [s for s in sentences if s.strip()]
    avg_sentence_len = n_words / max(len(active_sentences), 1)
    # Short sentences = more direct
    directness = max(0.0, min(1.0, 1.0 - (avg_sentence_len - 5.0) / 20.0))

    # --- Emotional tone (valence) ---
    # Positive emotions -> higher tone, negative -> lower
    tone_map: dict[int, float] = {
        0: 0.5,   # neutral
        1: 0.2,   # anger -> low
        2: 0.2,   # disgust -> low
        3: 0.3,   # fear -> low-mid
        4: 0.9,   # happiness -> high
        5: 0.2,   # sadness -> low
        6: 0.6,   # surprise -> mid-high
    }
    emotional_tone = tone_map.get(emotion, 0.5)

    # --- Accessibility (inverse of cognitive load) ---
    accessibility = 1.0 - cognitive_load

    return {
        "cognitive_load": cognitive_load,
        "formality": formality,
        "verbosity": verbosity,
        "emotionality": emotionality,
        "directness": directness,
        "emotional_tone": emotional_tone,
        "accessibility": accessibility,
        "reserved": 0.0,
    }


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_conditioning_tensor(conditioning: dict[str, float]) -> torch.Tensor:
    """Convert conditioning dict to an 8-dim tensor.

    Parameters
    ----------
    conditioning : dict[str, float]
        Conditioning values from ``derive_conditioning()``.

    Returns
    -------
    torch.Tensor
        Shape ``[8]``, float32.
    """
    return torch.tensor([
        conditioning["cognitive_load"],
        conditioning["formality"],
        conditioning["verbosity"],
        conditioning["emotionality"],
        conditioning["directness"],
        conditioning["emotional_tone"],
        conditioning["accessibility"],
        conditioning["reserved"],
    ], dtype=torch.float32)


def build_synthetic_user_state(
    conditioning: dict[str, float],
    dim: int = 64,
) -> torch.Tensor:
    """Build a synthetic user state embedding from conditioning.

    Since we do not have real user interaction data during offline training,
    we create a synthetic 64-dim embedding by tiling and adding noise to
    the conditioning signal. This ensures the model learns to use the
    cross-attention pathway.

    Parameters
    ----------
    conditioning : dict[str, float]
        Conditioning values.
    dim : int
        Dimensionality of the user state embedding (default 64).

    Returns
    -------
    torch.Tensor
        Shape ``[dim]``, float32.
    """
    cond_tensor = build_conditioning_tensor(conditioning)  # [8]

    # Tile to fill 64 dims (8 * 8 = 64)
    tiled = cond_tensor.repeat(dim // 8)

    # Add small Gaussian noise for diversity
    noise = torch.randn_like(tiled) * 0.1

    return (tiled + noise).clamp(0.0, 1.0)


def prepare_dataset(
    triples: list[dict[str, Any]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert triples into tokenized, padded tensors with conditioning.

    For each triple, the full sequence is: ``[BOS] history [SEP] response [EOS]``.
    Both ``input_ids`` and ``target_ids`` are the same sequence (teacher forcing),
    with the loss computed only on the response portion via cross-entropy
    with shifted targets.

    Parameters
    ----------
    triples : list[dict]
        From ``extract_triples()``.
    tokenizer : SimpleTokenizer
        Tokenizer with built vocabulary.
    max_seq_len : int
        Maximum sequence length (pad/truncate to this).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
        - ``"input_ids"`` -- ``[N, max_seq_len]``
        - ``"target_ids"`` -- ``[N, max_seq_len]``
        - ``"conditioning"`` -- ``[N, 8]``
        - ``"user_state"`` -- ``[N, 64]``
    """
    all_input_ids: list[list[int]] = []
    all_conditioning: list[torch.Tensor] = []
    all_user_state: list[torch.Tensor] = []

    for triple in triples:
        # Build the full sequence text
        full_text = triple["history"] + " [SEP] " + triple["response"]

        # Encode with special tokens (BOS, EOS)
        ids = tokenizer.encode(
            full_text,
            add_special=True,
            max_length=max_seq_len,
            padding=True,
        )

        all_input_ids.append(ids)

        # Derive conditioning from response
        cond = derive_conditioning(triple["response"], triple["emotion"])
        all_conditioning.append(build_conditioning_tensor(cond))
        all_user_state.append(build_synthetic_user_state(cond))

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    target_ids = input_ids.clone()  # For causal LM, target = input (shifted in loss)
    conditioning = torch.stack(all_conditioning)
    user_state = torch.stack(all_user_state)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "conditioning": conditioning,
        "user_state": user_state,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/processed/dialogue",
    vocab_size: int = 8000,
    max_seq_len: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_dialogues: int = 0,
    seed: int = 42,
) -> dict[str, Path]:
    """Run the full data preparation pipeline.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw dialogue datasets.
    output_dir : str
        Output directory for processed ``.pt`` files.
    vocab_size : int
        Vocabulary size for the tokenizer.
    max_seq_len : int
        Maximum sequence length.
    val_ratio : float
        Fraction of data for validation.
    test_ratio : float
        Fraction of data for testing.
    max_dialogues : int
        Maximum dialogues to use (0 = all).

    Returns
    -------
    dict[str, Path]
        Mapping from split name to saved file path.
    """
    # SEC: resolve all paths so log messages are absolute and consistent.
    raw_dir = Path(raw_data_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load raw data ---
    logger.info("Step 1: Loading raw dialogue data...")
    dialogues: list[dict[str, Any]] = []

    # Try DailyDialog
    dd_dir = raw_dir / "dailydialog"
    dd_data = load_dailydialog(dd_dir)
    dialogues.extend(dd_data)

    # Try EmpatheticDialogues
    ed_dir = raw_dir / "empathetic_dialogues"
    ed_data = load_empathetic_dialogues(ed_dir)
    dialogues.extend(ed_data)

    # Fallback to synthetic if no real data found
    if not dialogues:
        logger.warning(
            "No real dialogue data found. Generating synthetic fallback."
        )
        dialogues = generate_synthetic_dialogues(n_dialogues=5000, seed=seed)

    if max_dialogues > 0:
        dialogues = dialogues[:max_dialogues]

    logger.info("Total dialogues: %d", len(dialogues))

    # --- Step 2: Extract triples ---
    logger.info("Step 2: Extracting (history, response, emotion) triples...")
    triples = extract_triples(dialogues)

    # --- Step 3: Build tokenizer vocabulary ---
    logger.info("Step 3: Building tokenizer vocabulary (size=%d)...", vocab_size)
    all_texts = [t["history"] + " " + t["response"] for t in triples]
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(all_texts)
    logger.info("Tokenizer built: %d tokens", len(tokenizer))

    # Save tokenizer
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    logger.info("Tokenizer saved to: %s", tok_path)

    # --- Step 4: Tokenize and build dataset ---
    logger.info("Step 4: Tokenizing and building dataset (max_seq_len=%d)...", max_seq_len)
    dataset = prepare_dataset(triples, tokenizer, max_seq_len=max_seq_len)

    # --- Step 5: Split into train/val/test ---
    logger.info("Step 5: Splitting into train/val/test...")
    n_total = len(dataset["input_ids"])
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    # SEC: shuffle deterministically with the user-supplied seed so that
    # train/val/test splits are reproducible from the CLI flag.
    torch.manual_seed(seed)
    perm = torch.randperm(n_total)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    splits: dict[str, dict[str, torch.Tensor]] = {
        "train": {k: v[train_idx] for k, v in dataset.items()},
        "val": {k: v[val_idx] for k, v in dataset.items()},
        "test": {k: v[test_idx] for k, v in dataset.items()},
    }

    # --- Step 6: Save ---
    logger.info("Step 6: Saving processed data...")
    saved_paths: dict[str, Path] = {}

    for split_name, split_data in splits.items():
        path = out_dir / f"{split_name}.pt"
        torch.save(split_data, path)
        n_samples = len(split_data["input_ids"])
        logger.info("  %s: %d samples -> %s", split_name, n_samples, path)
        saved_paths[split_name] = path

    # --- Summary statistics ---
    logger.info("=" * 50)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 50)
    logger.info("Total samples:     %d", n_total)
    logger.info("Train samples:     %d", n_train)
    logger.info("Val samples:       %d", n_val)
    logger.info("Test samples:      %d", n_test)
    logger.info("Vocab size:        %d", len(tokenizer))
    logger.info("Max seq length:    %d", max_seq_len)
    logger.info("Output directory:  %s", out_dir)

    # Log conditioning distribution stats
    cond = splits["train"]["conditioning"]
    for i, name in enumerate([
        "cognitive_load", "formality", "verbosity", "emotionality",
        "directness", "emotional_tone", "accessibility", "reserved",
    ]):
        vals = cond[:, i]
        logger.info(
            "  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            name, vals.mean().item(), vals.std().item(),
            vals.min().item(), vals.max().item(),
        )

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dialogue datasets for SLM training.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw dialogue datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/dialogue",
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size for the tokenizer (default: 8000).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=0,
        help="Maximum dialogues to use, 0=all (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic fallback + split shuffle (default: 42).",
    )
    args = parser.parse_args()

    # SEC: bound-check arguments before any heavy work runs.
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be a positive integer")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be a positive integer")
    if not (0.0 <= args.val_ratio < 1.0):
        parser.error("--val-ratio must be in [0, 1)")
    if not (0.0 <= args.test_ratio < 1.0):
        parser.error("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("--val-ratio + --test-ratio must be < 1.0")
    if args.max_dialogues < 0:
        parser.error("--max-dialogues must be non-negative (0 = all)")
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    args = parse_args()
    run_pipeline(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_dialogues=args.max_dialogues,
        seed=args.seed,
    )

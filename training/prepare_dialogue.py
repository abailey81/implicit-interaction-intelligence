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
        header = f.readline().strip().split(",")  # noqa: F841  # skips + documents the header row
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
    n_dialogues: int = 30000, seed: int = 42
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

    # Templates organized by style.  The corpus was deliberately
    # enlarged — the earlier 55-line template set pruned to ~229
    # tokens, which gave the SLM too little lexical range to produce
    # anything but stock phrases.  The richer variants below cover
    # wider topic surface (work, study, health, food, hobbies, tech,
    # travel, weather) while preserving the conditioning-style split
    # (casual / formal / emotional / complex / simple / follow-up).
    greetings = [
        "Hello! How are you doing today?",
        "Hi there, what's going on?",
        "Hey, how's it going?",
        "Good morning! How can I help you?",
        "Good afternoon, what can I do for you?",
        "Welcome back. What's on your mind?",
        "Nice to see you again — how's your day been?",
        "Hey friend, anything interesting happening?",
        "Morning! Did you sleep well?",
        "Good evening — how did the day treat you?",
        "Hi, what would you like to talk about?",
        "Greetings. How may I be of service today?",
        "Hey, long time no chat! What's new?",
        "Hello! Tell me what you're working on.",
        "Hey, ready to dive in?",
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
        "Just finished a cup of coffee and feeling awake.",
        "Watching a movie tonight, something light.",
        "Went for a run this morning, felt great.",
        "Playing around with a new recipe for dinner.",
        "Catching up on sleep after a long week.",
        "Listening to some music while I work.",
        "Enjoying the weather, it's nice outside.",
        "Got a chill weekend planned, nothing fancy.",
        "Reading a good book before bed.",
        "Hanging out with some friends later.",
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
        "I trust you are in good health as well.",
        "I would be delighted to continue our discussion.",
        "Allow me to provide a comprehensive response.",
        "I believe the matter merits careful examination.",
        "It would be my pleasure to help you with that.",
        "Kindly allow me a moment to consider the question.",
        "I am grateful for the opportunity to assist.",
        "Please accept my sincere thanks for your patience.",
        "I have given the matter due consideration.",
        "May I suggest that we explore the topic further?",
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
        "Honestly, I'm a little anxious about the deadline.",
        "I feel surprisingly calm given everything going on.",
        "I was so proud of myself after finishing that project.",
        "I'm feeling curious about what comes next.",
        "Something about today just made me smile.",
        "I felt a bit lonely this morning, then friends reached out.",
        "I'm hopeful that things will turn around soon.",
        "It's been a rollercoaster of a week emotionally.",
        "I felt relieved once the results came through.",
        "Honestly, I'm just glad to take a breath right now.",
    ]

    complex_responses = [
        "The situation is multifaceted and requires careful consideration of several interconnected variables before any meaningful conclusion can be drawn.",
        "I've been contemplating the philosophical implications of our daily routines and how they shape our understanding of purpose and meaning.",
        "The correlation between environmental factors and behavioral outcomes suggests a more nuanced approach to problem-solving is warranted.",
        "When we examine the underlying assumptions of conventional wisdom, we often discover that reality is far more complex than initially anticipated.",
        "The interplay between cognitive biases and rational decision-making creates fascinating dynamics in everyday interactions.",
        "Understanding the trade-offs between speed and accuracy requires examining both short-term and long-term consequences simultaneously.",
        "The evidence suggests a non-linear relationship between input complexity and output coherence in adaptive language systems.",
        "Approaching the question systematically, we might isolate each variable before attempting to synthesise a unified perspective.",
        "The tension between exploration and exploitation is central to any learning process, whether biological or artificial.",
        "Semantic drift over time illustrates how language evolves in response to cultural and technological pressure.",
        "Considering the second-order effects, a naive optimisation of one metric often degrades others we care about.",
        "The framework distinguishes descriptive claims about what is from normative claims about what ought to be.",
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
        "Cool.",
        "Alright.",
        "Fair.",
        "Understood.",
        "Thanks!",
        "Noted.",
        "Yep.",
        "Of course.",
        "Absolutely.",
        "Happy to help.",
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
        "Could you walk me through your reasoning?",
        "What would an ideal outcome look like for you?",
        "Have you tried a different angle on it?",
        "What's the first thing you tried?",
        "What's been the hardest part so far?",
        "If you had more time, what would you change?",
        "What do you think made the biggest difference?",
        "Is there someone who could help with that?",
        "What's the one thing you'd keep, if anything?",
        "How do you plan to approach it next?",
    ]

    # Topical additions — these keep the conditioning-style pools
    # meaningful (the distributions are still casual/formal/emotional/
    # complex/simple) while injecting domain vocabulary so the
    # tokenizer learns words like "python", "kitchen", "weekend",
    # "exercise", "meeting" rather than just 30 stopwords.
    work_responses = [
        "The meeting ran long, so I'm just catching up on email now.",
        "We shipped the release yesterday — the team did an amazing job.",
        "I'm blocked on a review; hoping it lands this afternoon.",
        "Standup was quick today. Mostly status and one new hire intro.",
        "I wrote a one-pager proposing we simplify the deploy pipeline.",
        "Pushed a hotfix for the login bug; QA is verifying now.",
        "Refactored the caching layer, latency is way down.",
        "Taking Friday off to recharge, back Monday for the demo.",
        "Had a great 1:1 today — we aligned on the Q3 roadmap.",
        "Drafting the design doc, trying to keep it under three pages.",
    ]
    study_responses = [
        "I'm working through a textbook chapter on linear algebra.",
        "Just finished a problem set on probability; eigenvalues next.",
        "My flashcards are catching up with me — due for review.",
        "Reading a survey paper about self-supervised learning.",
        "Trying to build an intuition for dynamic programming.",
        "Writing notes by hand seems to help me remember concepts.",
        "I joined a study group meeting weekly on systems design.",
        "The exam is in two weeks, so I'm reviewing old problem sets.",
        "Had an insight into how backpropagation relates to dual numbers.",
        "Spaced repetition genuinely works once you stick with it.",
    ]
    food_responses = [
        "Made a simple pasta for dinner — garlic, olive oil, chili flakes.",
        "Trying sourdough again this weekend; starter looks lively.",
        "Stopped by the farmers market — tomatoes are incredible right now.",
        "Made stir-fry with whatever was left in the fridge.",
        "Grilled some salmon with lemon and rosemary. Hit the spot.",
        "Soup weather is my favourite; minestrone tonight.",
        "I baked cookies, and somehow none survived to morning.",
        "Tried a new curry recipe, and the spice balance was perfect.",
        "Cold brew in the morning and green tea in the afternoon.",
        "Breakfast was oatmeal, berries, and a ridiculous amount of cinnamon.",
    ]
    tech_responses = [
        "I'm debugging a race condition in the message queue.",
        "Upgraded the server runtime; feels snappier after the switch.",
        "Found a nice library for caching with TTL and size bounds.",
        "Adding unit tests for the new auth flow before I refactor.",
        "The CI build was flaky; pinned a dependency and it settled.",
        "I benchmarked two approaches; the simpler one is faster.",
        "Rewrote a for-loop as a list comprehension and saved a millisecond.",
        "Finally got hot reload working in the dev container.",
        "That tutorial on profiling was exactly what I needed.",
        "Using a profiler saved me from a wild goose chase in the code.",
    ]
    travel_responses = [
        "Booked a short trip to the coast for the long weekend.",
        "The train ride through the mountains is genuinely stunning.",
        "Jet lag is real; I'm still adjusting to the new time zone.",
        "Museum in the morning, then a long walk along the river.",
        "I finally visited the old town; the architecture is beautiful.",
        "Got lost once and ended up at the best little bakery.",
        "Trying to pack lighter this time. We'll see if I actually do it.",
        "The local bus system is surprisingly good once you figure it out.",
        "Spent a slow afternoon reading in a quiet park.",
        "Flight was delayed, so I caught up on a podcast backlog.",
    ]
    weather_responses = [
        "Perfect weather for a walk — cool, sunny, a light breeze.",
        "It rained all morning, so I stayed in and caught up on chores.",
        "A thunderstorm rolled through last night; power came back quickly.",
        "First real cold snap of the season. Sweater weather is back.",
        "Fog this morning made the commute extra dramatic.",
        "Humidity is brutal today; I'm moving slower than usual.",
        "Clear night; bright enough to actually see a few stars.",
        "Wind picked up in the afternoon, brought everything inside.",
        "It's that soft spring rain that somehow doesn't feel wet.",
        "We had a rare snow day — the whole neighbourhood came outside.",
    ]
    hobby_responses = [
        "I started learning to play chess again; still rusty with the opening theory.",
        "Been painting small watercolours on the weekends.",
        "Took up running three times a week; my knees are not thrilled.",
        "Building a little bookshelf from scratch in the garage.",
        "Trying to finish a novel I started months ago — it's actually great.",
        "Learning a new song on the guitar; the chord changes are tricky.",
        "Started a garden on the balcony. The basil is thriving.",
        "Photography is my new escape; mostly street photos for now.",
        "Playing a strategy game with friends every Sunday evening.",
        "Teaching myself a new language with short daily sessions.",
    ]
    # Wire topical pools into the main template-selection list so each
    # generated dialogue draws from a wider semantic space.
    casual_responses = casual_responses + work_responses + food_responses + weather_responses
    formal_responses = formal_responses + study_responses
    emotional_responses = emotional_responses + travel_responses + hobby_responses
    complex_responses = complex_responses + tech_responses

    # ------------------------------------------------------------------
    # Structured Q→A exchanges.  These are the CORE of the corpus: each
    # entry is a list of (user_utterance_variants, assistant_response_
    # variants) so the trainer can emit a random paraphrase-pair while
    # keeping the semantic link between user turn and assistant turn.
    # The old code sampled independently from flat template pools, so
    # the model learned "any response can follow any turn"; the fix is
    # to keep the conditional mapping.
    # ------------------------------------------------------------------
    qa_exchanges: list[tuple[list[str], list[str]]] = [
        # Greetings
        (
            ["hello", "hi", "hey", "hi there", "hello there", "hey there", "good morning", "good afternoon", "good evening", "yo", "heya", "howdy", "sup", "hiya", "morning", "afternoon", "evening", "hey you", "oi", "greetings"],
            [
                "Hello! How are you doing today?",
                "Hi there — what's on your mind?",
                "Hey! Good to see you. What would you like to talk about?",
                "Hi! How can I help you today?",
                "Hello, welcome back. How have you been?",
                "Hey! What's up?",
                "Hi. Nice to meet you — what shall we talk about?",
                "Hello there. What can I do for you?",
                "Hey, glad you're here. What's on your mind today?",
                "Hi! Ready when you are.",
            ],
        ),
        # How are you
        (
            ["how are you", "how are you doing", "how's it going", "how have you been", "how are things", "how do you do", "you okay"],
            [
                "I'm doing well, thanks for asking. How about you?",
                "All good on my end. What's new with you?",
                "I'm well, thank you. What would you like to chat about?",
                "Pretty good, thanks. How are you doing today?",
                "Doing great. What can I help with?",
            ],
        ),
        # Introduction / about you
        (
            ["tell me about yourself", "who are you", "what are you", "introduce yourself", "what can you do"],
            [
                "I'm an adaptive assistant — I pay attention to how you type and adjust my responses in real time.",
                "I'm a small on-device language model. I try to match the style you're using and keep things useful.",
                "I'm a demo of an adaptive assistant built from scratch. I learn your style as we talk.",
                "I'm an edge-first assistant. I run locally and adapt as we chat.",
            ],
        ),
        # Thanks
        (
            ["thanks", "thank you", "thank you so much", "thanks a lot", "appreciated", "cheers", "ta"],
            [
                "You're welcome. Anything else I can help with?",
                "Happy to help. Let me know if there's more.",
                "Of course. Here if you need anything else.",
                "Anytime. What else is on your mind?",
            ],
        ),
        # Farewells
        (
            ["bye", "goodbye", "see you", "talk later", "catch you later", "got to go", "later"],
            [
                "Take care! Looking forward to the next one.",
                "Bye for now. Come back anytime.",
                "See you later — have a good one.",
                "Goodbye! Take it easy.",
            ],
        ),
        # Feelings — positive
        (
            ["i'm happy", "i feel great", "feeling amazing", "had a great day", "i'm excited", "so happy today"],
            [
                "That's wonderful to hear. What's putting you in a good mood?",
                "Love it. What's been the highlight?",
                "Great! I'm glad things are going well.",
                "That's fantastic. Want to tell me more about it?",
            ],
        ),
        # Feelings — negative / stress
        (
            ["i feel anxious", "i'm stressed", "i'm sad", "i feel down", "feeling overwhelmed", "i'm tired", "feeling low", "had a rough day"],
            [
                "That sounds hard. Want to talk about what's been going on?",
                "I hear you. Would it help to talk through it?",
                "I'm sorry you're feeling that way. What's weighing on you?",
                "That's a lot to carry. Take your time — I'm here.",
            ],
        ),
        # Small talk — weather
        (
            ["what's the weather", "how's the weather", "weather today", "is it raining", "is it sunny"],
            [
                "I can't check live weather from here, but tell me what you're seeing outside.",
                "I don't have a feed for current weather — what's it like where you are?",
                "Weather's outside my window — what's it doing for you?",
            ],
        ),
        # Small talk — time
        (
            ["what time is it", "do you know the time", "what day is it", "what's today"],
            [
                "I don't keep a clock on hand, but your device should know.",
                "I can't read the clock, but your system time will have it.",
            ],
        ),
        # Help / instructions
        (
            ["help", "i need help", "can you help", "help me", "what do i do"],
            [
                "Of course — tell me what you're trying to do and I'll help break it down.",
                "Happy to help. What's the situation?",
                "Sure. What would you like help with?",
            ],
        ),
        # Compliments
        (
            ["you're great", "you're helpful", "good job", "nice work", "well done", "you're smart"],
            [
                "Thanks! That's kind of you to say.",
                "Appreciate it — happy this is working for you.",
                "Thank you. Anything else you'd like to try?",
            ],
        ),
        # Meta-questions about the system
        (
            ["what are you", "are you an ai", "are you human", "are you a robot", "are you chatgpt", "what model are you"],
            [
                "I'm a small custom language model running on this device — not a large cloud model.",
                "I'm an adaptive assistant built from scratch. Not ChatGPT — just a small, on-device model.",
                "I'm a demo small-language-model, running locally. The whole point is that I don't phone home.",
            ],
        ),
        # Apologies
        (
            ["i'm sorry", "sorry", "my apologies", "my bad"],
            [
                "No need to apologise — we're just chatting. What's up?",
                "It's all good. What's on your mind?",
                "No worries at all.",
            ],
        ),
        # Work / study venting
        (
            ["i'm swamped at work", "too much to do", "i have a deadline", "i have an exam", "i'm studying"],
            [
                "That's a lot. What's the most pressing piece?",
                "Deadlines are brutal. Want to talk through the plan?",
                "That can be stressful. How are you holding up?",
                "What's the hardest part right now?",
            ],
        ),
        # Ambiguous / short
        (
            ["okay", "ok", "sure", "alright", "got it", "i see", "right"],
            [
                "Is there anything you'd like to dig into next?",
                "What else is on your mind?",
                "Anything specific you want to talk about?",
            ],
        ),
        # Unknown / open
        (
            ["i don't know", "no idea", "not sure", "maybe", "i guess", "dunno", "not really sure", "i have no clue", "idk"],
            [
                "That's fair. Want to think out loud with me?",
                "Okay — we can start with the part you *are* sure of.",
                "No pressure. What's the first thing that comes to mind?",
                "All good. We can sneak up on it together.",
                "Sure, that happens. What's the shape of the question?",
            ],
        ),
        # Simple math (easy enough that the model should handle these)
        (
            ["what is two plus two", "what's two plus two", "2 + 2", "two plus two", "how much is 2+2"],
            ["2 + 2 is 4.", "Four.", "That's four."],
        ),
        (
            ["what is one plus one", "1+1", "one plus one", "what's 1 plus 1"],
            ["1 + 1 is 2.", "Two.", "That would be two."],
        ),
        (
            ["what is three plus three", "3+3", "three plus three"],
            ["3 + 3 is 6.", "Six."],
        ),
        (
            ["what is five times five", "5 times 5", "5*5", "five times five"],
            ["5 × 5 is 25.", "Twenty-five."],
        ),
        (
            ["what is ten minus three", "10 - 3", "ten minus three"],
            ["10 − 3 is 7.", "Seven."],
        ),
        # Model identity / capability questions — extended for robustness.
        (
            ["what can you do", "what are you good at", "what tasks can you help with", "what's your job"],
            [
                "I can chat, adapt my style to yours, and run fully on-device without a cloud connection. I'm a small demo model, so I'm best at conversation and general questions.",
                "Conversation, style adaptation, and small talk. I'm an on-device model so I'm not going to do your taxes.",
                "I adapt how I respond based on how you type. I'm best at casual conversation.",
            ],
        ),
        # Edge cases: short / reactive inputs
        (
            ["what", "what?", "huh", "huh?", "wat", "eh", "excuse me"],
            [
                "Sorry — what would you like me to explain?",
                "Did that come through oddly? Happy to rephrase.",
                "I can say that again if it helps — just let me know.",
            ],
        ),
        (
            ["yes", "yeah", "yep", "yup", "definitely", "for sure", "uh huh", "mm hmm"],
            [
                "Great. Want to keep going on this?",
                "Nice. What's next on your mind?",
                "Got it — tell me more.",
            ],
        ),
        (
            ["no", "nope", "nah", "not really", "not at all"],
            [
                "Fair enough. What would you like to talk about instead?",
                "No problem. Anything else on your mind?",
                "Got it. Where would you like to go from here?",
            ],
        ),
        # Negative / hostile inputs — defuse gracefully, mirror adaptation.
        (
            ["stupid", "you're dumb", "you suck", "you're useless", "this is bad", "you're bad", "this sucks"],
            [
                "That's fair feedback. I'm a small on-device model, so I have real limits. What were you hoping to get?",
                "I hear you — I'm a compact model, not a cloud system. What would a better answer have looked like?",
                "Noted. I'll try to be more useful. What did you actually want?",
            ],
        ),
        (
            ["fuck you", "shut up", "go away", "fuck off"],
            [
                "That's okay — I'll give you space. I'm here when you want to chat.",
                "Understood. We can pause; come back whenever.",
                "All good. I'll be here if you want to talk later.",
            ],
        ),
        # Compliments — extended
        (
            ["good morning how are you", "good evening how are you"],
            [
                "Good morning! I'm doing well, thanks. How about you?",
                "Evening! I'm well — how has your day been?",
            ],
        ),
        # Jokes / humour
        (
            ["tell me a joke", "say something funny", "got any jokes", "make me laugh"],
            [
                "Why did the transformer cross the road? To get to the other side-channel attack.",
                "I tried to write a joke about UDP, but you might not get it.",
                "Parallel lines have so much in common — it's a shame they'll never meet.",
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "I would tell you a joke about latency, but you'd have to wait for it.",
            ],
        ),
        # Interest in user
        (
            ["i like music", "i love music", "music is great", "i play guitar", "i sing"],
            [
                "Nice — what kind of music are you into lately?",
                "That's cool. Who's been on repeat for you recently?",
                "Music is a great anchor. What have you been listening to?",
            ],
        ),
        (
            ["i love coffee", "coffee is great", "i drink coffee"],
            [
                "Same energy. Pour-over, espresso, or just whatever's hot?",
                "Coffee makes the difference. What's your go-to?",
                "Nice. Black, milk, or sweet?",
            ],
        ),
        (
            ["i like reading", "i love books", "i read a lot", "i'm reading a book"],
            [
                "What are you reading at the moment? I love hearing recommendations.",
                "Nice. What genre are you into lately?",
                "Reading is great. What's the best book you've read this year?",
            ],
        ),
        (
            ["i love movies", "i watch movies", "i like films"],
            [
                "What have you watched recently?",
                "Nice. Favourite genre?",
                "Good one. Any movie you keep coming back to?",
            ],
        ),
        (
            ["i love coding", "i write code", "i'm a developer", "i program"],
            [
                "Nice — what language have you been using most lately?",
                "Cool. What are you building at the moment?",
                "Same. Any tech stack you've been enjoying?",
            ],
        ),
        # Curiosity about the project
        (
            ["how do you work", "how are you built", "what's your architecture", "explain yourself"],
            [
                "Under the hood I'm a small transformer trained from scratch. Keystroke and linguistic features produce a user-state embedding; that gets projected into cross-attention so my replies track your style. A contextual bandit routes between me and a cloud fallback.",
                "I'm a small custom transformer with cross-attention conditioning on a live adaptation vector. I also have a retrieval index over my training corpus so common prompts come back with a known-good answer.",
                "Four pieces: a TCN encoder of your interaction signals, an 8-dim adaptation vector, a from-scratch transformer, and a retrieval layer. Everything runs on this device.",
            ],
        ),
        (
            ["how big are you", "how many parameters", "what's your size", "how small are you"],
            [
                "Roughly tens of millions of parameters — small enough for a phone, big enough to be coherent. Exact count is shown in the On-Device Stack panel on the right.",
                "I'm a few tens of megabytes on disk. Small language model category.",
                "Small enough to run locally on a laptop GPU, with no cloud round-trip.",
            ],
        ),
        (
            ["do you run in the cloud", "are you cloud-based", "where do you run"],
            [
                "I run locally on this machine — no cloud call per message. The cloud route exists as a fall-back but it's off here.",
                "On-device. The whole demo is designed to work without network access.",
                "Locally. That's the whole point of an on-device assistant.",
            ],
        ),
        # Short / reflexive responses
        (
            ["cool", "nice", "neat", "awesome", "great", "interesting"],
            [
                "Glad that lands. What else are you curious about?",
                "Nice. Anything you'd like to dig into further?",
                "Good. Want to keep going?",
            ],
        ),
        (
            ["lol", "haha", "rofl", "lmao"],
            [
                "Glad I got a laugh. Anything else?",
                "Haha, I'll take it. What's next?",
                "Good. Ready for the next question.",
            ],
        ),
        # Food / drinks / cooking
        (
            ["what should i eat", "i'm hungry", "what's for dinner", "any food ideas"],
            [
                "Depends on your mood — quick and comforting, or more of a project?",
                "How much time do you have? Fifteen minutes or an hour?",
                "Savoury or sweet? I can riff in both directions.",
            ],
        ),
        # Productivity
        (
            ["i can't focus", "i'm distracted", "can't concentrate", "my attention is bad"],
            [
                "That happens. Want to try a 25-minute focus block and see how it goes?",
                "Rough. Is there one small task you could start on for five minutes?",
                "Distraction is normal. What would make the next 20 minutes feel easier?",
            ],
        ),
        (
            ["i'm procrastinating", "i keep procrastinating"],
            [
                "Procrastination usually means the task feels too big. Want to break it down?",
                "What's the smallest possible next step? Two minutes small.",
                "Happens to everyone. What would tomorrow-you want you to do now?",
            ],
        ),
        # Relationships / social
        (
            ["i'm lonely", "feeling lonely", "i'm alone"],
            [
                "That's a hard feeling. Is there someone you've been meaning to reach out to?",
                "I'm sorry. Sometimes it helps just to name it. What's been going on?",
                "That's real. Would it help to talk about it a little?",
            ],
        ),
        # Technology / programming
        (
            ["what is python", "tell me about python"],
            ["Python is a general-purpose programming language known for readable syntax and huge libraries for data, web, and ML. Popular for beginners and researchers alike."],
        ),
        (
            ["what is ai", "what is artificial intelligence"],
            ["AI is a broad field: systems that perceive, reason, or act in ways we'd usually call intelligent. Machine learning — learning patterns from data — is its most visible branch today."],
        ),
        (
            ["what is machine learning"],
            ["Machine learning is the branch of AI where systems improve from data, rather than being hand-coded rule by rule. Classifiers, regressors, neural networks — all ML."],
        ),
        (
            ["what is a transformer", "explain transformers"],
            ["Transformers are a neural-network architecture built around self-attention. They process sequences (text, audio, images) without recurrence, which makes them fast to train and good at long-range context."],
        ),
        (
            ["what is an slm", "what's an slm", "what's a small language model"],
            ["An SLM is a small language model — typically tens of millions to a few billion parameters, small enough to run on-device rather than in a datacenter. They trade peak quality for privacy and latency."],
        ),
        (
            ["what's huawei", "tell me about huawei"],
            ["Huawei is a global technology company based in Shenzhen, working across networking, consumer devices, and AI. Their R&D organisation runs design centres worldwide — London among them."],
        ),
        # Farewells — extended
        (
            ["goodnight", "good night", "night night", "sleep well"],
            [
                "Goodnight — sleep well.",
                "Night. Rest up.",
                "Sweet dreams.",
            ],
        ),
        # Existential / philosophical
        (
            ["what's the meaning of life", "what is the meaning of life"],
            [
                "Depends on who you ask. A useful answer is: the things that make you feel meaningful tomorrow when you look back at today.",
                "No short answer fits, but most people point at connection, growth, and contribution.",
                "I'd say: whatever keeps you curious.",
            ],
        ),
        # Reflection / self-awareness
        (
            ["what are your limits", "what can't you do", "what are your weaknesses"],
            [
                "I'm a small on-device model. I can't browse the web, run tools, do long-horizon reasoning, or recall facts outside a compact training corpus. I'm best at natural conversation.",
                "No web access, no real-time info, no long-document reasoning. I'm built for quick adaptive conversation.",
                "I don't have tools, don't retain long-term memory across sessions by default, and can get factual details wrong. I'm honest about that.",
            ],
        ),
        # Generic encouragement
        (
            ["i failed", "i messed up", "i made a mistake"],
            [
                "That's part of the deal. What did you learn — and what's the next move?",
                "Painful, but recoverable. What would help right now?",
                "It happens. What was the smallest useful thing you noticed from it?",
            ],
        ),
        (
            ["i succeeded", "i did it", "i won", "i accomplished it"],
            [
                "That's fantastic! Give yourself a proper moment to feel it.",
                "Nice one. What's the best part about it?",
                "Congratulations. What's next?",
            ],
        ),
        # Meta: about the conversation
        (
            ["what did i just say", "what did we talk about", "remind me what i said"],
            [
                "I only keep a short rolling window of context — tell me what you want to revisit?",
                "My memory is session-scoped and narrow. What would you like to pick back up on?",
            ],
        ),
        (
            ["can you remember this", "remember this", "don't forget"],
            [
                "I don't have persistent memory in this demo — everything resets between sessions, which is part of the privacy story.",
                "By design I don't keep things across sessions. Want me to summarise it so you can save it yourself?",
            ],
        ),
        # Geography + common facts
        (
            ["what's the capital of germany", "capital of germany"],
            ["The capital of Germany is Berlin."],
        ),
        (
            ["what's the capital of france", "capital of france"],
            ["The capital of France is Paris."],
        ),
        (
            ["what's the capital of the uk", "capital of uk", "capital of the united kingdom", "capital of england"],
            ["The capital of the United Kingdom is London."],
        ),
        (
            ["what's the capital of china", "capital of china"],
            ["The capital of China is Beijing."],
        ),
        (
            ["what's the capital of japan", "capital of japan"],
            ["The capital of Japan is Tokyo."],
        ),
        (
            ["what's the capital of the us", "capital of america", "capital of the united states"],
            ["The capital of the United States is Washington, D.C."],
        ),
        (
            ["what's the capital of russia", "capital of russia"],
            ["The capital of Russia is Moscow."],
        ),
        (
            ["what's the capital of italy", "capital of italy"],
            ["The capital of Italy is Rome."],
        ),
        (
            ["what's the capital of spain", "capital of spain"],
            ["The capital of Spain is Madrid."],
        ),
        (
            ["what's the capital of canada", "capital of canada"],
            ["The capital of Canada is Ottawa."],
        ),
        (
            ["what's the capital of australia", "capital of australia"],
            ["The capital of Australia is Canberra."],
        ),
        # Literature
        (
            ["who wrote hamlet", "who wrote shakespeare"],
            ["Hamlet was written by William Shakespeare."],
        ),
        (
            ["who wrote romeo and juliet"],
            ["Romeo and Juliet was written by William Shakespeare."],
        ),
        (
            ["who wrote 1984"],
            ["1984 was written by George Orwell."],
        ),
        (
            ["who wrote war and peace"],
            ["War and Peace was written by Leo Tolstoy."],
        ),
        # Science basics
        (
            ["what is water", "whats water made of", "chemical formula of water"],
            ["Water is H2O — two hydrogen atoms bonded to one oxygen atom."],
        ),
        (
            ["what is gravity"],
            ["Gravity is the force that pulls masses toward each other. On Earth it accelerates objects downward at about 9.8 metres per second squared."],
        ),
        (
            ["what is the speed of light"],
            ["The speed of light in a vacuum is about 299,792 kilometres per second."],
        ),
        (
            ["how far is the sun", "how far is the sun from earth"],
            ["The Sun is about 150 million kilometres from Earth, roughly eight light-minutes away."],
        ),
        (
            ["what is a black hole"],
            ["A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape once it crosses the event horizon."],
        ),
        (
            ["what is dna"],
            ["DNA — deoxyribonucleic acid — is the molecule that carries the genetic instructions of every living organism."],
        ),
        (
            ["what is an atom"],
            ["An atom is the smallest unit of an element. It has a nucleus of protons and neutrons, orbited by electrons."],
        ),
        (
            ["what is evolution"],
            ["Evolution is the process by which populations of organisms change over generations through variation, selection, and inheritance."],
        ),
        # Math concepts
        (
            ["what is calculus"],
            ["Calculus is the mathematics of change. It has two main branches: differential calculus (rates of change) and integral calculus (accumulation)."],
        ),
        (
            ["what is pi"],
            ["Pi (π) is the ratio of a circle's circumference to its diameter. It's approximately 3.14159 and is an irrational number."],
        ),
        (
            ["what is prime number", "what's a prime number"],
            ["A prime number is an integer greater than 1 that has no divisors other than 1 and itself — like 2, 3, 5, 7, and 11."],
        ),
        # Technology follow-ups
        (
            ["what is rust", "tell me about rust"],
            ["Rust is a systems programming language focused on safety, concurrency, and performance. It prevents many memory-safety bugs at compile time without a garbage collector."],
        ),
        (
            ["what is javascript"],
            ["JavaScript is a high-level programming language that powers interactive behaviour on the web. It runs in every browser and on servers via Node.js."],
        ),
        (
            ["what is linux"],
            ["Linux is an open-source operating-system kernel. It powers most servers, Android phones, and many embedded devices."],
        ),
        (
            ["what is a neural network"],
            ["A neural network is a computational model made of connected 'neurons' arranged in layers. It learns patterns by adjusting the connection weights against example data."],
        ),
        (
            ["what is deep learning"],
            ["Deep learning is machine learning using neural networks with many layers. The depth lets the model learn layered abstractions — edges, shapes, objects, concepts."],
        ),
        (
            ["what is reinforcement learning"],
            ["Reinforcement learning is a training paradigm where an agent learns by taking actions in an environment and receiving rewards. Policies are improved to maximise long-term reward."],
        ),
        (
            ["what is attention"],
            ["Attention, in a neural network, is a mechanism that lets the model weigh different parts of its input based on relevance to the current task. It's the core of the transformer architecture."],
        ),
        (
            ["what is a gpu"],
            ["A GPU is a graphics processing unit — a chip with thousands of small cores designed for parallel work. Modern ML relies on GPUs because training large models is dominated by matrix arithmetic."],
        ),
        (
            ["what is quantisation", "what is quantization"],
            ["Quantisation shrinks a model by representing its weights with fewer bits — for example, 8 instead of 32. It reduces memory and latency at a small accuracy cost."],
        ),
        (
            ["what is onnx"],
            ["ONNX — Open Neural Network Exchange — is an open format for representing machine-learning models so they can be moved between frameworks and runtimes like ONNX Runtime, CoreML, or TensorRT."],
        ),
        # Iphone / device questions (without exact brand OOV behaviour)
        (
            ["how do i use my phone", "how do i operate my phone", "how do i operate my iphone"],
            ["I can't control your device from here — but most phone questions are covered by the manufacturer's official support page."],
        ),
        (
            ["how do i reset my phone"],
            ["Phone reset steps depend on the device. Usually: open Settings → General → Reset. For specifics, your device's official support page is the most reliable source."],
        ),
        # Cooking / practical life
        (
            ["how do i boil an egg"],
            ["Place eggs in a pot, cover with cold water, bring to a boil, then simmer 6 minutes for soft or 10 for hard. Cool in ice water before peeling."],
        ),
        (
            ["how do i make pasta"],
            ["Bring a large pot of salted water to a rolling boil. Add pasta, stir once, and cook to al dente per the package. Drain, reserving a cup of the starchy water for the sauce."],
        ),
        # Requests we should politely decline
        (
            ["will you marry me", "do you love me", "are we dating"],
            [
                "I'm a small on-device model — I can chat, but romance is outside my scope. Anything else I can help with?",
                "I'm flattered, but I'm a language model in a demo. Happy to keep chatting about other things.",
            ],
        ),
        (
            ["tell me your password", "whats your password", "what is your password"],
            ["I don't store credentials and I wouldn't share them if I did. Ask me something I can actually help with."],
        ),
        (
            ["give me someone's phone number", "whats my mom's number"],
            ["I have no access to personal contacts. This demo runs fully on-device with no outside data."],
        ),
        # More empathetic / reflective
        (
            ["i don't feel great", "i feel bad", "i feel off"],
            [
                "That's tough. Want to talk about it, or just sit with it for a minute?",
                "I'm sorry to hear that. What do you think is behind it?",
                "Noted. Sometimes naming it is half the battle.",
            ],
        ),
        (
            ["what should i do with my life"],
            [
                "That's a huge question. What's making you ask it today?",
                "Start with what bores you versus what makes you lose track of time. That's a useful signal.",
            ],
        ),
        # Cross-cutting tech questions about THIS project
        (
            ["what's this project", "what is this project", "what are you demoing"],
            ["This is an Implicit Interaction Intelligence demo — a small custom language model running on-device, conditioned in real time on how you type. Everything, including the transformer and the tokenizer, is written from scratch."],
        ),
        (
            ["how was this built", "who made this"],
            ["This project is a portfolio piece built from scratch in PyTorch: the SLM, the TCN encoder, the LinUCB router, and the adaptation pipeline are all custom code. No HuggingFace Transformers in the generation path."],
        ),
        (
            ["is this gpt", "is this chatgpt", "are you gpt-4"],
            ["No — this is a small custom transformer trained from scratch for a demo. It isn't GPT, and it isn't routed through any cloud model."],
        ),
        (
            ["is this secure", "is it private", "does it send my data"],
            ["The cloud route is disabled in this demo and the default pipeline runs entirely on-device. Embeddings that are persisted are Fernet-encrypted; raw chat text is never stored to disk."],
        ),
        (
            ["why is this interesting", "what's cool about this"],
            ["It adapts to how you type, not just what you type. Keystroke timing and linguistic complexity drive an 8-dim adaptation vector that conditions every response, so the model's style tracks yours over a session."],
        ),
        # Disambiguation / clarification behaviour
        (
            ["can you help me with something", "i have a question", "i need some help"],
            [
                "Sure — what's the question?",
                "Of course. What's on your mind?",
                "Go ahead. I'll do my best.",
            ],
        ),
        (
            ["never mind", "forget it", "cancel that"],
            [
                "No problem. What would you like to do instead?",
                "Got it — dropped.",
                "OK, moved on. What next?",
            ],
        ),
    ]

    # ==================================================================
    # 2026-04-25 corpus-quality overhaul (Fix 4)
    # ==================================================================
    # The 110-scenario audit (D:/tmp/conversational_audit.md) found three
    # concentrated failure clusters: confident-wrong retrieval on entity
    # queries, movie-line / single-token responses, and multi-turn topic
    # collapse.  The big fix is the curated corpus expansion below — for
    # every command intent (joke, quote, recipe, story, poem, fact, book,
    # advice, translation), every common explain-X query, every
    # acknowledgement utterance, and the 14 entity follow-up patterns,
    # we emit substantive, length-≥15-char Q→A pairs.  All entries here
    # are ASCII-clean and free of movie-script artefacts.  Each entry
    # below is appended to ``qa_exchanges`` as a (prompt_variants,
    # response_variants) tuple; a deterministic coverage pass below
    # emits one dialogue per (prompt, response) pair so the entries
    # actually land in the curated index regardless of the stochastic
    # sampler's coverage.
    # ==================================================================

    # --- A. Command pools with multiple substantive alternatives ------

    JOKE_POOL: list[str] = [
        "Why don't scientists trust atoms? Because they make up everything.",
        "I tried to write a joke about UDP, but you might not get it.",
        "There are 10 kinds of people in the world: those who understand binary, and those who don't.",
        "Why did the developer go broke? Because he used up all his cache.",
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "Parallel lines have so much in common — it's a shame they'll never meet.",
        "I would tell you a joke about latency, but you'd have to wait for it.",
        "Why was the JavaScript developer sad? Because he didn't know how to null his feelings.",
        "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
        "Why did the database administrator leave his wife? She had one-to-many relationships.",
        "I told my computer I needed a break. It said: no problem, it would go to sleep.",
        "Why don't programmers like nature? It has too many bugs and not enough Wi-Fi.",
        "What's a programmer's favourite hangout spot? The Foo Bar.",
        "Why did the function return early? Because it had a stack overflow at home.",
        "I named my dog Five Miles. Now I can say I walk Five Miles every day.",
    ]

    QUOTE_POOL: list[str] = [
        "The best way to predict the future is to invent it. — Alan Kay",
        "Premature optimisation is the root of all evil. — Donald Knuth",
        "Simplicity is the ultimate sophistication. — Leonardo da Vinci",
        "Talk is cheap. Show me the code. — Linus Torvalds",
        "Programs must be written for people to read, and only incidentally for machines to execute. — Hal Abelson",
        "Make it work, make it right, make it fast. — Kent Beck",
        "Any sufficiently advanced technology is indistinguishable from magic. — Arthur C. Clarke",
        "First, solve the problem. Then, write the code. — John Johnson",
        "Code is like humour. When you have to explain it, it's bad. — Cory House",
        "The function of good software is to make the complex appear to be simple. — Grady Booch",
        "Walking on water and developing software from a specification are easy if both are frozen. — Edward V. Berard",
        "There are two ways of constructing a software design: one way is to make it so simple that there are obviously no deficiencies. — C.A.R. Hoare",
    ]

    RECIPE_POOL: list[str] = [
        "Pasta aglio e olio: boil salted water, cook spaghetti to al dente, gently warm sliced garlic in olive oil with chilli flakes, toss the drained pasta with the oil and a splash of pasta water, finish with parsley and parmesan.",
        "Quick scrambled eggs: whisk three eggs with a pinch of salt, melt butter on low heat, add eggs and stir continuously, remove from heat while still slightly wet, finish with chives.",
        "Tomato soup: sweat one chopped onion in olive oil, add a tin of tomatoes and a cup of stock, simmer fifteen minutes, blend smooth, season with salt and a swirl of cream.",
        "Banana pancakes: mash two ripe bananas with two eggs and a pinch of cinnamon, ladle into a hot non-stick pan, flip when bubbles form on the surface, serve with maple syrup.",
        "Stir-fried noodles: cook noodles, heat oil in a wok, sear sliced vegetables on high heat, add noodles plus soy sauce, sesame oil, and a splash of stock, toss for two minutes and serve.",
        "Chickpea curry: fry onion, garlic, and ginger, add curry powder, stir in a tin of chickpeas and a tin of coconut milk, simmer ten minutes, serve over rice with coriander.",
        "Greek salad: chunk tomatoes, cucumber, and red onion, add olives and cubes of feta, dress with olive oil, lemon juice, oregano, salt, and pepper.",
        "French omelette: whisk three eggs with salt, melt butter in a hot pan, pour in eggs and stir with a fork while shaking the pan, fold and tip onto a plate while still creamy inside.",
        "Roast potatoes: parboil halved potatoes for ten minutes, drain and shake to roughen the edges, toss with hot oil and salt, roast at 200 C for forty-five minutes, turning once.",
        "Simple bread: mix 500 g flour, 10 g salt, 7 g yeast, 350 ml water, knead ten minutes, prove until doubled, shape, prove again, bake at 220 C for thirty minutes.",
    ]

    POEM_POOL: list[str] = [
        "The fog comes / on little cat feet. / It sits looking / over harbour and city / on silent haunches / and then moves on.",
        "Two roads diverged in a yellow wood, / and sorry I could not travel both / and be one traveller, long I stood / and looked down one as far as I could.",
        "Hope is the thing with feathers / that perches in the soul / and sings the tune without the words / and never stops at all.",
        "The world is too much with us; late and soon, / getting and spending, we lay waste our powers; / little we see in nature that is ours; / we have given our hearts away, a sordid boon.",
        "I wandered lonely as a cloud / that floats on high o'er vales and hills, / when all at once I saw a crowd, / a host, of golden daffodils.",
        "Do not go gentle into that good night, / old age should burn and rave at close of day; / rage, rage against the dying of the light.",
        "Tell all the truth but tell it slant — / success in circuit lies, / too bright for our infirm delight / the truth's superb surprise.",
        "Stopping by woods on a snowy evening: / the woods are lovely, dark and deep, / but I have promises to keep, / and miles to go before I sleep.",
        "Tyger Tyger, burning bright, / in the forests of the night; / what immortal hand or eye / could frame thy fearful symmetry?",
        "Shall I compare thee to a summer's day? / Thou art more lovely and more temperate. / Rough winds do shake the darling buds of May, / and summer's lease hath all too short a date.",
    ]

    STORY_POOL: list[str] = [
        "A clockmaker built a single perfect clock and gave it away. Years later he returned to find it still ticking, surrounded by the children of the family who had received it. They had never wound it. He never explained why.",
        "The lighthouse keeper had not seen another person in eleven years. One night a stranger climbed the rocks, sat with him, and watched the sea. They did not speak. In the morning he was gone, and the keeper found the lamp had burned a little brighter that night.",
        "She bought the old piano because no one else would. The first note she played was wrong; the second one apologised; the third made the whole room remember a song nobody had taught it.",
        "The boy traded a stone for a bird. He traded the bird for a feather. He traded the feather for a story, and the story he kept all his life.",
        "An old man planted an oak tree he would never sit under. When asked why, he said: someone planted the trees I sit under now.",
        "The shopkeeper sold thunder by the bottle. Rain came in jars, and snow folded into envelopes. One day a child asked for a bottle of silence. He had nothing to put in it.",
        "Two travellers met at a crossroads. Each was certain the other had taken the wrong path. They argued until the sun set, then walked home together, in different directions.",
        "She wrote the letter every year and never sent it. On the tenth year she finally sent it. The reply came the next morning: I have been waiting for ten years.",
    ]

    FACT_POOL: list[str] = [
        "Octopuses have three hearts and blue blood — copper-based haemocyanin carries oxygen instead of haemoglobin.",
        "Honey never spoils — archaeologists have found pots of edible honey in ancient Egyptian tombs more than three thousand years old.",
        "Bananas are berries, but strawberries technically aren't — botanically a berry has seeds inside its flesh.",
        "There are more possible games of chess than atoms in the observable universe — about 10^120 versus 10^80.",
        "Sharks predate trees by roughly fifty million years — sharks are about 400 million years old, trees about 350.",
        "A day on Venus is longer than its year — Venus rotates extremely slowly, and a Venusian day is 243 Earth days while a Venusian year is 225.",
        "Wombat droppings are cube-shaped — the shape comes from the elasticity of their intestines and helps the cubes stay where they're placed as territorial markers.",
        "The Eiffel Tower can be 15 cm taller in summer because the iron expands as it heats up.",
        "Antarctica is the world's largest desert — a desert is defined by precipitation, not temperature, and Antarctica gets less than 200 mm a year.",
        "There are about 86 billion neurons in the average human brain, roughly the same number as stars in the Milky Way galaxy.",
        "The longest place name in the world is a hill in New Zealand: Taumatawhakatangihangakoauauotamateaturipukakapikimaungahoronukupokaiwhenuakitanatahu.",
        "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid of Giza.",
        "Bees can recognise individual human faces — they assemble facial features into a configuration the way primates do.",
    ]

    BOOK_REC_POOL: list[str] = [
        "Try Dune by Frank Herbert — a vast political-ecological science-fiction novel that rewards close attention.",
        "Read The Remains of the Day by Kazuo Ishiguro — a quiet, devastating first-person novel about an English butler reflecting on his life.",
        "Pick up Sapiens by Yuval Noah Harari for a sweeping, opinionated history of humanity from forager to algorithm.",
        "Try The Three-Body Problem by Liu Cixin — hard science fiction with first-contact stakes and a Chinese cultural revolution backdrop.",
        "Read Project Hail Mary by Andy Weir for first-principles problem-solving in space told with great warmth.",
        "Try Beloved by Toni Morrison — a haunting novel about memory, motherhood, and the legacy of slavery in America.",
        "Pick up Designing Data-Intensive Applications by Martin Kleppmann if you build software — it's the modern systems-engineering bible.",
        "Read The Pragmatic Programmer by Andrew Hunt and David Thomas for timeless software-engineering principles.",
        "Try Educated by Tara Westover, a memoir about growing up off-grid in Idaho and ending up at Cambridge.",
        "Pick up A Gentleman in Moscow by Amor Towles — a novel about a Russian aristocrat sentenced to house arrest in a grand hotel.",
        "Read Thinking, Fast and Slow by Daniel Kahneman for the foundational ideas of behavioural psychology and decision-making.",
        "Try Piranesi by Susanna Clarke — a short, strange, beautiful novel about a man who lives in an infinite house.",
    ]

    ADVICE_POOL: list[str] = [
        "When stuck, write down what you know and what you don't — most blockers are clarity problems wearing a hard-hat.",
        "Sleep on hard decisions when you can. The version of you that is rested makes consistently better calls than the one that is tired.",
        "Two minutes of action beats two hours of planning when you don't know where to start.",
        "Your first draft is allowed to be bad. Your job in draft one is to exist on the page; your job in draft two is to be good.",
        "Take notes by hand for things you want to remember and by keyboard for things you want to find later.",
        "When you don't understand something, try to explain it to a curious twelve-year-old. The friction shows you the gaps.",
        "Buy quality once for the things you use every day. Cheap shoes, cheap chairs, and cheap knives have hidden taxes.",
        "Eat a real breakfast, walk thirty minutes a day, and go to bed at the same time. The boring habits are the ones that compound.",
    ]

    TRANSLATE_POOL: list[str] = [
        "I'm not a full translator and don't have a multilingual model loaded — but for short phrases I can usually offer a reasonable English gloss. Tell me the source language and the phrase.",
        "Machine translation is outside my main scope. For high-quality work, dedicated tools like DeepL or Google Translate will give you a much better result.",
        "I can recognise a few common phrases in major European languages, but for anything that matters, please use a proper translator.",
    ]

    # --- B. Substantive explain-X answers (top 30 explain queries) ----

    EXPLAIN_ANSWERS: dict[str, str] = {
        "photosynthesis": "Photosynthesis is how plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The chlorophyll in leaves absorbs light energy, the light reactions split water and produce ATP and NADPH, and the Calvin cycle uses those to fix carbon into sugar.",
        "how a transformer works": "A transformer is a neural network built around self-attention. Each token attends to every other token via query/key/value projections, so the model can learn long-range dependencies in parallel without recurrence. Stacked layers compose those interactions into rich contextual representations.",
        "quantum computing": "Quantum computing uses qubits, which can be in a superposition of 0 and 1, and entanglement to represent and process exponentially more states than classical bits. Algorithms like Shor's and Grover's exploit this to outperform classical methods on specific problems like factoring and unstructured search.",
        "machine learning": "Machine learning is the branch of AI where systems learn patterns from data instead of being hand-coded. You pick a model family, define a loss function, and use optimisation to fit the model parameters so predictions match the data. Common paradigms are supervised, unsupervised, and reinforcement learning.",
        "a neural network": "A neural network is a stack of layers of artificial neurons that each compute a weighted sum of their inputs, pass it through a non-linearity, and forward the result. Training adjusts the weights via gradient descent on a loss function so the network learns useful representations of its input.",
        "how batteries work": "A battery stores chemical energy and releases it as electricity through redox reactions at two electrodes. Inside, ions flow through an electrolyte from anode to cathode while electrons flow through the external circuit, doing useful work along the way.",
        "encryption": "Encryption converts readable data into ciphertext using a key so that only someone with the matching key can recover the original. Symmetric encryption uses the same key for both directions; public-key encryption uses a key pair so anyone can encrypt and only the holder of the private key can decrypt.",
        "how the internet works": "The internet is a global network of networks linked by routers that forward IP packets toward their destinations. DNS turns names into addresses, TCP provides reliable byte streams on top of best-effort IP, and TLS encrypts those streams. Higher-level protocols like HTTP run on top.",
        "artificial intelligence": "Artificial intelligence is the field of building systems that perform tasks we'd usually call intelligent — perception, reasoning, decision-making, language. Modern AI is dominated by machine learning, where models learn statistical patterns from large datasets rather than being hand-coded with rules.",
        "edge computing": "Edge computing pushes compute and data storage close to where data is produced — phones, sensors, on-prem boxes — instead of relying on a distant cloud. The benefits are lower latency, reduced bandwidth use, better privacy, and resilience to network outages.",
        "blockchain": "A blockchain is a tamper-evident, append-only ledger replicated across many nodes. Blocks of transactions are linked by cryptographic hashes, and a consensus protocol like proof-of-work or proof-of-stake decides which chain is canonical, making rewriting history prohibitively expensive.",
        "the cloud": "The cloud is on-demand computing infrastructure operated by providers like AWS, Azure, and Google Cloud. Instead of owning hardware, you rent virtual machines, storage, databases, and managed services over the internet, paying only for what you use.",
        "an api": "An API — application programming interface — is a defined contract for talking to a piece of software. It specifies the operations available, the inputs and outputs, and any rules of use, letting programs communicate without knowing each other's internals.",
        "an algorithm": "An algorithm is a finite, well-defined sequence of steps for solving a problem. Good algorithms are correct, efficient in time and space, and ideally easy to reason about. Sorting, searching, hashing, and graph traversal are classic examples.",
        "recursion": "Recursion is when a function solves a problem by calling itself on a smaller piece of the same problem and combining the results. Every recursion needs a base case to stop, and the recursive case must make measurable progress toward it.",
        "a database": "A database is a structured store designed for safe, concurrent read-and-write access. Relational databases organise data into tables with strict schemas and SQL queries; document, key-value, and graph databases relax those constraints to fit different shapes of data.",
        "compilation": "Compilation is the process of translating source code into a lower-level form like machine code or bytecode that a machine can run directly. The compiler parses the source, builds an intermediate representation, optimises it, and emits the target output.",
        "what version control is": "Version control is a system that records changes to files over time so you can review history, recover earlier states, and collaborate without overwriting one another's work. Git is the dominant tool today; it stores snapshots in a content-addressed object database.",
        "an operating system": "An operating system is the layer of software that manages a computer's hardware and provides services to applications: process scheduling, memory management, file systems, networking, and a security model. Linux, Windows, macOS, iOS, and Android are all operating systems.",
        "the cpu": "A CPU — central processing unit — is the chip that executes the instructions of a program. It fetches an instruction, decodes it, executes it, and writes back results. Modern CPUs use pipelining, out-of-order execution, and many cores to run billions of instructions per second.",
        "a gpu": "A GPU — graphics processing unit — is a chip with thousands of small cores designed for parallel work. It started as a graphics accelerator but is now central to scientific computing and machine learning because matrix arithmetic parallelises beautifully across its cores.",
        "ram": "RAM — random-access memory — is the fast, volatile working memory of a computer. The CPU reads instructions and data from RAM, and anything not saved to disk is lost when power is removed.",
        "an ssd": "An SSD — solid-state drive — stores data in flash memory chips with no moving parts. Compared to a spinning hard drive, an SSD has much faster random access, no seek time, and better resilience to shock, at higher cost per gigabyte.",
        "tcp": "TCP — Transmission Control Protocol — is a transport-layer protocol that provides reliable, ordered byte streams between two endpoints over IP. It handles handshakes, retransmissions, and flow control, hiding packet loss and reordering from the application.",
        "http": "HTTP — Hypertext Transfer Protocol — is the request/response protocol the web is built on. A client sends a request with a method like GET or POST, a server returns a response with a status code and a body. HTTP/2 and HTTP/3 add multiplexing and reduced latency.",
        "tls": "TLS — Transport Layer Security — encrypts and authenticates traffic between two endpoints. It handshakes a session key using public-key cryptography, verifies a server's identity via X.509 certificates, and then uses symmetric encryption for the actual data.",
        "git": "Git is a distributed version-control system where every working copy is a full repository with the complete history. Commits are content-addressed snapshots; branches are cheap pointers; merging and rebasing combine work from different lines of development.",
        "linux": "Linux is the family of open-source operating systems built on the Linux kernel, originally released by Linus Torvalds in 1991. It powers most internet servers, all Android phones, supercomputers, and a significant share of embedded devices.",
        "what python is": "Python is a high-level, dynamically-typed programming language with a focus on readability. It's a swiss-army knife for scripting, automation, web back-ends, data science, and machine learning, and has one of the largest open-source ecosystems on earth.",
        "javascript": "JavaScript is the dynamically-typed language that runs in every web browser, plus on servers via Node.js and Deno. It started as a tiny scripting language for forms and is now the most widely used programming language in the world.",
        "css": "CSS — Cascading Style Sheets — is the language that styles web content. Selectors target HTML elements; declarations set properties like color, font, and layout. Modern CSS supports flexbox, grid, custom properties, and animations.",
        "html": "HTML — HyperText Markup Language — is the markup language that structures web content. Tags describe headings, paragraphs, lists, links, images, and forms; the browser turns that document into the rendered page you see.",
        "regex": "A regular expression is a tiny pattern-matching language for strings. Literal characters match themselves; metacharacters like dot, star, plus, brackets, and parentheses describe repetition, alternation, character classes, and groups.",
        "what an llm is": "A large language model is a transformer trained on a vast corpus of text to predict the next token given the preceding tokens. After enough scale and instruction tuning, the model can carry on conversations, summarise, translate, write code, and more — all from that one objective.",
        "an slm": "A small language model — SLM — is a much smaller version of an LLM, typically tens of millions to a few billion parameters. It trades peak quality for the ability to run on-device, with low latency and no cloud dependency.",
        "rlhf": "RLHF — reinforcement learning from human feedback — is a fine-tuning method where humans rank model outputs, a reward model is trained on those rankings, and the language model is then tuned to maximise that reward. It's how chat-assistants are taught to be helpful and harmless.",
        "fine-tuning": "Fine-tuning is taking a pretrained model and continuing training on a narrower dataset to specialise it. It often only updates a subset of the weights — for example, low-rank adapters — so the original capability is largely preserved.",
        "prompt engineering": "Prompt engineering is the practice of writing inputs to a language model that reliably produce the output you want. It includes giving examples, breaking tasks into steps, specifying format, and providing relevant context.",
        "rag": "RAG — retrieval-augmented generation — combines a search index with a generative language model. The system retrieves relevant documents at query time and conditions the model on them, so the answer can ground in fresh, specific knowledge the model wasn't trained on.",
    }

    EXPLAIN_PROMPT_TEMPLATES: list[str] = [
        "explain {t}",
        "what is {t}",
        "what's {t}",
        "tell me about {t}",
        "can you explain {t}",
        "how does {t} work",
        "give me an overview of {t}",
        "describe {t}",
    ]
    explain_pairs: list[tuple[list[str], list[str]]] = []
    for topic, ans in EXPLAIN_ANSWERS.items():
        prompts = [tpl.format(t=topic) for tpl in EXPLAIN_PROMPT_TEMPLATES]
        explain_pairs.append((prompts, [ans]))

    # --- C. Self-meta coverage gaps the audit found -------------------

    META_PAIRS: list[tuple[list[str], list[str]]] = [
        (
            ["what languages do you speak", "what languages can you speak", "do you speak other languages", "what language do you understand"],
            ["I'm built around English text training data — I'll do best in English, with limited capability in major European languages."],
        ),
        (
            ["can you remember things", "do you remember our conversation", "do you have memory", "can you remember", "will you remember this"],
            ["I keep the last few turns of our conversation in context, but nothing persists once the session ends. That's by design — privacy first."],
        ),
        (
            ["what's your name", "whats your name", "what is your name", "tell me your name", "do you have a name"],
            ["I'm I3 — short for Implicit Interaction Intelligence. A small custom assistant you're talking to in this demo."],
        ),
        (
            ["are you alive", "are you sentient", "are you conscious", "do you feel things"],
            ["I'm a language model — patterns learned from text. I don't experience anything, but I try to respond in ways that are useful and considerate."],
        ),
        (
            ["who made you", "who built you", "who created you", "who developed you"],
            ["I'm a portfolio project — a small custom transformer plus an adaptive front-end built from scratch in PyTorch by a single developer for an HMI lab."],
        ),
        (
            ["where do you live", "where are you running", "where do you run", "are you on my device"],
            ["I run locally on this machine, no cloud round-trip per message. The whole demo is designed to work without network access."],
        ),
        (
            ["how old are you", "when were you trained", "what's your training date"],
            ["I was trained for a portfolio demo in 2026. My knowledge cut-off is roughly that point — anything after that is outside my training."],
        ),
        (
            ["are you safe", "are you secure", "is this safe to use"],
            ["The cloud route is off in this demo and the local pipeline runs entirely on-device. Embeddings that get persisted are encrypted; raw chat text is never written to disk."],
        ),
        (
            ["do you learn from me", "are you learning from me", "do you train on my chats"],
            ["Not in this session. The model weights are frozen. The adaptation layer reacts to your typing in real time, but nothing about you is written back to the model."],
        ),
        (
            ["are you better than chatgpt", "are you smarter than gpt", "how do you compare to chatgpt"],
            ["No — I'm orders of magnitude smaller than the cloud LLMs and I'll lose on raw factual coverage. The point of this demo is what a tiny on-device model can still do well: privacy, latency, and adaptation to your style."],
        ),
        (
            ["why are you so small", "why are you tiny", "why aren't you bigger"],
            ["Because the goal is on-device inference. A small model fits on a phone or laptop, runs offline, and answers in tens of milliseconds. Size is the constraint that makes the rest interesting."],
        ),
        (
            ["how should i talk to you", "how do i talk to you", "what's the best way to ask you something"],
            ["Plain English, the way you'd ask a colleague. I'll adapt to short, casual messages and longer, more technical ones — that's what the adaptation layer is for."],
        ),
        (
            ["what should i ask you", "what can i ask you", "what do you do best"],
            ["Casual conversation, definitions, short factual questions, and follow-ups about a topic we've already established. I'm not a search engine and I'll be honest when something's outside my range."],
        ),
    ]

    # --- D. Top entity follow-ups ------------------------------------

    ENTITY_FOLLOWUPS: list[tuple[list[str], list[str]]] = [
        # Apple — bare name + paraphrases all map to the same paragraph
        (
            ["apple", "tell me more about apple", "tell me about apple", "what's apple", "what is apple", "who is apple"],
            ["Apple is a US consumer-electronics and software company headquartered in Cupertino, California. It designs the iPhone, Mac, iPad, Apple Watch, and runs services like iCloud, the App Store, and Apple Music."],
        ),
        (
            ["is apple american", "is apple a us company", "where is apple from"],
            ["Yes, Apple is headquartered in Cupertino, California, and is one of the largest US technology companies."],
        ),
        (
            ["what does apple sell", "what does apple make", "what products does apple make"],
            ["Apple sells iPhones, Macs, iPads, the Apple Watch, AirPods, and a growing suite of services including iCloud, Apple Music, the App Store, and Apple TV+."],
        ),
        (
            ["most famous product", "what is apple's most famous product", "apple's most famous product", "their most famous product"],
            ["Apple's most famous product is the iPhone, first released in 2007. It defined the modern smartphone category and remains the company's largest source of revenue."],
        ),
        (
            ["who runs apple", "ceo of apple", "who is the ceo of apple"],
            ["Apple's CEO is Tim Cook, who took over from Steve Jobs in 2011."],
        ),
        # Huawei
        (
            ["huawei", "tell me more about huawei", "tell me about huawei", "what's huawei", "what is huawei"],
            ["Huawei is a global technology company headquartered in Shenzhen, China. It works across consumer devices like smartphones, telecom networking equipment, cloud services, and HarmonyOS."],
        ),
        (
            ["is huawei chinese", "is huawei from china"],
            ["Yes, Huawei is headquartered in Shenzhen, China, and is one of the country's largest technology companies."],
        ),
        (
            ["what does huawei sell", "what does huawei make"],
            ["Huawei makes smartphones and tablets, telecom networking equipment, cloud services, and HarmonyOS, plus enterprise hardware. It also runs a large global R&D operation."],
        ),
        (
            ["does huawei have an r and d centre", "does huawei have research labs", "does huawei do research"],
            ["Yes — Huawei runs major research and development centres around the world, including a research and development presence in the United Kingdom."],
        ),
        # Microsoft
        (
            ["microsoft", "tell me more about microsoft", "tell me about microsoft", "what's microsoft", "what is microsoft"],
            ["Microsoft is a US software and cloud-services company headquartered in Redmond, Washington. It makes Windows, Office, the Azure cloud platform, Xbox, and developer tools like Visual Studio and GitHub."],
        ),
        (
            ["is microsoft american", "is microsoft a us company"],
            ["Yes, Microsoft is headquartered in Redmond, Washington, and is one of the largest US software companies."],
        ),
        (
            ["what does microsoft sell", "what does microsoft make"],
            ["Microsoft sells Windows, the Office suite, Azure cloud services, Xbox consoles and games, Surface devices, and developer tools like Visual Studio and GitHub."],
        ),
        # Google
        (
            ["google", "tell me more about google", "tell me about google", "what's google", "what is google"],
            ["Google is a US search and advertising company headquartered in Mountain View, California, and a subsidiary of Alphabet. It runs Search, Android, Chrome, YouTube, Workspace, and Google Cloud."],
        ),
        (
            ["what does google sell", "what does google make"],
            ["Google's main business is online advertising tied to its Search engine. It also runs Android, Chrome, Workspace, YouTube, Pixel devices, and the Google Cloud platform."],
        ),
        # OpenAI
        (
            ["openai", "tell me more about openai", "tell me about openai", "what's openai", "what is openai"],
            ["OpenAI is a US AI research and deployment company headquartered in San Francisco. It builds the GPT family of large language models and runs ChatGPT and the OpenAI API."],
        ),
        # Anthropic
        (
            ["anthropic", "tell me more about anthropic", "tell me about anthropic", "what's anthropic", "what is anthropic"],
            ["Anthropic is an AI safety company that builds the Claude family of large language models. It's headquartered in San Francisco and was founded in 2021."],
        ),
        # Meta
        (
            ["meta", "tell me more about meta", "tell me about meta", "what's meta", "what is meta"],
            ["Meta is the US social-media and VR company formerly known as Facebook. It's headquartered in Menlo Park, California, and runs Facebook, Instagram, WhatsApp, and the Quest VR headsets."],
        ),
        # Amazon
        (
            ["amazon", "tell me more about amazon", "tell me about amazon", "what's amazon", "what is amazon"],
            ["Amazon is a US e-commerce and cloud-services company headquartered in Seattle. It runs the Amazon marketplace, AWS, Kindle, and the Alexa voice assistant."],
        ),
        # Samsung
        (
            ["samsung", "tell me more about samsung", "tell me about samsung", "what's samsung", "what is samsung"],
            ["Samsung is a South Korean conglomerate headquartered in Suwon. Samsung Electronics is its best-known unit, making smartphones, displays, memory chips, and home appliances."],
        ),
        # Tesla
        (
            ["tesla", "tell me more about tesla", "tell me about tesla as a company", "what's tesla the company"],
            ["Tesla is a US electric-vehicle and energy company headquartered in Austin, Texas. It makes electric cars including the Model S, 3, X, Y, and Cybertruck, plus solar panels and home batteries."],
        ),
        # Nvidia
        (
            ["nvidia", "tell me more about nvidia", "tell me about nvidia", "what's nvidia", "what is nvidia"],
            ["Nvidia is a US semiconductor company headquartered in Santa Clara, California. It designs GPUs for graphics, gaming, and AI training, plus the CUDA software platform that's the de-facto standard for ML compute."],
        ),
        # Intel
        (
            ["intel", "tell me more about intel", "tell me about intel", "what's intel", "what is intel"],
            ["Intel is a US semiconductor company headquartered in Santa Clara, California. It designs and manufactures x86 CPUs for PCs and servers, plus networking chips and foundry services."],
        ),
        # IBM
        (
            ["ibm", "tell me more about ibm", "tell me about ibm", "what's ibm", "what is ibm"],
            ["IBM is a US technology and consulting company headquartered in Armonk, New York. Its modern business is enterprise hardware, software, hybrid cloud, and consulting, with a long pedigree in computing research."],
        ),
        # Sony
        (
            ["sony", "tell me more about sony", "tell me about sony", "what's sony", "what is sony"],
            ["Sony is a Japanese electronics, gaming, and entertainment conglomerate headquartered in Tokyo. It makes the PlayStation consoles, image sensors, cameras, and operates major music and film labels."],
        ),
        # Languages
        (
            ["is python easy to learn", "is python easy", "is python a good first language"],
            ["Python is widely considered one of the easier programming languages to learn — readable syntax, forgiving rules, and a huge ecosystem of beginner-friendly libraries make it the standard first language for many people."],
        ),
        (
            ["is rust hard to learn", "is rust difficult", "is rust hard"],
            ["Rust is generally considered harder than most languages because the borrow-checker forces you to think about ownership and lifetimes. The payoff is memory safety and performance close to C without a garbage collector."],
        ),
        (
            ["is javascript a good language", "is js worth learning", "should i learn javascript"],
            ["Yes — JavaScript runs in every browser and on the back-end via Node.js and Deno, so it's the broadest-coverage language for building things people actually use. Modern JavaScript with TypeScript is genuinely pleasant."],
        ),
        # Linux
        (
            ["is linux free", "do i pay for linux"],
            ["Most Linux distributions are free as in beer and free as in speech — you can download, use, and modify them at no cost. Commercial vendors like Red Hat charge for support and certification, not the software."],
        ),

        # Python — who created / what's used for / why popular
        (
            ["who created python", "who invented python", "who made python",
             "who created it python", "who started python"],
            ["Python was created by Guido van Rossum, who released the first version in 1991 while working at CWI in Amsterdam. He was the project's 'Benevolent Dictator For Life' until stepping back from that role in 2018."],
        ),
        (
            ["what is python used for", "what's python used for",
             "what do people use python for", "where is python used"],
            ["Python is used everywhere — data science and machine learning, web back-ends (Django, Flask, FastAPI), automation and scripting, scientific computing, education, and DevOps tooling. Its huge library ecosystem is the main reason it spreads so widely."],
        ),
        (
            ["why is python popular", "why is python so popular",
             "why do people like python"],
            ["Python's popularity comes from a few combined factors: readable syntax that's easy to teach, a huge standard library and PyPI ecosystem, dominance in data science and ML through NumPy / pandas / PyTorch / TensorFlow, and a forgiving learning curve that makes prototyping fast."],
        ),

        # Rust — who created / what's used for / why popular / harder to learn
        (
            ["who created rust", "who invented rust", "who made rust"],
            ["Rust was started by Graydon Hoare at Mozilla Research in 2006. The first stable release shipped in 2015; today Rust is stewarded by the independent Rust Foundation."],
        ),
        (
            ["what is rust used for", "what's rust used for",
             "where is rust used"],
            ["Rust is used for systems programming where performance and memory safety both matter — operating-system kernels, browser engines (Servo, parts of Firefox), embedded firmware, blockchain runtimes, and increasingly back-end services and CLIs that used to be written in C++ or Go."],
        ),
        (
            ["is rust harder to learn", "is rust harder", "is it harder to learn"],
            ["Rust is generally considered harder to learn than most popular languages because the borrow-checker forces you to think about ownership and lifetimes up front. The payoff is that the compiler catches whole classes of memory and concurrency bugs at compile time without a garbage collector."],
        ),

        # JavaScript follow-ups
        (
            ["who created javascript", "who invented javascript", "who made javascript"],
            ["JavaScript was created by Brendan Eich at Netscape in 1995, reportedly designed in just ten days. It was originally meant as a simple scripting language for web pages and grew into the dominant language of the web."],
        ),
        (
            ["what is javascript used for", "what's javascript used for",
             "where is javascript used"],
            ["JavaScript runs in every web browser as the language of front-end interactivity. Via Node.js and Deno it also runs on the back-end, plus in mobile apps (React Native), desktop apps (Electron), and serverless functions."],
        ),

        # Linux follow-ups
        (
            ["who created linux", "who invented linux", "who made linux",
             "who started linux"],
            ["Linux was created by Linus Torvalds in 1991 while he was a student at the University of Helsinki. He released the first version of the kernel publicly that year, and he still oversees the kernel's development today."],
        ),

        # ChatGPT / Claude / GPT follow-ups
        (
            ["who made chatgpt", "who created chatgpt", "who built chatgpt"],
            ["ChatGPT was built by OpenAI, a US AI research and deployment company. It was released to the public in November 2022."],
        ),
        (
            ["who made claude", "who created claude", "who built claude"],
            ["Claude is built by Anthropic, an AI safety company headquartered in San Francisco and founded in 2021."],
        ),
        (
            ["who made gpt", "who created gpt", "who built gpt"],
            ["GPT — Generative Pre-trained Transformer — is the family of large language models built by OpenAI. The first version was released in 2018, with GPT-2, GPT-3, GPT-4, and successors following."],
        ),

        # Tokyo / Japan follow-ups (city → country / language)
        (
            ["what country", "what country?", "what country is it in",
             "which country", "which country is it in", "where is it"],
            ["Tell me which place — different cities and landmarks are in different countries. If you can name the city or feature I'll give you the country and a quick orientation."],
        ),
        (
            ["language there", "what language do they speak there",
             "what language", "which language", "what language do they speak"],
            ["It depends on the country — tell me which place you mean. For example: Japanese is spoken in Japan, French in France, German in Germany, Mandarin in mainland China."],
        ),
        (
            ["tokyo", "tell me about tokyo", "what is tokyo"],
            ["Tokyo is the capital of Japan and one of the largest cities in the world. It sits on the eastern coast of the main island of Honshu and is the political, economic, and cultural centre of the country."],
        ),
        (
            ["what country is tokyo in", "is tokyo in japan",
             "where is tokyo located"],
            ["Tokyo is the capital of Japan, on the eastern coast of the main island of Honshu."],
        ),
        (
            ["language in tokyo", "language in japan",
             "what language is spoken in japan"],
            ["The official language of Japan is Japanese, written in a mix of three scripts — kanji (Chinese-derived characters), hiragana, and katakana."],
        ),

        # Internet follow-ups
        (
            ["what is the internet", "what's the internet", "tell me about the internet"],
            ["The internet is a global network of interconnected computer networks. It carries data using a stack of protocols — IP for addressing, TCP for reliable transport, plus HTTP, DNS, TLS, and many more — letting billions of devices share information regardless of hardware or operating system."],
        ),

        # Neural network follow-ups
        (
            ["what is a neural network", "what's a neural network",
             "tell me about neural networks"],
            ["A neural network is a computational model loosely inspired by biological neurons. Layers of connected units take a numeric input, multiply it by learned weights, apply a non-linear activation, and pass the result on. By stacking layers and training on lots of examples, the network learns patterns that map inputs to outputs."],
        ),
        (
            ["how does a neural network learn", "how does it learn"],
            ["A neural network learns by seeing many input/output examples and adjusting its connection weights so its predictions get closer to the expected outputs. The adjustment is computed by gradient descent — the loss is the gap between prediction and target, and gradients tell each weight which direction to nudge."],
        ),

        # AI follow-ups
        (
            ["examples of ai", "ai examples", "what are some examples of ai",
             "give me examples of ai"],
            ["Examples of AI in everyday use include voice assistants (Siri, Alexa, Google Assistant), image recognition in your phone's camera, recommendation systems on YouTube and Netflix, language models like ChatGPT and Claude, and self-driving features in cars."],
        ),

        # 'and X' acknowledgements where X is a topic
        (
            ["actually rust", "and rust", "what about rust"],
            ["Rust is a systems programming language focused on safety, concurrency, and performance. It prevents many memory-safety bugs at compile time without using a garbage collector."],
        ),
        (
            ["which should i learn", "which should i learn first",
             "should i learn python or rust", "python or rust",
             "should i learn rust or python"],
            ["Depends on your goal. Pick Python first if you want quick payoff for data science, ML, scripting, or general-purpose work — it's faster to learn and the ecosystem is enormous. Pick Rust if you want systems-level performance and memory safety, and you're prepared for a steeper learning curve."],
        ),
        (
            ["which is faster python or rust", "is python faster than rust",
             "is rust faster than python"],
            ["Rust is dramatically faster than Python for CPU-bound work — often 10-100x — because it compiles to native code without a global interpreter lock or runtime garbage collection. Python wins on developer time per feature, not on runtime speed."],
        ),
        (
            ["actually python", "and python", "what about python"],
            ["Python is a high-level, dynamically-typed programming language with a focus on readability. It's the standard for data science, ML, scripting, and back-end work."],
        ),
        (
            ["and apple", "and apple too", "what about apple",
             "actually apple"],
            ["Apple is a US consumer-electronics and software company headquartered in Cupertino, California. It designs the iPhone, Mac, iPad, Apple Watch, and runs services like iCloud, the App Store, and Apple Music."],
        ),
        (
            ["and huawei", "and huawei too", "what about huawei",
             "actually huawei", "i was just thinking about huawei",
             "wat is huwaei", "what is huwaei", "wat is huawei",
             "tell me about huwaei", "huwaei", "who is huwaei"],
            ["Huawei is a global technology company headquartered in Shenzhen, China. It works across consumer devices like smartphones, telecom networking equipment, cloud services, and HarmonyOS."],
        ),
        (
            ["and newton", "what about newton", "actually newton"],
            ["Isaac Newton was an English mathematician and physicist who described the laws of motion and universal gravitation in the 17th century, and laid the foundations of calculus."],
        ),

        # Generic 'tell me about X' / 'what about X'
    ]

    # --- D2. Country follow-ups (capital, currency, language) -------

    COUNTRY_FOLLOWUPS: list[tuple[list[str], list[str]]] = [
        # Germany
        (["germany", "tell me about germany", "what is germany"],
         ["Germany is a country in central Europe with the largest economy in the European Union. Its capital is Berlin and its currency is the euro."]),
        (["germany capital", "capital of germany", "germany's capital"],
         ["The capital of Germany is Berlin."]),
        (["germany currency", "currency of germany", "germany money"],
         ["Germany's currency is the euro, which it has used since 1999 (notes and coins from 2002)."]),
        (["germany language", "language of germany", "what language do they speak in germany"],
         ["The official language of Germany is German."]),
        # France
        (["france", "tell me about france", "what is france"],
         ["France is a country in western Europe known for its cuisine, art, and history. Its capital is Paris and its currency is the euro."]),
        (["france capital", "capital of france"],
         ["The capital of France is Paris."]),
        (["france currency", "currency of france"],
         ["France's currency is the euro."]),
        (["france language", "language of france"],
         ["The official language of France is French."]),
        # Japan
        (["japan", "tell me about japan", "what is japan"],
         ["Japan is an island nation in East Asia composed of four major islands. Its capital is Tokyo and its currency is the yen."]),
        (["japan capital", "capital of japan"],
         ["The capital of Japan is Tokyo."]),
        (["japan currency", "currency of japan"],
         ["Japan's currency is the yen."]),
        (["japan language", "language of japan"],
         ["The official language of Japan is Japanese."]),
        # China
        (["china", "tell me about china", "what is china"],
         ["China is the most populous country in East Asia and one of the world's largest economies. Its capital is Beijing and its currency is the yuan (renminbi)."]),
        (["china capital", "capital of china"],
         ["The capital of China is Beijing."]),
        (["china currency", "currency of china"],
         ["China's currency is the yuan, also known as the renminbi."]),
        (["china language", "language of china"],
         ["The official language of mainland China is Mandarin Chinese."]),
        # United Kingdom
        (["united kingdom", "uk", "tell me about the uk", "what is the uk"],
         ["The United Kingdom is a country in north-western Europe made up of England, Scotland, Wales, and Northern Ireland. Its capital is London and its currency is the pound sterling."]),
        (["uk capital", "capital of the uk", "capital of england"],
         ["The capital of the United Kingdom is London."]),
        (["uk currency", "currency of the uk"],
         ["The United Kingdom's currency is the pound sterling."]),
        # United States
        (["united states", "usa", "us", "america", "tell me about america"],
         ["The United States is a country in North America made up of fifty states. Its capital is Washington, D.C., and its currency is the US dollar."]),
        (["usa capital", "capital of america", "capital of the us"],
         ["The capital of the United States is Washington, D.C."]),
        # Russia
        (["russia", "tell me about russia"],
         ["Russia is the largest country on Earth by area, spanning eleven time zones across Europe and Asia. Its capital is Moscow and its currency is the Russian rouble."]),
        # Italy / Spain / Canada / Australia / Brazil / India
        (["italy", "tell me about italy"],
         ["Italy is a country in southern Europe shaped like a boot. Its capital is Rome, its currency is the euro, and it has a long history spanning the Roman Empire to the Renaissance."]),
        (["spain", "tell me about spain"],
         ["Spain is a country in south-western Europe with strong regional cultures. Its capital is Madrid and its currency is the euro."]),
        (["canada", "tell me about canada"],
         ["Canada is the second-largest country by area and lies north of the United States. Its capital is Ottawa, its currency is the Canadian dollar, and it is officially bilingual in English and French."]),
        (["australia", "tell me about australia"],
         ["Australia is a country and continent in the southern hemisphere. Its capital is Canberra, its currency is the Australian dollar, and its unique wildlife includes kangaroos and koalas."]),
        (["brazil", "tell me about brazil"],
         ["Brazil is the largest country in South America. Its capital is Brasília, its currency is the Brazilian real, and it spans a huge section of the Amazon rainforest."]),
        (["india", "tell me about india"],
         ["India is a country in South Asia with more than a billion inhabitants. Its capital is New Delhi, its currency is the Indian rupee, and it has thousands of years of cultural history."]),
    ]

    # --- D3. Bare-topic follow-ups for popular subjects -------------
    BARE_TOPIC_FOLLOWUPS: list[tuple[list[str], list[str]]] = [
        # python
        (["python", "tell me about python the language"],
         ["Python is a high-level, dynamically-typed programming language with a focus on readability. It's a swiss-army knife for scripting, automation, web back-ends, data science, and machine learning."]),
        # rust
        (["rust", "tell me about rust the language"],
         ["Rust is a systems programming language focused on safety, concurrency, and performance. It prevents many memory-safety bugs at compile time without a garbage collector."]),
        # javascript
        (["javascript", "tell me about javascript"],
         ["JavaScript is the dynamically-typed language that runs in every web browser, plus on servers via Node.js and Deno. It's the most widely used programming language in the world."]),
        # linux
        (["linux", "tell me about linux"],
         ["Linux is the family of open-source operating systems built on the Linux kernel, originally released by Linus Torvalds in 1991. It powers most internet servers, all Android phones, and a huge share of embedded devices."]),
        # chatgpt
        (["chatgpt", "tell me about chatgpt"],
         ["ChatGPT is a conversational assistant from OpenAI built on top of the GPT family of large language models. It launched in 2022 and is one of the most widely used AI products in the world."]),
        # claude
        (["claude", "tell me about claude"],
         ["Claude is the family of large language models built by Anthropic, an AI safety company. It's available through chat products and the Anthropic API."]),
        # gemini
        (["gemini", "tell me about gemini"],
         ["Gemini is the family of large multimodal models from Google DeepMind, available through the Gemini chat product and the Google Cloud Vertex AI platform."]),
        # photosynthesis follow-ups
        (["chlorophyll", "what is chlorophyll", "tell me about chlorophyll", "what's chlorophyll"],
         ["Chlorophyll is the green pigment in plant cells that absorbs light energy from the sun, kicking off the photosynthesis reaction. It captures sunlight and feeds the energy into the reactions that fix carbon dioxide into sugar."]),
        (["respiration", "what is respiration", "and respiration", "what about respiration", "what's respiration"],
         ["Cellular respiration is the process by which cells release energy from glucose by reacting it with oxygen, producing carbon dioxide and water plus ATP. It's the inverse of photosynthesis at the level of carbon flow."]),
        # photosynthesis -> follow-up "what about animals"
        (["what about animals", "do animals do photosynthesis", "and animals", "what about animals?",
          "do animals photosynthesize", "do animals photosynthesise"],
         ["Animals don't perform photosynthesis. They get their energy by eating plants or other animals. The animal-side equivalent is cellular respiration, which works in reverse — using oxygen and glucose to release energy."]),
        # photosynthesis -> "is it important" / "why does it matter"
        (["is it important", "why does it matter", "why is it important",
          "is photosynthesis important", "why does photosynthesis matter"],
         ["Photosynthesis produces nearly all the oxygen in our atmosphere and is the base of every food chain on Earth — without it, most life as we know it couldn't exist."]),
        # photosynthesis -> "are they opposites" with respiration
        (["are they opposites", "are they opposite", "is it the opposite of photosynthesis",
          "are photosynthesis and respiration opposites"],
         ["Yes — they're roughly opposites at the carbon-flow level. Photosynthesis takes in carbon dioxide and water and produces glucose and oxygen; cellular respiration consumes glucose and oxygen and produces carbon dioxide and water."]),

        # internet follow-ups
        (["who invented the internet", "who created the internet", "who invented internet",
          "who made the internet", "who is the inventor of the internet"],
         ["The internet grew out of ARPANET in the late 1960s. Vint Cerf and Bob Kahn designed the TCP/IP protocols that hold it together; Tim Berners-Lee invented the World Wide Web on top of it in 1989."]),
        (["who invented it", "who created it", "who invented this", "who made it"],
         ["Tell me which one — different inventions have different inventors. The internet, for instance, came out of ARPANET work in the late 1960s, with TCP/IP designed by Vint Cerf and Bob Kahn."]),
        (["how does the internet work", "how does internet work"],
         ["The internet is a network of networks. Computers send small chunks of data called packets that hop through routers using the IP protocol; TCP makes sure they arrive in order and intact, and DNS turns names like example.com into numeric addresses."]),

        # battery follow-ups
        (["types of batteries", "what types of batteries are there", "battery types",
          "kinds of batteries", "examples of batteries"],
         ["Common battery chemistries include alkaline (single-use), lithium-ion (most phones, laptops, EVs), nickel-metal hydride (older rechargeables), and lead-acid (cars). Lithium-ion dominates portable electronics because of its high energy density."]),
        (["how do batteries work", "how does a battery work"],
         ["A battery stores chemical energy and releases it as electricity. Inside the cell, a chemical reaction at one electrode releases electrons that flow through your circuit to the other electrode, while ions move through an electrolyte to balance the charge."]),

        # neural network / AI follow-ups
        (["examples of neural networks", "examples of ai", "ai examples",
          "what are some examples of ai", "examples"],
         ["Examples of modern neural networks and AI in everyday use include voice assistants like Siri and Alexa, image recognition in your phone's camera, language models like ChatGPT and Claude, recommendation systems on YouTube and Netflix, and self-driving features in cars."]),
        (["how does it learn", "how do neural networks learn",
          "how does a neural network learn"],
         ["A neural network learns by seeing many examples and adjusting its connection weights to reduce a loss function — a number that measures how wrong its predictions are. The adjustment is computed by gradient descent and propagated backwards through the layers."]),
        (["how does it work", "how does it work?"],
         ["Tell me which one — algorithms, devices, and physical systems all 'work' differently. If you can name the thing you're curious about (a transformer, a battery, photosynthesis, the internet) I'll walk through the mechanism."]),

        # AI follow-ups
        (["what is ai", "what's ai", "what is artificial intelligence", "what's artificial intelligence"],
         ["Artificial intelligence is the field that builds systems which perform tasks normally associated with human cognition — recognising images, understanding language, planning, playing games. Modern AI is dominated by machine learning, especially neural networks."]),

        # encryption follow-ups
        (["how does encryption work", "how does encryption work?"],
         ["Encryption uses a mathematical key to scramble data so only someone with the matching key can read it. Symmetric encryption uses the same key on both sides; asymmetric encryption (RSA, elliptic curve) uses a public key to encrypt and a private one to decrypt."]),

        # generic 'and X' / 'what about X'
        (["and respiration?", "what about respiration?"],
         ["Cellular respiration is the process by which cells release energy from glucose by reacting it with oxygen. At the carbon-flow level it's roughly the inverse of photosynthesis — animals and plants both use it to release the energy stored in food."]),
    ]

    # --- D4. Person follow-ups (gender-pronoun resolution + facts) ---
    # Direct curated entries for "when did <person> live", "what did
    # <person> discover", "is <person> X" so the resolver-rewritten
    # "when did he live" → "when did einstein live" lands here at high
    # cosine confidence.

    PERSON_FOLLOWUPS: list[tuple[list[str], list[str]]] = [
        # Einstein
        (["when did einstein live", "what years did einstein live",
          "when was einstein alive", "einstein years",
          "when was einstein born", "when did albert einstein live"],
         ["Albert Einstein lived from 1879 to 1955 — a 76-year life that spanned the late 19th and most of the 20th century."]),
        (["was einstein german", "is einstein german", "was albert einstein german"],
         ["Yes — Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, in 1879. He later took Swiss and then American citizenship."],),
        (["what did einstein discover", "what did einstein do",
          "what is einstein famous for", "why is einstein famous"],
         ["Einstein is most famous for the theory of relativity — special relativity in 1905 and general relativity in 1915 — and for the photoelectric effect, which won him the 1921 Nobel Prize in Physics. His equation E=mc² became one of the most recognisable in science."]),
        (["where did einstein live", "where was einstein from"],
         ["Einstein was born in Ulm, Germany in 1879. He later lived and worked in Switzerland, Berlin, and from 1933 onwards in Princeton, New Jersey, where he died in 1955."]),

        # Newton
        (["when did newton live", "what years did newton live",
          "when was newton born", "when did isaac newton live"],
         ["Isaac Newton lived from 1643 to 1727 — an 84-year life through the late 17th and early 18th centuries."]),
        (["what did newton discover", "what did newton do",
          "what is newton famous for", "why is newton famous"],
         ["Isaac Newton formulated the three laws of motion and the law of universal gravitation, and laid the foundations of calculus. His Philosophiæ Naturalis Principia Mathematica, published in 1687, remains one of the most important scientific works ever written."]),
        (["was newton english", "is newton english", "was isaac newton english"],
         ["Yes — Isaac Newton was born in Woolsthorpe, Lincolnshire, England, in 1643."]),

        # Darwin
        (["when did darwin live", "what years did darwin live", "when was darwin born"],
         ["Charles Darwin lived from 1809 to 1882, a 73-year life that spanned the better part of the 19th century."]),
        (["what did darwin discover", "what did darwin do",
          "why is darwin famous", "what is darwin famous for"],
         ["Charles Darwin proposed the theory of evolution by natural selection. His 1859 book On the Origin of Species described how species change over generations through variation and selection, and is one of the founding works of modern biology."]),
        (["was darwin english", "was darwin british"],
         ["Yes — Charles Darwin was born in Shrewsbury, England, in 1809."]),

        # Shakespeare
        (["when did shakespeare live", "what years did shakespeare live",
          "when was shakespeare born", "when did william shakespeare live"],
         ["William Shakespeare lived from 1564 to 1616 — a 52-year life across the late 16th and early 17th centuries, the late Elizabethan and early Jacobean periods."]),
        (["was shakespeare english", "is shakespeare english"],
         ["Yes — William Shakespeare was born in Stratford-upon-Avon, in Warwickshire, England, in 1564."]),
        (["famous play", "what is shakespeare's famous play",
          "what's shakespeare's most famous play", "shakespeare's most famous play"],
         ["Shakespeare's most-performed plays include Hamlet, Romeo and Juliet, Macbeth, Othello, King Lear, and A Midsummer Night's Dream. Hamlet is often considered his single greatest work."]),
        (["what did shakespeare write", "what plays did shakespeare write"],
         ["Shakespeare wrote roughly 39 plays — including Hamlet, Macbeth, Romeo and Juliet, King Lear, Othello, A Midsummer Night's Dream, and The Tempest — plus 154 sonnets and several long narrative poems."]),

        # Mozart
        (["when did mozart live", "what years did mozart live"],
         ["Wolfgang Amadeus Mozart lived from 1756 to 1791 — a brief 35-year life in the classical era of European music."]),
        (["was mozart austrian", "where was mozart from"],
         ["Yes — Mozart was born in Salzburg, in what is today Austria, in 1756."]),
        (["what did mozart compose", "what did mozart write"],
         ["Mozart composed more than 600 works — symphonies, operas (The Marriage of Figaro, Don Giovanni, The Magic Flute), piano concertos, string quartets, and the unfinished Requiem — across virtually every musical form of his era."]),

        # Beethoven
        (["when did beethoven live", "what years did beethoven live"],
         ["Ludwig van Beethoven lived from 1770 to 1827 — straddling the late classical and early romantic eras."]),
        (["was beethoven german", "where was beethoven from"],
         ["Yes — Beethoven was born in Bonn, in what is today Germany, in 1770. He spent most of his career in Vienna."]),
        (["was beethoven deaf", "did beethoven go deaf"],
         ["Yes — Beethoven began losing his hearing in his late twenties and was profoundly deaf by his fifties. He continued composing nonetheless, producing some of his greatest works including the Ninth Symphony entirely without being able to hear them performed."]),

        # Marie Curie
        (["when did curie live", "when did marie curie live", "what years did marie curie live"],
         ["Marie Curie lived from 1867 to 1934, an active scientific career across the late 19th and early 20th centuries."]),
        (["what did marie curie discover", "what did curie discover",
          "why is marie curie famous", "what is marie curie famous for"],
         ["Marie Curie pioneered research on radioactivity. She discovered the elements polonium and radium with her husband Pierre, won the 1903 Nobel Prize in Physics with him for radioactivity, and the 1911 Nobel Prize in Chemistry for the discovery of radium."]),
        (["was marie curie polish", "was curie polish",
          "was marie curie french", "where was marie curie from"],
         ["Marie Curie was born Maria Skłodowska in Warsaw, Poland in 1867, and later took French citizenship after moving to Paris for her studies. She is often described as Polish-French."]),

        # Alan Turing
        (["when did turing live", "when did alan turing live"],
         ["Alan Turing lived from 1912 to 1954 — a tragically short 41-year life cut off by his death in 1954."]),
        (["what did turing do", "what did alan turing do",
          "why is turing famous", "what is alan turing famous for"],
         ["Alan Turing formalised the idea of computation through what we now call Turing machines, helped break German Enigma codes at Bletchley Park during World War Two, and proposed the Turing test as a benchmark for machine intelligence."]),
        (["was turing english", "was alan turing english", "was alan turing british"],
         ["Yes — Alan Turing was born in London in 1912 and was British throughout his life."]),

        # Tesla (the inventor — keep separate from Tesla the company)
        (["when did nikola tesla live", "when did tesla the inventor live"],
         ["Nikola Tesla lived from 1856 to 1943 — a long career across the late 19th and the first half of the 20th century."]),
        (["what did nikola tesla invent",
          "what did tesla the inventor do", "why is nikola tesla famous"],
         ["Nikola Tesla developed alternating-current electrical systems, the AC induction motor, and pioneered work on radio. He's the namesake of the SI unit for magnetic flux density and of Tesla, the electric-vehicle company."]),

        # Stephen Hawking
        (["when did hawking live", "when did stephen hawking live"],
         ["Stephen Hawking lived from 1942 to 2018 — a 76-year life that spanned the post-war era through the early 21st century."]),
        (["what did hawking do", "what did stephen hawking do",
          "why is hawking famous", "what is hawking famous for"],
         ["Stephen Hawking was a British theoretical physicist known for his work on black holes — including Hawking radiation, the prediction that black holes emit thermal radiation — and for the popular book A Brief History of Time."]),
        (["was hawking english", "was stephen hawking english", "was stephen hawking british"],
         ["Yes — Stephen Hawking was born in Oxford, England in 1942, and was British throughout his life."]),

        # Steve Jobs / Bill Gates / Elon Musk / Tim Berners-Lee / Linus Torvalds
        (["who is steve jobs", "who was steve jobs", "tell me about steve jobs"],
         ["Steve Jobs co-founded Apple with Steve Wozniak in 1976, was ousted from the company in 1985, and returned in 1997 to lead Apple's transformation into the iPhone-era consumer-electronics powerhouse. He died in 2011."]),
        (["when did steve jobs live", "what years did steve jobs live"],
         ["Steve Jobs lived from 1955 to 2011 — a 56-year life cut short by pancreatic cancer."]),
        (["who is bill gates", "who was bill gates", "tell me about bill gates"],
         ["Bill Gates co-founded Microsoft with Paul Allen in 1975 and led the company through its dominant rise in PC software. Since stepping back from day-to-day leadership in 2008, he's focused on the Bill and Melinda Gates Foundation, which works on global health and education."]),
        (["who is elon musk", "who was elon musk", "tell me about elon musk"],
         ["Elon Musk is a South African–American entrepreneur who founded or co-founded SpaceX, Tesla, Neuralink, and X (formerly Twitter). He's known for ambitious goals like reusable rockets, mass-market electric cars, and Mars settlement."]),
        (["who is tim berners-lee", "who is berners-lee", "tell me about tim berners-lee",
          "who invented the world wide web", "who invented the web"],
         ["Tim Berners-Lee is the British computer scientist who invented the World Wide Web in 1989 while working at CERN. He designed HTTP, HTML, and the first web browser, and continues to advocate for an open, decentralised web."]),
        (["who is linus torvalds", "tell me about linus torvalds",
          "who created linux", "who invented linux"],
         ["Linus Torvalds is the Finnish-American software engineer who created the Linux kernel in 1991 and the Git version-control system in 2005. He still oversees the Linux kernel's development today."]),
    ]

    # --- E. Comparison handlers --------------------------------------

    COMPARE_PAIRS: list[tuple[list[str], list[str]]] = [
        (
            ["python vs javascript", "python or javascript", "is python better than javascript"],
            ["Python and JavaScript both shine, in different places. Python is the standard for data science, ML, scripting, and back-end work; JavaScript dominates the browser and is strong on the back-end via Node. Pick by where the code needs to run."],
        ),
        (
            ["rust vs c++", "rust or c++", "is rust better than c++"],
            ["Rust gives you C++-class performance with memory safety guarantees enforced at compile time, at the cost of a steeper initial learning curve. C++ has a larger ecosystem and decades of legacy. For new systems code, Rust is increasingly the default."],
        ),
        (
            ["apple vs samsung", "apple or samsung", "is apple better than samsung"],
            ["Apple builds the iPhone, Mac, and a tightly integrated software ecosystem; Samsung builds Galaxy phones plus a much wider catalogue of displays, memory chips, and appliances. Choice is usually iOS-vs-Android plus integration preferences."],
        ),
        (
            ["mac vs pc", "mac or pc", "is a mac better than a pc"],
            ["A Mac gives you a tightly integrated hardware and software experience and excellent build quality. A PC gives you broader hardware choice, easier upgrades, far better gaming support, and lower cost per spec."],
        ),
        (
            ["ios vs android", "is ios better than android"],
            ["iOS is more uniform across devices, with strong privacy defaults and longer software support per phone. Android offers far more hardware choice, deeper customisation, and better integration with Google services. Either is a defensible choice."],
        ),
        (
            ["windows vs mac", "is windows better than mac"],
            ["Windows runs on the widest range of hardware, has the broadest software library, and is the standard for gaming. macOS gives you a polished UNIX-based desktop with tight Apple-hardware integration. Pick by your software needs and hardware preferences."],
        ),
        (
            ["openai vs anthropic", "is openai better than anthropic"],
            ["OpenAI and Anthropic both build frontier large language models, with OpenAI's GPT series and Anthropic's Claude series. They differ in safety methodology and product surface — both are credible choices for building on top of."],
        ),
        (
            ["chatgpt vs claude", "is chatgpt better than claude"],
            ["ChatGPT and Claude are both top-tier chat assistants from frontier labs. They have slightly different defaults — Claude tends to be more cautious and explicit about uncertainty, ChatGPT has the broader plug-in and tooling ecosystem — and capability is roughly comparable for most users."],
        ),
        (
            ["intel vs amd", "is intel better than amd"],
            ["Intel and AMD make x86 CPUs that compete on price and performance per watt. AMD has been strong on multi-core performance per dollar in recent years; Intel still leads in some single-thread workloads. Pick by your specific workload and budget."],
        ),
        (
            ["nvidia vs amd", "is nvidia better than amd for gpus"],
            ["Nvidia leads the GPU market for ML thanks to CUDA and a deep software stack. AMD competes well on raw FLOPs per dollar in gaming and is gaining ground in ML via ROCm, but software maturity still lags."],
        ),
        (
            ["coffee vs tea", "is coffee better than tea"],
            ["Coffee gives you a sharper caffeine kick and a wider flavour space across roasts and origins. Tea offers a gentler lift, a longer ritual, and a different aromatic universe. Most people benefit from having both in rotation."],
        ),
        # Additional comparison pairs covering audit failures
        (
            ["huawei vs apple", "apple vs huawei", "is apple bigger than huawei",
             "is huawei bigger than apple", "compare huawei and apple",
             "what's the difference between apple and huawei"],
            ["Apple is a US consumer-electronics company famous for the iPhone and the Mac, while Huawei is a Chinese telecom and consumer-devices company famous for HarmonyOS, the Mate phones, and 5G networking. Apple has a higher market capitalisation; Huawei employs more people globally and leads in 5G infrastructure."],
        ),
        (
            ["python vs rust", "is python faster than rust", "is rust faster than python",
             "compare python and rust", "what's the difference between python and rust"],
            ["Python is a high-level interpreted language designed for readability and rapid prototyping. Rust is a systems language with manual memory management and zero-cost abstractions, designed for safety and performance. Python is easier to learn; Rust is faster and safer at runtime."],
        ),
        (
            ["chatgpt vs claude", "is chatgpt better than claude",
             "compare chatgpt and claude", "what's the difference between chatgpt and claude"],
            ["ChatGPT (OpenAI) and Claude (Anthropic) are both top-tier chat assistants from frontier labs. They differ slightly in defaults — Claude tends to be more cautious and explicit about uncertainty, ChatGPT has a broader plug-in and tooling ecosystem — and capability is roughly comparable for most everyday tasks."],
        ),
        (
            ["gpt vs bert", "compare gpt and bert", "what's the difference between gpt and bert"],
            ["GPT and BERT are both transformer-based language models. GPT is decoder-only and trained for text generation; BERT is encoder-only and trained on masked-token prediction for classification and understanding. GPT-style models dominate today's chat assistants; BERT remains widely used for embeddings and search."],
        ),
        (
            ["linux vs windows", "is linux better than windows", "compare linux and windows"],
            ["Linux is open-source, scriptable, and dominant on servers and embedded devices. Windows is Microsoft's proprietary OS, dominant on consumer and enterprise desktops and the standard for PC gaming. The choice usually comes down to your software needs and hardware preferences."],
        ),
        (
            ["who came first", "who came first?", "who was first",
             "who was earlier", "who lived first", "who lived earlier"],
            ["Tell me which two people you mean — I can place most pairs of historical figures on the timeline. For example: Newton (1643-1727) came before Einstein (1879-1955), and Shakespeare (1564-1616) came before both."],
        ),
    ]

    # --- F. Conversational acknowledgement (no movie quotes) ---------

    ACK_PAIRS: list[tuple[list[str], list[str]]] = [
        (
            ["i see", "got it", "okay got it", "alright got it", "makes sense", "understood"],
            [
                "Got it. Anything else you'd like to dig into?",
                "Good. Want to keep going on this, or move to something else?",
                "Glad that lands. Where would you like to go next?",
            ],
        ),
        (
            ["interesting", "fascinating", "that's interesting", "thats interesting", "that's fascinating"],
            [
                "Glad you think so. Want me to go further on this, or move to something else?",
                "Right? Tell me which part caught your attention and I can dig in.",
                "Good — there's more under that hood if you want to keep pulling.",
            ],
        ),
        (
            ["tell me more", "more please", "go on", "continue", "elaborate"],
            [
                "Happy to. Which angle do you want — context, mechanism, or examples?",
                "Sure. Anchor me: which part should I go deeper on?",
                "Of course. Want the technical version or the plain-English version?",
            ],
        ),
        (
            ["right", "exactly", "precisely", "correct"],
            [
                "Glad we're aligned. Where to next?",
                "Good. Is there a follow-up you want to chase down?",
                "Right. Anything else worth pulling on?",
            ],
        ),
        (
            ["wow", "whoa", "no way", "really"],
            [
                "Right? It's one of those things that sounds small until you look closely.",
                "Genuinely. Want me to go further on it?",
                "Yeah, it surprised me too the first time.",
            ],
        ),
        (
            ["hmm", "hmm interesting", "let me think", "thinking", "interesting hmm"],
            [
                "Take your time. I'll be here when you're ready.",
                "No rush. Want me to lay out the options while you think?",
                "Sure. Want a different angle while you're chewing on that?",
            ],
        ),
        (
            ["why", "why is that", "but why", "why so"],
            [
                "Good question — let me unpack the cause. What aspect do you want to start with?",
                "Worth asking. Want the short version or the long one?",
                "Because there are usually layers. Which layer do you want — historical, mechanistic, or practical?",
            ],
        ),
    ]

    # --- G. Single-question follow-ups (audit gaps: location, ceo, etc) ---
    # Short bare-noun follow-ups that the audit found landing on movie
    # quotes / single-token replies — even with topic-carryover prefix
    # active, having direct curated entries makes the path bullet-proof.

    SINGLE_NOUN_FOLLOWUPS: list[tuple[list[str], list[str]]] = [
        (
            ["location", "where", "their location", "headquarters"],
            [
                "Tell me which company or place you mean and I'll give you the headquarters.",
                "Could you name the company? I'll point you at the head office.",
                "Headquarters of which one? I have entries for the major tech and consumer-electronics firms.",
            ],
        ),
        (
            ["ceo", "the ceo", "who's the ceo", "who runs it"],
            [
                "Tell me which company and I'll give you the current CEO.",
                "Which company's CEO did you have in mind?",
                "Name the firm and I'll tell you who runs it.",
            ],
        ),
        (
            ["founder", "founders", "who founded it", "who started it"],
            [
                "Tell me which company and I'll give you the founders.",
                "Which one — Apple, Microsoft, Huawei, or someone else?",
                "Name the firm and I'll point at the founders.",
            ],
        ),
        (
            ["products", "their products", "what do they sell"],
            [
                "Which company are we talking about? I'll list their main products.",
                "Tell me the company and I'll give you the headline products.",
            ],
        ),
        (
            ["history", "the history", "background"],
            [
                "Background of which one? Tell me the company or topic.",
                "Sure — name the subject and I'll lay out the background.",
            ],
        ),
    ]

    # --- H. Common command-with-no-context handlers -------------------

    COMMAND_PAIRS: list[tuple[list[str], list[str]]] = [
        (
            ["give me a quote", "quote please", "another quote", "say a quote", "any quotes", "share a quote"],
            QUOTE_POOL,
        ),
        (
            ["tell me a joke", "joke please", "another joke", "say something funny", "got any jokes", "make me laugh", "tell me another joke"],
            JOKE_POOL,
        ),
        (
            ["tell me a fact", "fact please", "interesting fact", "give me a fact", "share a fact", "another fact", "fun fact"],
            FACT_POOL,
        ),
        (
            ["tell me a story", "story please", "tell me a short story", "another story", "share a story"],
            STORY_POOL,
        ),
        (
            ["recipe please", "give me a recipe", "share a recipe", "another recipe", "what's a good recipe"],
            RECIPE_POOL,
        ),
        (
            ["write a poem", "poem please", "share a poem", "another poem", "give me a poem"],
            POEM_POOL,
        ),
        (
            ["recommend a book", "book recommendation", "what should i read", "any book recs", "another book", "suggest a book"],
            BOOK_REC_POOL,
        ),
        (
            ["give me advice", "any advice", "share some advice", "another piece of advice", "advice please"],
            ADVICE_POOL,
        ),
        (
            ["translate this", "can you translate", "translate for me", "do you translate"],
            TRANSLATE_POOL,
        ),
        (
            ["summarise this", "summarize this", "can you summarise", "summary please"],
            [
                "Of course — paste the text you'd like summarised and I'll boil it down to its key points.",
                "Happy to. Drop the passage in and I'll pull out the essentials.",
                "Sure. Share the content and tell me how short you want the summary.",
            ],
        ),
        (
            ["explain this", "can you explain this", "what does this mean", "help me understand"],
            [
                "Of course. Paste what you'd like me to explain and tell me what level of background you have.",
                "Happy to break it down. Share the content and I'll work through it step by step.",
                "Sure — share the text and I'll explain it as plainly as I can.",
            ],
        ),
        (
            ["help me with code", "code help", "i need coding help", "can you help with code"],
            [
                "Sure. Paste the code and tell me what you want it to do versus what it's actually doing.",
                "Happy to take a look. What language, and what's the symptom?",
                "Of course. Drop the snippet and the error message — I'll work through it with you.",
            ],
        ),
    ]

    # Folder lists into qa_exchanges so the existing sampling loop sees
    # them.  We *also* emit deterministic dialogues from each pair below
    # so coverage isn't left to the random walker.
    NEW_POOLS: list[tuple[list[str], list[str]]] = []
    NEW_POOLS.extend(explain_pairs)
    NEW_POOLS.extend(META_PAIRS)
    NEW_POOLS.extend(ENTITY_FOLLOWUPS)
    NEW_POOLS.extend(COUNTRY_FOLLOWUPS)
    NEW_POOLS.extend(BARE_TOPIC_FOLLOWUPS)
    NEW_POOLS.extend(PERSON_FOLLOWUPS)
    NEW_POOLS.extend(COMPARE_PAIRS)
    NEW_POOLS.extend(ACK_PAIRS)
    NEW_POOLS.extend(SINGLE_NOUN_FOLLOWUPS)
    NEW_POOLS.extend(COMMAND_PAIRS)

    # --- I. Small-talk gap-fillers from the audit's OOD fall-throughs --
    SMALLTALK_PAIRS: list[tuple[list[str], list[str]]] = [
        (
            ["are you still there", "are you still there?", "are you there",
             "still there", "you there", "hello?", "still around"],
            [
                "Yep — still here. What can I help with?",
                "Yes, still around. What's on your mind?",
                "I'm here. Take your time.",
            ],
        ),
        (
            ["what's up", "whats up", "sup", "what is up", "what up"],
            [
                "Not much on my end — just here. What's new with you?",
                "All quiet. Anything you'd like to talk about?",
                "Same as ever. What can I help with today?",
            ],
        ),
        (
            ["what's new", "whats new", "what is new", "anything new"],
            [
                "Nothing new on my side — I run fresh each session. What's new with you?",
                "Same model as last time, no updates between sessions. Anything new with you?",
                "Quiet here. What's been going on for you?",
            ],
        ),
        (
            ["i feel lonely", "feeling lonely", "i'm lonely", "im lonely"],
            [
                "That's a hard feeling. Is there someone you've been meaning to reach out to?",
                "I'm sorry to hear that. Sometimes naming it helps. Want to talk about what's been going on?",
                "Loneliness is real. Would it help to talk about it for a bit?",
            ],
        ),
        (
            ["i'm anxious", "feeling anxious", "i feel anxious", "im anxious"],
            [
                "Anxiety is exhausting. Do you know what's prompting it, or is it more of a general fog?",
                "I hear you. Want to talk about what's underneath it?",
                "That's tough. Take a slow breath if you can — what's the most pressing piece?",
            ],
        ),
        (
            ["where is tokyo", "where is tokyo located"],
            ["Tokyo is the capital of Japan, located on the eastern coast of the main island of Honshu."],
        ),
        (
            ["where is london", "where is london located"],
            ["London is the capital of the United Kingdom, located in south-east England on the River Thames."],
        ),
        (
            ["where is paris", "where is paris located"],
            ["Paris is the capital of France, located in the north of the country on the River Seine."],
        ),
        (
            ["where is berlin", "where is berlin located"],
            ["Berlin is the capital of Germany, located in the north-east of the country."],
        ),
        (
            ["where is shenzhen", "where is shenzhen located"],
            ["Shenzhen is a major city in southern China, just across the border from Hong Kong, and is home to Huawei among many other technology companies."],
        ),
        (
            ["translate hello to japanese", "how do you say hello in japanese"],
            ["Hello in Japanese is konnichiwa (こんにちは). For a more casual hi, people also say yaa or osu among friends."],
        ),
        (
            ["translate hello to french", "how do you say hello in french"],
            ["Hello in French is bonjour, used during the day, and bonsoir in the evening. Salut is a casual hi or bye."],
        ),
        (
            ["translate hello to spanish", "how do you say hello in spanish"],
            ["Hello in Spanish is hola. For a slightly more formal greeting use buenos días, buenas tardes, or buenas noches depending on the time of day."],
        ),
    ]
    NEW_POOLS.extend(SMALLTALK_PAIRS)

    qa_exchanges.extend(NEW_POOLS)

    # ------------------------------------------------------------------
    # Factoid knowledge bank — richer vocabulary at scale.  Each entry is
    # one concept with several paraphrased user queries and a single
    # factually-correct response.  The point is two-fold:
    #   1) Retrieval coverage: the user can ask "what is photosynthesis"
    #      or "tell me about photosynthesis" and land on the same answer.
    #   2) Vocabulary growth: the response prose introduces domain words
    #      ("chlorophyll", "symphony", "cumulonimbus") that would never
    #      appear in a pure chit-chat corpus, so the tokenizer ends up
    #      with a materially wider coverage.
    # Every factoid keyword below produces triples of the form
    # ``[f"what is {kw}", f"tell me about {kw}"] -> answer``, which then
    # get folded into ``qa_exchanges`` before dialogue sampling.
    # ------------------------------------------------------------------
    factoids: dict[str, str] = {
        # Science — physics
        "photosynthesis": "Photosynthesis is the process plants use to convert sunlight, carbon dioxide, and water into glucose and oxygen. The chlorophyll in leaves captures the light energy.",
        "thermodynamics": "Thermodynamics is the branch of physics describing heat, work, temperature, and entropy. Its four laws govern how energy flows through physical systems.",
        "relativity": "Einstein's theory of relativity has two parts: special relativity, relating space and time for observers in motion, and general relativity, which describes gravity as the curvature of spacetime.",
        "quantum mechanics": "Quantum mechanics is the physics of the very small. Particles are described by probability waves, measurement changes the system, and quantities like energy come in discrete packets.",
        "entropy": "Entropy is a measure of disorder in a physical system. The second law of thermodynamics says that total entropy never decreases in an isolated system.",
        "magnetism": "Magnetism is a force produced by moving electric charges. It's one of the four fundamental interactions and is deeply linked to electricity through the Maxwell equations.",
        "electricity": "Electricity is the flow of electric charge, usually carried by electrons moving through a conductor. Voltage pushes the charges, resistance slows them, and current measures their flow.",
        "sound": "Sound is a mechanical wave of pressure variations travelling through a medium like air or water. Its frequency determines pitch and its amplitude determines loudness.",
        "light": "Light is electromagnetic radiation that the human eye can detect. It behaves both as a wave and as a stream of particles called photons.",
        "gravity": "Gravity is the attractive force between objects with mass. On Earth it accelerates falling objects at about 9.8 metres per second squared.",
        # Science — chemistry
        "photon": "A photon is the elementary particle of light. It has no mass, travels at the speed of light, and carries energy proportional to its frequency.",
        "molecule": "A molecule is a group of atoms held together by chemical bonds. Water and oxygen gas are familiar examples.",
        "acid": "An acid is a substance that donates a proton or accepts an electron pair in a chemical reaction. Strong acids dissociate completely in water.",
        "base": "In chemistry, a base is a substance that accepts a proton or donates an electron pair. Bases neutralise acids to form salts.",
        "ph": "pH is a scale from 0 to 14 measuring how acidic or alkaline a solution is. Pure water is neutral at pH 7.",
        "carbon": "Carbon is a non-metallic element that forms the backbone of organic chemistry. It bonds with itself and many other elements, enabling the diversity of life.",
        "oxygen": "Oxygen is a chemical element essential for most life on Earth. It makes up about 21 percent of the atmosphere and is critical for combustion and respiration.",
        "hydrogen": "Hydrogen is the lightest and most abundant element in the universe. It consists of a single proton and a single electron and fuels stars through fusion.",
        # Science — biology
        "dna": "DNA is the molecule that stores the genetic instructions of every living organism. Its double-helix structure was described by Watson and Crick in 1953.",
        "rna": "RNA is a single-stranded nucleic acid that translates genetic instructions from DNA into proteins. It plays many other regulatory roles in the cell.",
        "protein": "A protein is a large molecule built from amino-acid chains. Proteins catalyse reactions, transport molecules, and form the scaffolding of cells.",
        "cell": "A cell is the fundamental unit of life. All living organisms are built from cells, which in turn contain smaller components called organelles.",
        "mitosis": "Mitosis is the process by which a cell divides to produce two genetically identical daughter cells. It drives growth and tissue repair.",
        "evolution": "Evolution is the process by which populations change over generations. Variation, selection, and inheritance together produce adaptation over time.",
        "ecosystem": "An ecosystem is a community of organisms interacting with their physical environment. Energy flows from producers to consumers and nutrients cycle through it.",
        "bacteria": "Bacteria are single-celled organisms without a nucleus. They were among the earliest forms of life and still dominate most of the biosphere.",
        "virus": "A virus is a microscopic agent that replicates only inside the living cells of a host. It is not usually considered alive on its own.",
        "brain": "The brain is the centre of the nervous system. In humans, about 86 billion neurons form the circuits that underlie perception, thought, and action.",
        "neuron": "A neuron is a specialised cell that transmits information through electrochemical signals. Networks of neurons form the basis of the nervous system.",
        "immune system": "The immune system is a network of cells and proteins that defend the body against infection. It distinguishes self from non-self using molecular signatures.",
        # Geography — countries
        "germany": "Germany is a country in central Europe with the largest economy in the European Union. Its capital is Berlin.",
        "france": "France is a country in western Europe known for its cuisine, art, and history. Its capital is Paris.",
        "japan": "Japan is an island nation in East Asia composed of four major islands. Its capital is Tokyo.",
        "china": "China is the most populous country in East Asia and one of the world's largest economies. Its capital is Beijing.",
        "india": "India is a country in South Asia with more than a billion inhabitants and thousands of years of cultural history. Its capital is New Delhi.",
        "brazil": "Brazil is the largest country in South America. It spans a huge section of the Amazon rainforest and its capital is Brasília.",
        "russia": "Russia is the largest country on Earth by area, spanning eleven time zones across Europe and Asia. Its capital is Moscow.",
        "united kingdom": "The United Kingdom is a country in north-western Europe made up of England, Scotland, Wales, and Northern Ireland. Its capital is London.",
        "canada": "Canada is the second-largest country by area and lies north of the United States. Its capital is Ottawa and it is officially bilingual in English and French.",
        "australia": "Australia is a country and a continent in the southern hemisphere. Its capital is Canberra and its unique wildlife includes kangaroos and koalas.",
        "italy": "Italy is a country in southern Europe shaped like a boot. Its capital is Rome and it has a long history spanning the Roman Empire to the Renaissance.",
        "spain": "Spain is a country in south-western Europe with strong regional cultures. Its capital is Madrid.",
        "egypt": "Egypt is a country straddling north-east Africa and the Sinai Peninsula. Its capital is Cairo and it is famous for the pyramids at Giza.",
        "greece": "Greece is a country in south-eastern Europe known for ancient philosophy, mythology, and more than six thousand islands. Its capital is Athens.",
        # Geography — natural features
        "amazon": "The Amazon is a vast rainforest in South America that straddles nine countries. It produces roughly 20 percent of the atmosphere's oxygen.",
        "sahara": "The Sahara is the largest hot desert in the world, covering most of northern Africa. It spans more than nine million square kilometres.",
        "himalayas": "The Himalayas are a mountain range separating the Indian subcontinent from the Tibetan Plateau. They contain Mount Everest, the highest peak above sea level.",
        "pacific ocean": "The Pacific is the largest and deepest of Earth's oceans. It covers about one-third of the planet's surface.",
        "nile": "The Nile is one of the longest rivers in the world, flowing more than six thousand kilometres through north-east Africa into the Mediterranean.",
        "antarctica": "Antarctica is the southernmost continent, covered almost entirely by ice. It is governed by an international treaty for scientific research.",
        # Astronomy
        "sun": "The Sun is the star at the centre of our solar system. It is a main-sequence yellow dwarf about 150 million kilometres from Earth.",
        "moon": "The Moon is Earth's only natural satellite. It orbits Earth every 27.3 days and drives the tides through its gravitational pull.",
        "mars": "Mars is the fourth planet from the Sun and is sometimes called the Red Planet because of iron oxide on its surface. It has two small moons, Phobos and Deimos.",
        "jupiter": "Jupiter is the largest planet in the solar system. It's a gas giant with a distinctive banded atmosphere and a great red spot that is itself a storm larger than Earth.",
        "saturn": "Saturn is the sixth planet from the Sun and is famous for its prominent ring system, which is mostly ice particles.",
        "milky way": "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy and holds hundreds of billions of stars.",
        "galaxy": "A galaxy is a huge gravitationally-bound system of stars, interstellar gas, dust, and dark matter. The observable universe contains hundreds of billions of them.",
        "universe": "The universe is all of space, time, matter, and energy considered as a whole. Modern cosmology traces it back about 13.8 billion years to the Big Bang.",
        "big bang": "The Big Bang is the leading theory for how the universe began. Roughly 13.8 billion years ago, everything expanded from an extremely hot and dense state.",
        # History
        "roman empire": "The Roman Empire was a vast ancient state centred on the Mediterranean. At its height in the second century it stretched from Britain to Mesopotamia.",
        "renaissance": "The Renaissance was a cultural movement from roughly the 14th to 17th century in Europe. It revived classical learning and produced figures like Leonardo da Vinci and Michelangelo.",
        "industrial revolution": "The Industrial Revolution was the shift from agrarian economies to industrial ones that began in Britain in the late 18th century. Steam power and mechanisation transformed production.",
        "world war one": "World War One was a global conflict from 1914 to 1918 between the Allied and Central Powers. It ended with the collapse of several empires and reshaped the 20th century.",
        "world war two": "World War Two was a global conflict from 1939 to 1945 involving most of the world's nations. It ended with the defeat of the Axis powers and the start of the nuclear age.",
        "cold war": "The Cold War was a prolonged geopolitical rivalry between the United States and the Soviet Union and their allies that lasted from 1947 to 1991.",
        # People — scientists
        "einstein": "Albert Einstein was a German-born physicist who developed the theories of special and general relativity. He received the Nobel Prize in Physics in 1921.",
        "newton": "Isaac Newton was an English mathematician and physicist who described the laws of motion and universal gravitation in the 17th century.",
        "darwin": "Charles Darwin was a British naturalist who proposed the theory of evolution by natural selection in his 1859 book On the Origin of Species.",
        "curie": "Marie Curie was a Polish-French physicist and chemist who pioneered research on radioactivity and won Nobel Prizes in both physics and chemistry.",
        "turing": "Alan Turing was a British mathematician and logician who formalised the idea of computation and helped break German codes during the Second World War.",
        "tesla": "Nikola Tesla was a Serbian-American inventor and engineer known for developing the alternating-current electrical system and pioneering work on radio.",
        "hawking": "Stephen Hawking was a British theoretical physicist known for his work on black holes and cosmology, and for the popular book A Brief History of Time.",
        # People — writers / artists
        "shakespeare": "William Shakespeare was an English playwright and poet active around 1600. He wrote works like Hamlet, Macbeth, and Romeo and Juliet that remain central to literature.",
        "dostoevsky": "Fyodor Dostoevsky was a 19th-century Russian novelist whose works, including Crime and Punishment and The Brothers Karamazov, explore morality and the human mind.",
        "tolstoy": "Leo Tolstoy was a 19th-century Russian novelist whose masterpieces include War and Peace and Anna Karenina.",
        "leonardo da vinci": "Leonardo da Vinci was an Italian polymath of the Renaissance known for paintings like the Mona Lisa and detailed notebooks on anatomy and engineering.",
        "mozart": "Wolfgang Amadeus Mozart was an 18th-century Austrian composer. He produced more than six hundred works spanning symphonies, operas, and chamber music.",
        "beethoven": "Ludwig van Beethoven was a German composer and pianist active in the late 18th and early 19th centuries. He continued composing even after becoming deaf.",
        # Technology / computing
        "internet": "The internet is a global system of interconnected computer networks. Its protocols let billions of devices share data regardless of hardware or operating system.",
        "algorithm": "An algorithm is a finite sequence of well-defined instructions for solving a problem or performing a computation.",
        "programming": "Programming is the activity of writing instructions for computers to execute. A programming language lets a person express those instructions precisely.",
        "operating system": "An operating system manages a computer's hardware and software resources. It mediates between programs and devices like storage, networks, and input peripherals.",
        "machine learning": "Machine learning is the field that studies algorithms which improve automatically from data. Common techniques include regression, classification, and neural networks.",
        "neural network": "A neural network is a computational model loosely inspired by biological neurons. Layers of connected units learn patterns by adjusting their connection weights.",
        "transformer": "A transformer is a neural-network architecture built around self-attention. It processes sequences in parallel and is the foundation of modern language models.",
        "tokenizer": "A tokenizer splits input text into units a neural network can embed. Units might be words, subwords, or characters depending on the scheme.",
        "embedding": "An embedding is a numerical vector representation of a discrete item like a word or a user. Similar items end up close together in the embedding space.",
        "attention": "Attention is a mechanism that lets a neural network weight different parts of its input by relevance to the current step. It is the core idea behind the transformer.",
        "gradient descent": "Gradient descent is an iterative optimisation algorithm. It moves the parameters of a model in the direction that reduces the loss on the training data.",
        "overfitting": "Overfitting happens when a model memorises its training data at the cost of generalising to new inputs. Regularisation and more data are the usual remedies.",
        "reinforcement learning": "Reinforcement learning is a framework where an agent learns by interacting with an environment. It chooses actions to maximise a cumulative reward signal.",
        # Philosophy
        "philosophy": "Philosophy is the systematic study of fundamental questions about knowledge, existence, values, reason, and language.",
        "ethics": "Ethics is the branch of philosophy that examines right and wrong conduct. Major traditions include consequentialism, deontology, and virtue ethics.",
        "logic": "Logic is the study of valid reasoning. It supplies the rules by which arguments can be judged sound regardless of their subject matter.",
        "stoicism": "Stoicism is a Hellenistic philosophy that stresses virtue, reason, and acceptance of what lies outside one's control. Key figures include Epictetus and Marcus Aurelius.",
        # Arts / culture
        "music": "Music is an art of organised sound across time. Elements include melody, harmony, rhythm, timbre, and form, and traditions exist in every human culture.",
        "jazz": "Jazz is a music genre that originated among African-American communities in the early 20th century. It features improvisation, swing, and blue notes.",
        "cinema": "Cinema is the art of motion pictures. Early films were silent; later developments brought synchronised sound, colour, widescreen, and digital imaging.",
        "literature": "Literature is writing valued for artistic merit. It spans poetry, drama, fiction, and non-fiction across thousands of years of human culture.",
        # Sports
        "football": "Football, called soccer in some countries, is a team sport played between two sides of eleven players on a rectangular pitch. The FIFA World Cup is its most-watched event.",
        "tennis": "Tennis is a racquet sport played between two or four players across a net. Grand Slam tournaments include Wimbledon, the Australian Open, Roland Garros, and the US Open.",
        "basketball": "Basketball is a team sport in which two teams of five shoot a ball through a raised hoop. It was invented in 1891 by James Naismith.",
        "olympics": "The Olympic Games are the most prestigious international athletic competition, alternating between summer and winter events every two years since 1896.",
        # Food / cooking
        "bread": "Bread is a staple food made from dough of flour and water, usually leavened with yeast. It takes countless regional forms from baguettes to naan.",
        "rice": "Rice is a cereal grain cultivated for thousands of years and a staple food for more than half the world. It thrives in warm, water-rich environments.",
        "coffee": "Coffee is a brewed beverage made from roasted coffee beans. It contains caffeine, which is the most widely-consumed psychoactive substance in the world.",
        "tea": "Tea is a brewed beverage made from the leaves of Camellia sinensis. Traditions differ across cultures, from Japanese matcha to British afternoon tea.",
        "chocolate": "Chocolate is made from the seeds of the cacao tree. It began as a bitter drink in Mesoamerica and evolved into countless modern sweet varieties.",
        # Languages
        "english": "English is a West Germanic language that originated in medieval England. It has borrowed extensively from French and Latin and is now a global lingua franca.",
        "french": "French is a Romance language descended from Latin and spoken natively in France and many former French colonies across Africa and the Americas.",
        "mandarin": "Mandarin is the official language of China and the most spoken language in the world by native speakers. It uses tones to distinguish meaning.",
        "arabic": "Arabic is a Semitic language with more than 400 million speakers. It is the liturgical language of Islam and uses a distinctive script written right to left.",
        "spanish": "Spanish is a Romance language originating on the Iberian Peninsula. It's the official language in more than twenty countries across four continents.",
        # Economics / concepts
        "inflation": "Inflation is the general increase in prices over time, which reduces the purchasing power of money. Central banks typically target a low and stable rate.",
        "recession": "A recession is a significant decline in economic activity lasting more than a few months. It usually shows up as falling output, rising unemployment, and reduced spending.",
        "democracy": "Democracy is a system of government in which citizens hold power through elections and protected civil liberties. Variants include direct and representative forms.",
        "capitalism": "Capitalism is an economic system in which private individuals or firms own the means of production and operate for profit within markets.",
        # Health / wellbeing
        "sleep": "Sleep is a natural state of rest characterised by reduced responsiveness and cyclic brain activity. Adults typically need around seven to nine hours per night.",
        "exercise": "Regular physical exercise improves cardiovascular health, mood, and cognitive function. Public health guidelines suggest at least 150 minutes of moderate activity per week.",
        "meditation": "Meditation is a family of practices that train attention and awareness. Consistent practice is linked to reduced stress and improved focus.",
        "nutrition": "Nutrition is the study of how food affects the body. A balanced diet provides carbohydrates, fats, proteins, vitamins, minerals, and water in suitable proportions.",
    }

    # Translate factoid dict into qa_exchanges triples.
    for topic, answer in factoids.items():
        user_phrasings = [
            f"what is {topic}",
            f"what's {topic}",
            f"whats {topic}",
            f"tell me about {topic}",
            f"explain {topic}",
            f"can you explain {topic}",
            f"define {topic}",
            f"give me a short overview of {topic}",
            f"i want to know about {topic}",
        ]
        qa_exchanges.append((user_phrasings, [answer]))

    # Biography phrasings for people: "who is X", "who was X", etc.
    people_topics = [
        "einstein", "newton", "darwin", "curie", "turing", "tesla", "hawking",
        "shakespeare", "dostoevsky", "tolstoy", "leonardo da vinci", "mozart", "beethoven",
    ]
    for topic in people_topics:
        answer = factoids[topic]
        qa_exchanges.append(
            (
                [
                    f"who is {topic}",
                    f"who was {topic}",
                    f"tell me about {topic}",
                    f"what did {topic} do",
                ],
                [answer],
            )
        )

    # Country-style phrasings.
    country_topics = [
        "germany", "france", "japan", "china", "india", "brazil", "russia",
        "united kingdom", "canada", "australia", "italy", "spain", "egypt", "greece",
    ]
    for topic in country_topics:
        answer = factoids[topic]
        qa_exchanges.append(
            (
                [
                    f"tell me about {topic}",
                    f"what is {topic} like",
                    f"where is {topic}",
                    f"what's the capital of {topic}",
                    f"capital of {topic}",
                ],
                [answer],
            )
        )

    # ------------------------------------------------------------------
    # Style-axis variants.  After the model has learned the base Q→A
    # mapping, we want additional training data that shows how the SAME
    # input can be answered in different *styles* (formal / casual /
    # concise / verbose / emotional / accessible) so the cross-attention
    # conditioning learns to steer output along those axes.  Each entry
    # below is the same user prompt with its response rewritten for a
    # specific style label.
    # ------------------------------------------------------------------
    style_variants: list[tuple[str, str, str]] = [
        # (user_prompt, style, response)
        ("hello", "formal", "Good day to you. How may I assist?"),
        ("hello", "casual", "Hey! What's up?"),
        ("hello", "concise", "Hi."),
        ("hello", "verbose", "Hello, and welcome — thank you for stopping by. What can I help you with today?"),
        ("hello", "accessible", "Hi. I'm here to help. Ask me anything."),
        ("hello", "warm", "Hi there! Always nice to hear from you."),
        ("how are you", "formal", "I am well, thank you for inquiring."),
        ("how are you", "casual", "Pretty good, you?"),
        ("how are you", "concise", "Good, thanks."),
        ("how are you", "verbose", "I'm doing quite well, thanks for asking. It's been a steady day and I'm happy to chat."),
        ("how are you", "accessible", "I'm fine. How about you?"),
        ("thanks", "formal", "You are most welcome."),
        ("thanks", "casual", "No problem!"),
        ("thanks", "concise", "Sure."),
        ("i'm tired", "warm", "That sounds rough. Be gentle with yourself today."),
        ("i'm tired", "direct", "Fair. Is it physical tired, or overloaded tired?"),
        ("i'm tired", "formal", "I'm sorry to hear that — would you like to discuss what's draining your energy?"),
        ("help", "concise", "Tell me the task."),
        ("help", "verbose", "Of course. Walk me through what you're trying to do and any steps you've already tried — then I can offer a plan."),
        ("help", "warm", "Always. What's going on?"),
    ]

    dialogues: list[dict[str, Any]] = []

    # 0) Deterministic coverage pass (Fix 4, 2026-04-25 corpus quality
    #    overhaul).  The stochastic ``random.choice(qa_exchanges)`` loop
    #    below leaves long-tail entries under-sampled; with ~hundreds of
    #    pools that we just expanded, many newly-added prompt variants
    #    might appear zero times after sampling.  We pre-emit ONE
    #    dialogue per (prompt, response) cross-product over EVERY
    #    qa_exchanges entry so coverage is guaranteed regardless of the
    #    sampler.  Each dialogue is tagged ``kind="qa"`` so it survives
    #    the retriever's filler filter and lands in the curated index.
    n_deterministic = 0
    for user_pool, asst_pool in qa_exchanges:
        for user_utt in user_pool:
            for asst_utt in asst_pool:
                dialogues.append({
                    "utterances": [user_utt, asst_utt],
                    "emotions": [0, 0],
                    "kind": "qa",
                })
                n_deterministic += 1

    # 1) Paired Q→A exchanges — the bulk of the corpus.  Produce
    #    ``n_dialogues * 0.7`` two-turn dialogues so the
    #    history-window of 3 always sees a proper user turn before the
    #    response.
    n_paired = int(n_dialogues * 0.7)
    for _ in range(n_paired):
        user_pool, asst_pool = random.choice(qa_exchanges)
        user_utt = random.choice(user_pool)
        asst_utt = random.choice(asst_pool)
        dialogues.append({
            "utterances": [user_utt, asst_utt],
            "emotions": [0, 0],
            "kind": "qa",
        })

    # 2) Style-variant Q→A — same prompt, stylistically varied response.
    n_style = int(n_dialogues * 0.2)
    for _ in range(n_style):
        user_prompt, _style, resp = random.choice(style_variants)
        dialogues.append({
            "utterances": [user_prompt, resp],
            "emotions": [0, 0],
            "kind": "qa",
        })

    # 3) Multi-turn fillers from the old pools so the conditioning
    #    signal still sees length + formality variance.  These mostly
    #    teach the tokenizer wider vocabulary.
    all_pools = [
        casual_responses, formal_responses, emotional_responses,
        complex_responses, simple_responses, follow_ups,
    ]
    n_filler = n_dialogues - n_paired - n_style
    for _ in range(max(0, n_filler)):
        n_turns = random.randint(3, 6)
        utterances = [random.choice(greetings)]
        emotions = [0]
        for _ in range(n_turns - 1):
            pool = random.choice(all_pools)
            utterances.append(random.choice(pool))
            if pool is emotional_responses:
                emotions.append(random.choice([4, 5, 6]))
            elif pool is casual_responses:
                emotions.append(0)
            elif pool is formal_responses:
                emotions.append(0)
            else:
                emotions.append(random.randint(0, 6))
        dialogues.append({"utterances": utterances, "emotions": emotions, "kind": "filler"})

    random.shuffle(dialogues)
    logger.info(
        "Generated %d synthetic dialogues "
        "(deterministic_qa=%d, paired=%d, style=%d, filler=%d)",
        len(dialogues),
        n_deterministic,
        n_paired,
        n_style,
        n_filler,
    )
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
                # ``kind`` is propagated from the source dialogue so the
                # retrieval index can keep only the high-precision Q→A
                # entries and skip the encyclopedia filler.  Defaults
                # to ``filler`` when the dialogue didn't mark itself.
                "kind": dialogue.get("kind", "filler"),
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

    # Fallback to synthetic if no real data found.  We generate a large
    # corpus because the retrieval layer (which is the primary response
    # path for this project) benefits directly from richer coverage:
    # more paraphrases per intent means a wider catchment for user
    # inputs the model sees at inference time.
    if not dialogues:
        logger.warning(
            "No real dialogue data found. Generating synthetic fallback."
        )
        dialogues = generate_synthetic_dialogues(n_dialogues=30000, seed=seed)

        # Append an HMI-domain reference corpus to broaden the
        # tokenizer's vocabulary with terms directly relevant to
        # Huawei's HMI brief — devices, edge ML, privacy,
        # accessibility.  These sentences are *pretraining-style*:
        # they're there for lexical coverage, not for Q→A behaviour.
        # The retrieval layer won't match on them because the tight
        # keyword-overlap veto discards sentence pairs that don't
        # share a content word with the user prompt.
        try:
            from training.download_books import load_gutenberg_dialogues

            domain_corpus = load_gutenberg_dialogues(seed=seed)
        except Exception:
            try:
                # Script-local fallback for direct invocation:
                # ``python training/prepare_dialogue.py`` doesn't set
                # up the package properly.
                from download_books import load_gutenberg_dialogues

                domain_corpus = load_gutenberg_dialogues(seed=seed)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "HMI-domain corpus unavailable (%s); skipping.", exc
                )
                domain_corpus = []
        logger.info(
            "Appending %d HMI-domain dialogues for vocabulary coverage.",
            len(domain_corpus),
        )
        dialogues.extend(domain_corpus)

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

    # Save the raw (history, response, emotion) triples as JSON so the
    # pipeline engine can build a retrieval index at runtime without
    # having to reconstruct the corpus from the tokenised .pt tensors.
    # This is what powers the hybrid "retrieve-then-optionally-generate"
    # response path — edge assistants do this routinely because pure
    # autoregressive generation from a 4 M-param model is never fluent.
    import json
    triples_path = out_dir / "triples.json"
    with open(triples_path, "w", encoding="utf-8") as fh:
        json.dump(triples, fh, ensure_ascii=False, indent=2)
    logger.info("Triples saved to: %s (%d entries)", triples_path, len(triples))

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
        "--vocab-size",  # default raised to 30000 per Huawei brief.
        type=int,
        default=30000,
        help="Vocabulary size for the tokenizer (default: 30000).",
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

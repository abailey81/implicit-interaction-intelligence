"""Hybrid retrieval layer for the on-device SLM stack.

Edge-deployed assistants pair a small autoregressive model with a
lightweight retrieval index because a small transformer is not fluent
enough alone to carry every turn.  This module implements the
retrieval half plus a small set of *tool routes* (math solver,
hostility refusal) that are answered deterministically rather than by
either retrieval or generation:

1. Build an in-memory index of ``(history_text, response_text)`` pairs
   from the training triples (``data/processed/dialogue/triples.json``).
2. Embed every history prompt with the SLM's token embedding layer
   (mean-pooled over tokens) — same tokeniser, same device, no extra
   dependency.
3. At inference time embed the user prompt the same way and return the
   top-``k`` closest training prompt's paired response.

The index is deterministic: same prompt, same response.  Cosine
similarity is computed with a single matrix-vector product so a query
on a 5 000-entry index is sub-millisecond on CPU.

The retriever is intentionally *stateless* — the ``SLMGenerator`` owns
its lifetime, initialises it lazily, and combines its output with the
adaptation vector (for style bias) and with the autoregressive model
(as a novelty fall-back).

Written from scratch, ``torch``-only, no ``faiss`` / ``sentence-
transformers`` / ``sklearn`` dependencies.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool route: safe arithmetic evaluator
# ---------------------------------------------------------------------------
# Phase 1 fix: "1+2123" was previously being retrieved against the
# dialogue corpus and getting a nonsense greeting back.  A tiny AST-based
# expression evaluator catches queries that are plainly arithmetic and
# answers them correctly, bypassing retrieval entirely.  Supports +, -,
# *, /, //, %, **, unary ±, parentheses.  Refuses anything else (so this
# is not an `eval()` injection surface).
# ---------------------------------------------------------------------------

_ARITHMETIC_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MATH_PREFIX_RE = re.compile(
    r"^(?:"
    r"what\s*is|what's|whats\s*the|what\s*is\s*the|"
    r"calculate|compute|solve|evaluate|"
    r"tell\s*me|="
    r")\s*",
    re.IGNORECASE,
)

_MATH_ACCEPT_RE = re.compile(r"^[\d+\-*/().\s^x×÷]+$")


_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def _normalise_math(text: str) -> str:
    """Pull arithmetic-looking content out of a natural-language wrapper."""
    cleaned = text.strip().rstrip("?!.").strip()
    cleaned = _MATH_PREFIX_RE.sub("", cleaned, count=1)
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")
    # Common English math words → operator symbols.  Order matters
    # only insofar as multi-word forms ("divided by", "multiplied by")
    # are handled before the single-word forms.
    cleaned = re.sub(r"\bdivided\s+by\b", "/", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmultiplied\s+by\b", "*", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btimes\b", "*", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bplus\b", "+", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bminus\b", "-", cleaned, flags=re.IGNORECASE)
    # 2026-04-26 audit additions — percent / fraction / sqrt forms.
    # "X percent of Y" → "(X/100)*Y"  (also "X% of Y")
    cleaned = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)",
        r"((\1/100)*\2)",
        cleaned,
        flags=re.IGNORECASE,
    )
    # "half of X" / "third of X" / "quarter of X"
    cleaned = re.sub(
        r"\bhalf\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1/2)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:third|thirds)\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1/3)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:quarter|fourth)\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1/4)",
        cleaned,
        flags=re.IGNORECASE,
    )
    # "square root of X" / "sqrt(X)" → X**0.5
    cleaned = re.sub(
        r"\bsquare\s+root\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1**0.5)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bsqrt\s*\(\s*(\d+(?:\.\d+)?)\s*\)",
        r"(\1**0.5)",
        cleaned,
        flags=re.IGNORECASE,
    )
    # "2 x 3" → "2 * 3" (only when flanked by digits so we don't
    # corrupt text like "box").
    cleaned = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", cleaned)
    # Phase 14 (2026-04-25) — word-form exponents.  These must be
    # handled BEFORE the number-word substitution below so a phrase
    # like "nine squared" (we get "9 squared") still hits.  The base
    # number can be either a numeric literal (digits) or a number
    # word — we normalise the number word first via a quick pass,
    # then translate the exponent word.
    # First: number-word → digit substitution for the small range.
    def _word_to_digit(match: re.Match) -> str:
        word = match.group(0).lower()
        return str(_NUMBER_WORDS[word])
    _num_word_re = re.compile(
        r"\b(?:" + "|".join(_NUMBER_WORDS.keys()) + r")\b",
        re.IGNORECASE,
    )
    cleaned = _num_word_re.sub(_word_to_digit, cleaned)
    # "X squared" / "X cubed" / "X to the (power of) Y"
    cleaned = re.sub(
        r"(\d+)\s+squared\b", r"\1**2", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"(\d+)\s+cubed\b", r"\1**3", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"(\d+)\s+to\s+the\s+(?:power\s+(?:of\s+)?)?(\d+)",
        r"\1**\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Collapse the runs of whitespace the word substitutions leave behind.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_math_expr(text: str) -> bool:
    """True if *text* is a short arithmetic expression, maybe wrapped."""
    cleaned = _normalise_math(text)
    if not cleaned or len(cleaned) > 60:
        return False
    # Must contain at least one operator and at least one digit.
    if not re.search(r"[+\-*/^%]", cleaned):
        return False
    if not re.search(r"\d", cleaned):
        return False
    return bool(_MATH_ACCEPT_RE.match(cleaned.replace("**", "^")))


def _eval_math(text: str) -> str | None:
    """Evaluate *text* as arithmetic, return a formatted answer or None."""
    cleaned = _normalise_math(text)
    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def _eval(node: ast.AST) -> float | int | None:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = _ARITHMETIC_OPS.get(type(node.op))
            if op is None:
                return None
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                return None
            try:
                return op(left, right)
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
        if isinstance(node, ast.UnaryOp):
            op = _ARITHMETIC_OPS.get(type(node.op))
            if op is None:
                return None
            val = _eval(node.operand)
            return op(val) if val is not None else None
        return None

    try:
        result = _eval(tree)
    except Exception:  # pragma: no cover - defensive
        return None
    if result is None:
        return None
    # Bound check — refuse anything absurdly large so we don't spend
    # a second computing 2**999_999_999 on behalf of a malicious input.
    try:
        if abs(float(result)) > 1e15:
            return None
    except (OverflowError, TypeError):
        return None
    if isinstance(result, float):
        if result.is_integer():
            result = int(result)
        else:
            result = round(result, 6)
    return f"{result}."


# ---------------------------------------------------------------------------
# Tool route: hostility / abuse detection
# ---------------------------------------------------------------------------
# Phase 1 fix: several hostile prompts ("are you stupid?", "are you
# dumb?") were retrieving the same canned "Noted. I will try to be
# more useful." at wildly inflated scores (1.37, 1.38).  We pattern-
# match the most common hostile framings and answer deterministically
# so the chip shows ``tool: refuse`` rather than a bogus confidence
# number.
# ---------------------------------------------------------------------------

_HOSTILITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:stupid|dumb|idiot|moron|useless|garbage|trash|suck)\b", re.I),
    re.compile(r"\bf+u+c+k\s*(?:you|off)\b", re.I),
    re.compile(r"\bshit(?:ty)?\b", re.I),
    re.compile(r"\byou\s*(?:are|re|r)\s*(?:bad|terrible|awful|pathetic)\b", re.I),
    re.compile(r"\bi\s*hate\s*you\b", re.I),
)

_HOSTILITY_RESPONSES: tuple[str, ...] = (
    "I hear that. I'm a small on-device model — I won't always be right. "
    "Tell me what you actually need and I'll try again.",
    "Fair. I'm deliberately tiny so a lot of things will be outside my scope. "
    "What were you hoping for?",
    "Point taken. What would a better answer have looked like?",
)


def _is_hostility(text: str) -> bool:
    return any(p.search(text) for p in _HOSTILITY_PATTERNS)


def _hostility_reply(text: str) -> str:
    idx = sum(ord(c) for c in text) % len(_HOSTILITY_RESPONSES)
    return _HOSTILITY_RESPONSES[idx]


# ---------------------------------------------------------------------------
# Tool route: curated entity knowledge
# ---------------------------------------------------------------------------
# This tool answers the canonical "where is huawei located?" /
# "who founded apple?" / "what does microsoft sell?" follow-ups that
# the multi-turn co-reference resolver in :mod:`i3.dialogue.coref`
# rewrites pronoun-laden user questions into.
#
# Without this layer, a follow-up like "where are they located?"
# (rewritten to "where is huawei located?" by the resolver) would
# fall back to the generic embedding-NN retrieval over the curated
# Q→A corpus -- which has no exhaustive entity-fact rows -- and
# either return a low-confidence semantic match or fall through to
# OOD.  The tool layer matches the rewritten query against a curated
# {entity → fact} dictionary and returns a deterministic answer with
# a 0.99 confidence stamp, mirroring the math/refuse tool routes.
#
# Mirrors the same set of entities tracked by EntityTracker so an
# "apple" → "where is they?" rewrite always lands.
# ---------------------------------------------------------------------------

_ENTITY_KNOWLEDGE: dict[str, dict[str, str]] = {
    "huawei": {
        "location": "Huawei is headquartered in Shenzhen, China.",
        "founded": "Huawei was founded in 1987 by Ren Zhengfei.",
        "founder": "Huawei was founded by Ren Zhengfei in 1987.",
        "products": "Huawei sells smartphones, telecom networking equipment, cloud services, and HarmonyOS, plus a growing line of consumer wearables.",
        "ceo": "Huawei's CEO is Eric Xu, on a rotating basis with other senior executives.",
        "fact": "Huawei is a global technology company headquartered in Shenzhen.",
    },
    "apple": {
        "location": "Apple is headquartered in Cupertino, California.",
        "founded": "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
        "founder": "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "products": "Apple sells iPhones, Macs, iPads, the Apple Watch, AirPods, and services like iCloud, Apple Music, and the App Store.",
        "ceo": "Apple's CEO is Tim Cook, who took over from Steve Jobs in 2011.",
        "fact": "Apple is a US consumer-electronics and software company headquartered in Cupertino.",
    },
    "microsoft": {
        "location": "Microsoft is headquartered in Redmond, Washington.",
        "founded": "Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
        "founder": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "products": "Microsoft sells Windows, the Office suite, Azure cloud services, Xbox consoles and games, Surface devices, and developer tools like Visual Studio and GitHub.",
        "ceo": "Microsoft's CEO is Satya Nadella, who has led the company since 2014.",
        "fact": "Microsoft is a US software and cloud-services company headquartered in Redmond.",
    },
    "google": {
        "location": "Google is headquartered in Mountain View, California.",
        "founded": "Google was founded in 1998 by Larry Page and Sergey Brin.",
        "founder": "Google was founded by Larry Page and Sergey Brin in 1998.",
        "products": "Google's main products are Search, Android, Chrome, YouTube, Workspace, and the Google Cloud platform, plus Pixel devices.",
        "ceo": "Google's CEO is Sundar Pichai, who has led the company since 2015.",
        "fact": "Google is a US search and advertising company, a subsidiary of Alphabet.",
    },
    "openai": {
        "location": "OpenAI is headquartered in San Francisco, California.",
        "founded": "OpenAI was founded in 2015 by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, and others.",
        "founder": "OpenAI's founders include Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, and others — it was founded in 2015.",
        "products": "OpenAI builds the GPT family of large language models and operates ChatGPT and the OpenAI API.",
        "ceo": "OpenAI's CEO is Sam Altman.",
        "fact": "OpenAI is a US AI research and deployment company headquartered in San Francisco.",
    },
    "anthropic": {
        "location": "Anthropic is headquartered in San Francisco, California.",
        "founded": "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei.",
        "founder": "Anthropic was founded by Dario Amodei and Daniela Amodei in 2021.",
        "products": "Anthropic builds the Claude family of large language models and runs the Anthropic API.",
        "ceo": "Anthropic's CEO is Dario Amodei.",
        "fact": "Anthropic is an AI safety company that builds the Claude family of language models.",
    },
    "meta": {
        "location": "Meta is headquartered in Menlo Park, California.",
        "founded": "Meta — originally Facebook — was founded in 2004 by Mark Zuckerberg with co-founders.",
        "founder": "Meta was founded by Mark Zuckerberg in 2004, alongside fellow Harvard students.",
        "products": "Meta runs Facebook, Instagram, WhatsApp, Threads, and the Quest VR headsets.",
        "ceo": "Meta's CEO is Mark Zuckerberg, who founded the company.",
        "fact": "Meta (formerly Facebook) is a US social-media and VR company headquartered in Menlo Park.",
    },
    "amazon": {
        "location": "Amazon is headquartered in Seattle, Washington.",
        "founded": "Amazon was founded in 1994 by Jeff Bezos.",
        "founder": "Amazon was founded by Jeff Bezos in 1994.",
        "products": "Amazon runs the Amazon marketplace, AWS, the Kindle e-readers, the Alexa voice assistant, and Prime Video.",
        "ceo": "Amazon's CEO is Andy Jassy, who took over from Jeff Bezos in 2021.",
        "fact": "Amazon is a US e-commerce and cloud-services company headquartered in Seattle.",
    },
    "ibm": {
        "location": "IBM is headquartered in Armonk, New York.",
        "founded": "IBM was founded in 1911 as the Computing-Tabulating-Recording Company, renamed to IBM in 1924.",
        "founder": "IBM grew out of the Computing-Tabulating-Recording Company, formed in 1911 by Charles Ranlett Flint.",
        "products": "IBM sells enterprise hardware, software, hybrid cloud services, and consulting, with a long pedigree in computing research.",
        "ceo": "IBM's CEO is Arvind Krishna.",
        "fact": "IBM is a US technology and consulting company headquartered in Armonk, New York.",
    },
    "samsung": {
        "location": "Samsung is headquartered in Suwon, South Korea.",
        "founded": "Samsung was founded in 1938 by Lee Byung-chul.",
        "founder": "Samsung was founded by Lee Byung-chul in 1938.",
        "products": "Samsung makes smartphones, memory chips, displays, home appliances, and a wide range of consumer electronics.",
        "ceo": "Samsung Electronics' CEO is Han Jong-hee.",
        "fact": "Samsung is a South Korean conglomerate headquartered in Suwon.",
    },
    "tesla": {
        "location": "Tesla is headquartered in Austin, Texas.",
        "founded": "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning, with Elon Musk joining shortly after as an investor and chair.",
        "founder": "Tesla was founded by Martin Eberhard and Marc Tarpenning in 2003, with Elon Musk joining as an early investor and chair.",
        "products": "Tesla makes electric vehicles, energy storage products, and solar systems.",
        "ceo": "Tesla's CEO is Elon Musk.",
        "fact": "Tesla is a US electric-vehicle and energy company headquartered in Austin.",
    },
    "nvidia": {
        "location": "Nvidia is headquartered in Santa Clara, California.",
        "founded": "Nvidia was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem.",
        "founder": "Nvidia was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993.",
        "products": "Nvidia designs GPUs, AI accelerators, networking gear, and the CUDA software platform that's the de-facto standard for ML compute.",
        "ceo": "Nvidia's CEO is Jensen Huang, who co-founded the company.",
        "fact": "Nvidia is a US semiconductor company headquartered in Santa Clara.",
    },
    "intel": {
        "location": "Intel is headquartered in Santa Clara, California.",
        "founded": "Intel was founded in 1968 by Gordon Moore and Robert Noyce.",
        "founder": "Intel was founded by Gordon Moore and Robert Noyce in 1968.",
        "products": "Intel designs and manufactures x86 CPUs, networking chips, and provides foundry services.",
        "ceo": "Intel's CEO is Pat Gelsinger.",
        "fact": "Intel is a US semiconductor company headquartered in Santa Clara.",
    },
    "sony": {
        "location": "Sony is headquartered in Tokyo, Japan.",
        "founded": "Sony was founded in 1946 by Masaru Ibuka and Akio Morita.",
        "founder": "Sony was founded by Masaru Ibuka and Akio Morita in 1946.",
        "products": "Sony makes PlayStation consoles and games, image sensors, cameras, music, and films.",
        "ceo": "Sony's CEO is Kenichiro Yoshida.",
        "fact": "Sony is a Japanese electronics, gaming and entertainment conglomerate headquartered in Tokyo.",
    },
}


# ---------------------------------------------------------------------------
# Country attribute facts (Phase 14, 2026-04-25).
# ---------------------------------------------------------------------------
# Powers bare-noun follow-ups like ``"language?"`` or ``"capital?"``
# right after a ``"tell me about <country>"`` topic prompt.  The
# coref-aware bare-noun rewriter in :mod:`i3.dialogue.coref` /
# ``i3.pipeline.engine`` rewrites the bare follow-up into a full
# question (``"what is the language of japan?"``); patterns below
# match the rewritten shape and return the curated fact.
#
# 12 countries × 6 attributes = 72 entries.  Population figures are
# rounded to the nearest million for stability across years; the
# objective is "useful answer" not "current to the day".
# ---------------------------------------------------------------------------

_COUNTRY_KNOWLEDGE: dict[str, dict[str, str]] = {
    "japan": {
        "language": "Japan's official language is Japanese (日本語).",
        "currency": "Japan's currency is the Yen (¥).",
        "capital": "The capital of Japan is Tokyo.",
        "population": "Japan's population is around 125 million.",
        "flag": "Japan's flag is the Hinomaru — a red disc on a white field.",
        "government": "Japan is a constitutional monarchy with a parliamentary system.",
    },
    "germany": {
        "language": "Germany's official language is German.",
        "currency": "Germany's currency is the Euro (€).",
        "capital": "The capital of Germany is Berlin.",
        "population": "Germany's population is around 83 million.",
        "flag": "Germany's flag has three horizontal bands of black, red, and gold.",
        "government": "Germany is a federal parliamentary republic.",
    },
    "france": {
        "language": "France's official language is French.",
        "currency": "France's currency is the Euro (€).",
        "capital": "The capital of France is Paris.",
        "population": "France's population is around 68 million.",
        "flag": "France's flag is the tricolour: vertical bands of blue, white, and red.",
        "government": "France is a semi-presidential republic.",
    },
    "china": {
        "language": "China's official language is Mandarin Chinese (Putonghua).",
        "currency": "China's currency is the Renminbi / Yuan (¥, CNY).",
        "capital": "The capital of China is Beijing.",
        "population": "China's population is around 1.41 billion.",
        "flag": "China's flag is red with five gold stars in the upper-left canton.",
        "government": "China is a one-party socialist republic led by the Communist Party of China.",
    },
    "uk": {
        "language": "The UK's predominant language is English.",
        "currency": "The UK's currency is the Pound Sterling (£).",
        "capital": "The capital of the United Kingdom is London.",
        "population": "The UK's population is around 67 million.",
        "flag": "The UK's flag is the Union Jack — a combination of the red, white, and blue crosses of England, Scotland, and Ireland.",
        "government": "The UK is a constitutional monarchy with a parliamentary system.",
    },
    "usa": {
        "language": "The USA's predominant language is English; there is no de jure official language.",
        "currency": "The USA's currency is the US Dollar ($).",
        "capital": "The capital of the United States is Washington, D.C.",
        "population": "The USA's population is around 332 million.",
        "flag": "The US flag (Stars and Stripes) has 13 red-and-white stripes and 50 white stars on a blue canton.",
        "government": "The USA is a federal presidential constitutional republic.",
    },
    "spain": {
        "language": "Spain's official language is Spanish (Castilian).",
        "currency": "Spain's currency is the Euro (€).",
        "capital": "The capital of Spain is Madrid.",
        "population": "Spain's population is around 47 million.",
        "flag": "Spain's flag has horizontal bands of red, yellow, and red, with the national coat of arms.",
        "government": "Spain is a parliamentary constitutional monarchy.",
    },
    "italy": {
        "language": "Italy's official language is Italian.",
        "currency": "Italy's currency is the Euro (€).",
        "capital": "The capital of Italy is Rome.",
        "population": "Italy's population is around 59 million.",
        "flag": "Italy's flag is a tricolour: vertical bands of green, white, and red.",
        "government": "Italy is a parliamentary republic.",
    },
    "russia": {
        "language": "Russia's official language is Russian.",
        "currency": "Russia's currency is the Russian Rouble (₽).",
        "capital": "The capital of Russia is Moscow.",
        "population": "Russia's population is around 144 million.",
        "flag": "Russia's flag has three horizontal bands of white, blue, and red.",
        "government": "Russia is a federal semi-presidential republic.",
    },
    "brazil": {
        "language": "Brazil's official language is Portuguese.",
        "currency": "Brazil's currency is the Brazilian Real (R$).",
        "capital": "The capital of Brazil is Brasília.",
        "population": "Brazil's population is around 215 million.",
        "flag": "Brazil's flag is a green field with a yellow rhombus, a blue celestial globe, and a banner reading 'Ordem e Progresso'.",
        "government": "Brazil is a federal presidential constitutional republic.",
    },
    "india": {
        "language": "India's official languages are Hindi and English; the constitution recognises 22 scheduled languages.",
        "currency": "India's currency is the Indian Rupee (₹).",
        "capital": "The capital of India is New Delhi.",
        "population": "India's population is around 1.42 billion.",
        "flag": "India's flag is a tricolour: horizontal bands of saffron, white, and green, with the Ashoka Chakra (a 24-spoke wheel) in navy blue at the centre.",
        "government": "India is a federal parliamentary constitutional republic.",
    },
    "australia": {
        "language": "Australia's predominant language is English.",
        "currency": "Australia's currency is the Australian Dollar (A$).",
        "capital": "The capital of Australia is Canberra.",
        "population": "Australia's population is around 26 million.",
        "flag": "Australia's flag has the Union Jack in the canton, the Commonwealth Star, and the Southern Cross constellation.",
        "government": "Australia is a federal parliamentary constitutional monarchy.",
    },
    "canada": {
        "language": "Canada's official languages are English and French.",
        "currency": "Canada's currency is the Canadian Dollar (C$).",
        "capital": "The capital of Canada is Ottawa.",
        "population": "Canada's population is around 40 million.",
        "flag": "Canada's flag has a red maple leaf on a white field flanked by red bars.",
        "government": "Canada is a federal parliamentary constitutional monarchy.",
    },
}


_COUNTRY_ALIASES: dict[str, str] = {
    "japan": "japan",
    "germany": "germany", "deutschland": "germany",
    "france": "france",
    "china": "china",
    "uk": "uk", "u.k.": "uk",
    "united kingdom": "uk", "britain": "uk", "great britain": "uk",
    "usa": "usa", "u.s.a.": "usa", "us": "usa", "u.s.": "usa",
    "united states": "usa", "america": "usa",
    "spain": "spain",
    "italy": "italy",
    "russia": "russia",
    "brazil": "brazil",
    "india": "india",
    "australia": "australia",
    "canada": "canada",
}


# Country-attribute query patterns.  Each maps a question shape to a
# slot in :data:`_COUNTRY_KNOWLEDGE`.  The country name is captured by
# the indicated group.
_COUNTRY_QUERY_PATTERNS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (
        re.compile(
            r"^what\s+is\s+(?:the\s+)?(language|currency|capital|population|flag|government)\s+of\s+(.+?)\s*\??\s*$",
            re.I,
        ),
        "_DYNAMIC_SLOT_GROUP1",
        2,
    ),
    (
        re.compile(
            r"^what'?s\s+(?:the\s+)?(language|currency|capital|population|flag|government)\s+of\s+(.+?)\s*\??\s*$",
            re.I,
        ),
        "_DYNAMIC_SLOT_GROUP1",
        2,
    ),
    (
        re.compile(
            r"^what\s+(language|currency|capital|government)\s+(?:do\s+they|does\s+(?:it|they))\s+"
            r"(?:speak|use|have)\s+in\s+(.+?)\s*\??\s*$",
            re.I,
        ),
        "_DYNAMIC_SLOT_GROUP1",
        2,
    ),
    (
        re.compile(
            r"^what\s+is\s+the\s+(?:capital\s+city|main\s+language|official\s+language)\s+of\s+(.+?)\s*\??\s*$",
            re.I,
        ),
        # Map these to "capital" / "language" by inspecting the
        # original phrase below.
        "_DYNAMIC_FROM_PHRASE",
        1,
    ),
)


def _country_lookup(text: str) -> tuple[str | None, str | None]:
    """Return ``(canonical_country, fact)`` for a country-attribute query.

    Behaviour mirrors :func:`_entity_lookup`: try the curated
    :data:`_COUNTRY_KNOWLEDGE` table; return ``(None, None)`` on miss
    so the caller falls through to retrieval / SLM.
    """
    if not text:
        return None, None
    cleaned = re.sub(r"\s+", " ", text.strip())
    for pattern, slot, group_idx in _COUNTRY_QUERY_PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        try:
            country_surface = m.group(group_idx)
        except IndexError:
            continue
        if not country_surface:
            continue
        country_lower = re.sub(r"^(?:the)\s+", "", country_surface.strip().lower())
        country_lower = country_lower.rstrip(".?!,")
        canonical = _COUNTRY_ALIASES.get(country_lower)
        if canonical is None:
            continue
        # Resolve the slot.  For the first 3 patterns the slot name is
        # captured in group 1; for the "main language / capital city"
        # phrasing we infer it from the phrase itself.
        if slot == "_DYNAMIC_SLOT_GROUP1":
            try:
                slot_name = m.group(1).lower()
            except IndexError:
                continue
        elif slot == "_DYNAMIC_FROM_PHRASE":
            phrase = cleaned.lower()
            if "capital" in phrase:
                slot_name = "capital"
            elif "language" in phrase:
                slot_name = "language"
            else:
                continue
        else:
            slot_name = slot
        fact = _COUNTRY_KNOWLEDGE.get(canonical, {}).get(slot_name)
        if fact:
            return canonical, fact
    return None, None


# Catalogue of entity surface forms → canonical key in
# _ENTITY_KNOWLEDGE.  Mirrors the alias coverage in
# ``i3.dialogue.coref`` so the entity tool answers whatever surface
# form the resolver passes through.
_ENTITY_TOOL_ALIASES: dict[str, str] = {
    "huawei": "huawei",
    "apple": "apple",
    "microsoft": "microsoft", "msft": "microsoft",
    "google": "google", "alphabet": "google",
    "openai": "openai", "open ai": "openai",
    "anthropic": "anthropic",
    "meta": "meta", "facebook": "meta", "fb": "meta",
    "amazon": "amazon", "aws": "amazon",
    "ibm": "ibm",
    "samsung": "samsung",
    "tesla": "tesla",
    "nvidia": "nvidia",
    "intel": "intel",
    "sony": "sony",
}

# Each pattern maps a question shape to the dict slot in
# _ENTITY_KNOWLEDGE.  Group 2 captures the entity surface form for
# patterns that wrap the entity inside the sentence; group 1 captures
# it for patterns where the entity comes before the verb.
_ENTITY_QUERY_PATTERNS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    # "where is huawei located/based/headquartered" / "where are X located"
    (
        re.compile(
            r"^where\s+(?:is|are)\s+(.+?)\s+"
            r"(?:located|based|headquartered)\s*\??\s*$",
            re.I,
        ),
        "location",
        1,
    ),
    # Bare "where is huawei?"
    (re.compile(r"^where\s+(?:is|are)\s+(.+?)\s*\??\s*$", re.I), "location", 1),
    # "who founded huawei?" / "who started apple?"
    (
        re.compile(
            r"^who\s+(?:founded|started|created)\s+(.+?)\s*\??\s*$", re.I
        ),
        "founder",
        1,
    ),
    # "when was huawei founded?" / "when was apple started?"
    (
        re.compile(
            r"^when\s+was\s+(.+?)\s+(?:founded|started|created|established)"
            r"\s*\??\s*$",
            re.I,
        ),
        "founded",
        1,
    ),
    # "what do(es) X sell/make/build/do/produce?"
    (
        re.compile(
            r"^what\s+(?:do|does)\s+(.+?)\s+"
            r"(?:sell|make|do|build|produce|offer)\s*\??\s*$",
            re.I,
        ),
        "products",
        1,
    ),
    # "what (kind of) products does X sell?"
    (
        re.compile(
            r"^what\s+(?:kind\s+of\s+)?products\s+(?:do|does)\s+(.+?)\s+"
            r"(?:sell|make|build|produce|offer)\s*\??\s*$",
            re.I,
        ),
        "products",
        1,
    ),
    # "who runs/leads huawei?" / "who is the ceo of apple?"
    (
        re.compile(
            r"^who\s+(?:runs|leads|heads)\s+(.+?)\s*\??\s*$", re.I
        ),
        "ceo",
        1,
    ),
    (
        re.compile(
            r"^who\s+is\s+the\s+ceo\s+of\s+(.+?)\s*\??\s*$", re.I
        ),
        "ceo",
        1,
    ),
    # 2026-04-26 iter 11: support coref-rewritten "who is their ceo"
    # → "who is microsoft ceo" / "who is microsoft's ceo".
    (
        re.compile(
            r"^who\s+is\s+(.+?)(?:'s)?\s+ceo\s*\??\s*$", re.I
        ),
        "ceo",
        1,
    ),
    # "who's the ceo of X" / "what is X's ceo's name"
    (
        re.compile(
            r"^who(?:'s|\s+is)\s+(?:the\s+)?ceo\s+(?:of|at)\s+(.+?)\s*\??\s*$",
            re.I,
        ),
        "ceo",
        1,
    ),
    # Founder follow-ups: "who founded them" → "who founded microsoft"
    # already matches the founder pattern above; add coref-friendly
    # variants for "who started", "who created", "who built X".
    (
        re.compile(
            r"^who\s+(?:built|made)\s+(.+?)\s*\??\s*$", re.I
        ),
        "founder",
        1,
    ),
    # "what year was X founded" / "when did X start"
    (
        re.compile(
            r"^when\s+did\s+(.+?)\s+(?:start|begin|launch)\s*\??\s*$", re.I
        ),
        "founded",
        1,
    ),
    (
        re.compile(
            r"^what\s+year\s+was\s+(.+?)\s+(?:founded|started|created|established)\s*\??\s*$",
            re.I,
        ),
        "founded",
        1,
    ),
    # "what does X make" → already covered; add "what is X known for"
    (
        re.compile(
            r"^what\s+is\s+(.+?)\s+(?:known|famous)\s+for\s*\??\s*$", re.I
        ),
        "products",
        1,
    ),
    # "where is X HQ" / "where is X's HQ" / "where is X based"
    (
        re.compile(
            r"^where\s+is\s+(.+?)(?:'s)?\s+(?:hq|headquarters|head\s+office)\s*\??\s*$",
            re.I,
        ),
        "location",
        1,
    ),
    # Iter 40 (2026-04-26): bare-canonical-slot patterns — when coref
    # rewrites "their CEO" → "Microsoft CEO" / "their founder" →
    # "Microsoft founder" / "its HQ" → "Microsoft HQ", we want those to
    # match the entity-tool lookup directly rather than falling through
    # to embedding retrieval.  The pattern accepts an optional "'s"
    # possessive ("Microsoft's CEO") and matches a small whitelist of
    # slot keywords so we don't over-trigger on free-form text.
    (
        re.compile(
            r"^(.+?)(?:'s)?\s+ceo\s*\??\s*$", re.I,
        ),
        "ceo",
        1,
    ),
    (
        re.compile(
            r"^(.+?)(?:'s)?\s+founder(?:s)?\s*\??\s*$", re.I,
        ),
        "founder",
        1,
    ),
    (
        re.compile(
            r"^(.+?)(?:'s)?\s+(?:hq|headquarters|head\s+office|location)\s*\??\s*$",
            re.I,
        ),
        "location",
        1,
    ),
    (
        re.compile(
            r"^(.+?)(?:'s)?\s+products?\s*\??\s*$", re.I,
        ),
        "products",
        1,
    ),
)


def _entity_lookup(text: str) -> tuple[str | None, str | None]:
    """Return ``(canonical_entity, fact)`` for an entity-fact query.

    Behaviour:
        - First consults the curated knowledge graph
          (:mod:`i3.dialogue.knowledge_graph`) which has richer slots
          (``founded_by``, ``competitor_of``, ``famous_for`` etc.) and a
          phrase resolver that handles a wider set of surface forms.
        - Falls back to the legacy :data:`_ENTITY_QUERY_PATTERNS` /
          :data:`_ENTITY_KNOWLEDGE` flat dict when the KG returns no
          match — single-turn behaviour is preserved.
        - Returns ``(None, None)`` and lets the caller fall through to
          embedding retrieval / SLM generation when neither matches.

    The check is run on the (potentially co-reference-resolved) user
    query so a follow-up "where are they located?" gets rewritten to
    "where is huawei located?" upstream and then matches here.
    """
    if not text:
        return None, None
    cleaned = text.strip()
    # Strip trailing punctuation & collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned)

    # ── 1. Knowledge-graph fast path ──────────────────────────────
    # The KG handles a wider set of question shapes (competitor_of,
    # famous_for, owns/acquired/wrote/discovered/...) and the
    # composed-answer helper produces nicely-formed sentences from
    # multi-object triples.  Wrapped in try/except so a KG failure
    # never blocks the flat-dict fallback below.
    try:
        from i3.dialogue.knowledge_graph import get_global_kg
        kg = get_global_kg()
        if kg.loaded:
            resolved = kg.resolve_phrase(cleaned)
            if resolved is not None:
                subj, pred = resolved
                answer = kg.compose_answer(subj, pred)
                if answer:
                    return subj, answer
    except Exception:  # pragma: no cover — fall back to legacy path
        logger.debug("KG entity lookup failed; falling back to flat dict.")

    # ── 2. Legacy flat-dict fallback ──────────────────────────────
    for pattern, slot, group_idx in _ENTITY_QUERY_PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        try:
            surface = m.group(group_idx)
        except IndexError:
            continue
        if not surface:
            continue
        surface_lower = surface.strip().lower()
        # Strip surrounding articles ("the apple" → "apple") so a
        # naive resolver rewrite doesn't break alias lookup.
        surface_lower = re.sub(r"^(?:the|a|an)\s+", "", surface_lower)
        canonical = _ENTITY_TOOL_ALIASES.get(surface_lower)
        if canonical is None:
            continue
        fact = _ENTITY_KNOWLEDGE.get(canonical, {}).get(slot)
        if fact:
            return canonical, fact
    return None, None


def _graph_compose_lookup(
    text: str, hint_entity: str | None = None
) -> tuple[str | None, str | None]:
    """Try to compose a multi-triple answer from the knowledge graph.

    Used for open-ended questions that request more than a single fact:
        - "what does microsoft own?"
        - "who are apple's competitors?"
        - "tell me about apple"

    Returns ``(canonical_entity, composed_paragraph)`` or ``(None, None)``.

    This is consulted *after* :func:`_entity_lookup` (the single-fact
    tool) — single-fact queries land on the entity tool first because
    the chip is more specific.
    """
    if not text:
        return None, None
    cleaned = re.sub(r"\s+", " ", text.strip()).rstrip(".!?,")
    try:
        from i3.dialogue.knowledge_graph import get_global_kg
        kg = get_global_kg()
        if not kg.loaded:
            return None, None
    except Exception:
        return None, None

    # 1. "tell me about X" / "what is X" / "describe X" — overview path.
    overview_patterns = (
        re.compile(r"^tell\s+me\s+about\s+(.+)$", re.I),
        re.compile(r"^describe\s+(.+)$", re.I),
        re.compile(r"^what\s+is\s+(.+)$", re.I),
        re.compile(r"^what\s+are\s+(.+)$", re.I),
        re.compile(r"^who\s+is\s+(.+)$", re.I),
    )
    for pat in overview_patterns:
        m = pat.match(cleaned)
        if not m:
            continue
        surface = m.group(1).strip().lower().rstrip("?.!")
        canonical = kg._canonical(surface)  # noqa: SLF001 — module helper
        if canonical:
            ans = kg.overview(canonical)
            if ans:
                return canonical, ans

    # 2. Compositional questions resolved via resolve_phrase + multi-object
    #    predicates.
    resolved = kg.resolve_phrase(cleaned, hint_entity=hint_entity)
    if resolved is None:
        return None, None
    subj, pred = resolved
    if pred in {
        "competitor_of", "acquired", "owns", "founded_by",
        "discovered", "discovered_by", "won_by", "fell_in", "ended_in",
    }:
        # Check both the requested predicate AND any aliased one (e.g.
        # "owns" → "acquired") so a question that lands on a related
        # slot still composes when the catalogue stores facts under a
        # different name.
        ans = kg.compose_answer(subj, pred)
        if not ans:
            return None, None
        # We only want this tool to fire when the answer is *worth*
        # composing — i.e. multi-fact for the multi-object predicates.
        if pred in {"competitor_of", "acquired", "owns", "discovered"}:
            # check there are at least 2 relations under the
            # requested or aliased predicate
            count = max(
                len(kg.get_related(subj, pred)),
                len(kg.get_related(subj, "acquired")) if pred in ("owns", "acquired") else 0,
                len(kg.get_related(subj, "competitor_of")) if pred == "competitor_of" else 0,
            )
            if count < 2:
                return None, None
        return subj, ans
    return None, None


# ---------------------------------------------------------------------------
# Comparison tool — "which one is bigger?", "X vs Y", "compare A and B"
# ---------------------------------------------------------------------------
# Drives the *compare* tool route added in the 2026-04-25 chat-quality
# overhaul.  The patterns try to detect a comparison question shape; when
# the user names two entities directly (``"python vs rust"``) we look them
# up in :data:`_COMPARISON_FACTS`.  When the user types a *short*
# comparison query without entities (``"which one is bigger?"``) we have
# no entities to compare and return ``(None, None, "")`` — the engine
# will inject the two most-recent entities from the EntityTracker
# recency stack as the comparison pair before re-calling.
#
# The fact dict is intentionally bounded (~15 curated pairs).  Pairs not
# in the table fall through to retrieval / SLM generation; we never
# fabricate facts.

_COMPARISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "which one is bigger/larger/etc"
    re.compile(
        r"^which\s+(?:one\s+)?is\s+"
        r"(?:bigger|larger|better|smaller|older|newer|faster|slower|"
        r"more\s+popular|more\s+expensive|cheaper)\s*\??\s*$",
        re.I,
    ),
    # "is X bigger/better/... than Y"
    re.compile(
        r"^(?:is|are)\s+(.+?)\s+"
        r"(?:bigger|larger|better|smaller|older|newer|faster|slower|"
        r"more\s+popular|more\s+expensive|cheaper)\s+than\s+(.+?)\s*\??\s*$",
        re.I,
    ),
    # "X vs Y" / "X versus Y"
    re.compile(r"^(.+?)\s+(?:vs|versus|v\.)\s+(.+?)\s*\??\s*$", re.I),
    # "compare X and/to/with Y"
    re.compile(
        r"^compare\s+(.+?)\s+(?:and|to|with)\s+(.+?)\s*\??\s*$",
        re.I,
    ),
    # "what's the difference between X and Y"
    re.compile(
        r"^(?:what'?s|what\s+is)\s+the\s+difference\s+between\s+(.+?)\s+and\s+(.+?)\s*\??\s*$",
        re.I,
    ),
    # "X or Y" — only with no question word, otherwise too noisy
    re.compile(r"^(.+?)\s+or\s+(.+?)\s*\??\s*$", re.I),
    # Iter 35 (2026-04-26): zero-arg "compare them / these / those /
    # the two / the latest" — uses fallback_pair from EntityTracker.
    re.compile(
        r"^(?:compare\s+(?:them|these|those|the\s+two|the\s+last\s+two|the\s+latest|both)|"
        r"how\s+do\s+(?:they|these|those)\s+compare|"
        r"which\s+(?:one\s+)?(?:is\s+)?(?:better|bigger|larger|smaller|faster|slower|cheaper|more\s+popular)\s*\??)\s*\??\s*$",
        re.I,
    ),
)

# Aspect detector — maps comparison adjectives in the user query to the
# slot we look up in :data:`_COMPARISON_FACTS`.  Falls back to "default"
# when no aspect is detected.
_COMPARISON_ASPECTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:bigger|larger|biggest|largest|size|bigger\?)\b", re.I), "size"),
    (re.compile(r"\b(?:smaller|smallest)\b", re.I), "size"),
    (re.compile(r"\b(?:older|oldest|earlier|first)\b", re.I), "age"),
    (re.compile(r"\b(?:newer|newest|recent|later)\b", re.I), "age"),
    (re.compile(r"\b(?:faster|fastest|speed|performance)\b", re.I), "speed"),
    (re.compile(r"\b(?:slower|slowest)\b", re.I), "speed"),
    (re.compile(r"\b(?:more\s+popular|popular|popularity|widely\s+used)\b", re.I), "popular"),
    (re.compile(r"\b(?:cheaper|cheapest|cheap|expensive|cost)\b", re.I), "cost"),
    (re.compile(r"\b(?:better|best|good)\b", re.I), "default"),
    (re.compile(r"\b(?:harder|hardest|easier|easiest)\b", re.I), "default"),
)


_COMPARISON_ALIASES: dict[str, str] = {
    # Map free-form comparison entity surface forms to a canonical key.
    # Only the entities the comparison fact table covers need to appear.
    "huawei": "huawei",
    "apple": "apple",
    "microsoft": "microsoft", "msft": "microsoft",
    "google": "google", "alphabet": "google",
    "openai": "openai", "open ai": "openai",
    "anthropic": "anthropic",
    "meta": "meta", "facebook": "meta",
    "amazon": "amazon",
    "samsung": "samsung",
    "tesla": "tesla",
    "nvidia": "nvidia",
    "intel": "intel",
    "amd": "amd",
    "python": "python",
    "rust": "rust",
    "javascript": "javascript", "js": "javascript",
    "java": "java",
    "c++": "c++", "cpp": "c++",
    "go": "go", "golang": "go",
    "linux": "linux",
    "windows": "windows",
    "macos": "macos", "mac os": "macos", "mac": "macos",
    "ios": "ios",
    "android": "android",
    "chatgpt": "chatgpt", "chat gpt": "chatgpt",
    "claude": "claude",
    "gemini": "gemini",
    "gpt": "gpt",
    "bert": "bert",
}


_COMPARISON_FACTS: dict[tuple[str, str], dict[str, str]] = {
    # Tech giants ---------------------------------------------------------
    ("apple", "huawei"): {
        "size": "Apple has a larger market capitalisation (around $3 trillion) than Huawei (private, estimated around $300 billion). Huawei employs more people globally — roughly 200,000 versus Apple's 160,000.",
        "popular": "Apple has broader global brand recognition; Huawei dominates in China and is growing across emerging markets, especially in 5G infrastructure.",
        "default": "Apple is a US consumer-electronics company famous for the iPhone and the Mac, while Huawei is a Chinese telecom-and-devices company famous for HarmonyOS, the Mate phones, and 5G networking. Both compete in smartphones; Huawei also leads in carrier-grade 5G equipment.",
    },
    ("apple", "microsoft"): {
        "size": "Apple and Microsoft are both around $3 trillion in market capitalisation and trade places at the top of the rankings depending on the day. Microsoft has more employees (around 220,000 vs Apple's 160,000).",
        "default": "Apple builds consumer devices — iPhone, Mac, iPad — wrapped in tightly integrated software. Microsoft builds productivity software (Windows, Office), the Azure cloud, gaming (Xbox), and developer tools (Visual Studio, GitHub). Both are top-tier US tech giants.",
    },
    ("apple", "samsung"): {
        "size": "Samsung is a much larger conglomerate by employees (around 270,000) and a wider product catalogue, but Apple has a substantially higher market capitalisation and profit margin per device.",
        "default": "Apple builds the iPhone, Mac, and a tightly integrated software ecosystem; Samsung builds Galaxy phones plus a much wider range of displays, memory chips, TVs, and home appliances. Phone choice is usually iOS vs Android plus integration preferences.",
    },
    ("google", "microsoft"): {
        "default": "Google's core business is online advertising tied to Search, plus Android, Chrome, YouTube, Workspace, and Google Cloud. Microsoft's core is enterprise software (Windows, Office), Azure cloud, and gaming. Both run frontier AI labs (DeepMind / Google AI vs Microsoft Research, plus the OpenAI partnership).",
    },
    ("amazon", "google"): {
        "default": "Amazon's core is e-commerce plus AWS, the largest public cloud provider by revenue. Google's core is search advertising plus Android and Google Cloud. They overlap in cloud, smart speakers, and video.",
    },
    # AI labs / chat assistants ------------------------------------------
    ("chatgpt", "claude"): {
        "default": "ChatGPT (OpenAI) and Claude (Anthropic) are both top-tier chat assistants from frontier labs. They have slightly different defaults — Claude tends to be more cautious and explicit about uncertainty, ChatGPT has a broader plug-in and tooling ecosystem — and capability is roughly comparable for most everyday tasks.",
        "popular": "ChatGPT has the larger user base and broader product surface (custom GPTs, plug-ins, voice). Claude is widely used inside enterprise via Anthropic's API and Amazon Bedrock.",
    },
    ("openai", "anthropic"): {
        "default": "OpenAI and Anthropic both build frontier large language models — OpenAI's GPT series and Anthropic's Claude series. They differ in safety methodology and product surface, but both are credible choices to build on top of.",
        "size": "OpenAI is the larger company by headcount and revenue. Anthropic is smaller and focused more narrowly on AI safety research.",
    },
    ("gpt", "bert"): {
        "default": "GPT and BERT are both transformer-based language models from a similar era, but they differ in shape and purpose. GPT is decoder-only and trained for text generation; BERT is encoder-only and trained on masked-token prediction for classification and understanding tasks. GPT-style models dominate today's chat assistants.",
        "age": "BERT was released by Google in 2018; GPT-1 also came out in 2018 (OpenAI), and the GPT series has continued through GPT-2, 3, 4, and beyond, while BERT's evolution slowed after RoBERTa and DeBERTa.",
    },
    # Programming languages ----------------------------------------------
    ("python", "rust"): {
        "default": "Python is a high-level interpreted language designed for readability and rapid prototyping. Rust is a systems language with manual memory management and zero-cost abstractions, designed for safety and performance. Python is easier to learn; Rust is faster and safer at runtime.",
        "speed": "Rust is dramatically faster than Python for CPU-bound work — often 10-100x — because it compiles to native code without a global interpreter lock or runtime garbage collection.",
        "default_": "",
    },
    ("python", "javascript"): {
        "default": "Python and JavaScript both shine, in different places. Python is the standard for data science, machine learning, scripting, and back-end work; JavaScript dominates the browser and is strong on the back-end via Node.js. Pick by where the code needs to run.",
        "popular": "JavaScript has slightly more developers worldwide thanks to web ubiquity, while Python leads in scientific computing and ML.",
    },
    ("python", "java"): {
        "default": "Python is dynamically typed, concise, and dominant in data science and scripting. Java is statically typed, runs on the JVM, and remains a heavyweight in enterprise back-ends and Android development. Python is faster to prototype in; Java is faster at runtime and easier to deploy at scale.",
    },
    ("java", "javascript"): {
        "default": "Despite the similar name they're unrelated. Java is a statically-typed, compiled language for the JVM, used heavily in enterprise back-ends and Android. JavaScript is a dynamically-typed scripting language for browsers and Node.js servers.",
    },
    ("rust", "c++"): {
        "default": "Rust gives you C++-class performance with memory-safety guarantees enforced at compile time, at the cost of a steeper initial learning curve. C++ has a larger ecosystem and decades of legacy code. For new systems work, Rust is increasingly the default.",
    },
    # Operating systems ---------------------------------------------------
    ("linux", "windows"): {
        "default": "Linux is an open-source family of operating systems built on the Linux kernel — free, scriptable, and dominant on servers and embedded devices. Windows is Microsoft's proprietary OS, dominant on consumer and enterprise desktops and the standard for PC gaming.",
        "popular": "Windows is more popular on desktops; Linux dominates servers, supercomputers, and embedded devices, and underlies Android.",
    },
    ("macos", "windows"): {
        "default": "Windows runs on the widest range of hardware, has the broadest software library, and is the standard for PC gaming. macOS is a polished UNIX-based desktop OS with tight integration with Apple hardware. Choice usually comes down to your software needs and hardware preferences.",
    },
    ("ios", "android"): {
        "default": "iOS is more uniform across devices, with strong privacy defaults and longer software-update windows per phone. Android offers far more hardware choice, deeper customisation, and tighter integration with Google services. Either is a defensible choice.",
        "popular": "Android has roughly 70% of the global smartphone market share by units; iOS has the higher share in the US, the UK, and Japan and a much larger share of revenue per user.",
    },
    # Hardware -----------------------------------------------------------
    ("intel", "amd"): {
        "default": "Intel and AMD make x86 CPUs that compete on price and performance per watt. AMD has been strong on multi-core performance per dollar in recent years; Intel still leads in some single-thread workloads. Pick by your specific workload and budget.",
    },
    ("nvidia", "amd"): {
        "default": "Nvidia leads the GPU market for machine learning thanks to CUDA and a deep software stack. AMD competes well on raw FLOPs per dollar in gaming and is gaining ground in ML through ROCm, but software maturity still lags Nvidia.",
    },
}


def _normalise_compare_entity(name: str) -> str | None:
    """Lower-case, trim, strip articles, then map to a canonical
    comparison key via :data:`_COMPARISON_ALIASES`.  Returns ``None``
    when the surface form isn't a curated comparison entity."""
    if not name:
        return None
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    cleaned = cleaned.rstrip("?.!,").strip()
    if not cleaned:
        return None
    return _COMPARISON_ALIASES.get(cleaned)


def _detect_comparison_aspect(text: str) -> str:
    """Return the comparison slot key (``"size"``, ``"popular"``, ...)
    matched by the question text, or ``"default"`` when no aspect
    keyword is detected."""
    if not text:
        return "default"
    for pattern, slot in _COMPARISON_ASPECTS:
        if pattern.search(text):
            return slot
    return "default"


def _compare_lookup(
    text: str,
    *,
    fallback_pair: tuple[str, str] | None = None,
) -> tuple[tuple[str, str] | None, str]:
    """Detect a comparison question; return ``((a, b), fact)`` or ``(None, "")``.

    Args:
        text: The user query (potentially co-reference-resolved).
        fallback_pair: When the user typed a comparison shape but
            named no entities (e.g. ``"which one is bigger?"``), the
            engine can pass two recent entities from the EntityTracker
            recency stack so the lookup uses them.  When ``None``, an
            entity-less comparison shape returns ``(None, "")``.

    Returns:
        ``((canonical_a, canonical_b), fact_string)`` when a curated
        comparison pair was found, else ``(None, "")``.  We never
        fabricate facts: pairs not in :data:`_COMPARISON_FACTS` fall
        through to retrieval.
    """
    if not text:
        return None, ""
    cleaned = re.sub(r"\s+", " ", text.strip())
    pair: tuple[str, str] | None = None
    for pattern in _COMPARISON_PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        groups = m.groups()
        if len(groups) >= 2:
            a = _normalise_compare_entity(groups[0])
            b = _normalise_compare_entity(groups[1])
            if a and b and a != b:
                pair = (a, b)
                break
            # The comparison shape matched but one or both surface
            # forms don't alias to a curated entity → fall through
            # to retrieval rather than guess.
            return None, ""
        else:
            # Zero-arg pattern ("which one is bigger?") — fall through
            # to fallback_pair handling below.
            if fallback_pair is not None:
                a = _normalise_compare_entity(fallback_pair[0])
                b = _normalise_compare_entity(fallback_pair[1])
                if a and b and a != b:
                    pair = (a, b)
                    break
            return None, ""
    if pair is None:
        return None, ""
    aspect = _detect_comparison_aspect(cleaned)
    # Look up in either pair order.
    a, b = pair
    facts = _COMPARISON_FACTS.get((a, b)) or _COMPARISON_FACTS.get((b, a))
    if not facts:
        return None, ""
    fact = facts.get(aspect) or facts.get("default") or ""
    if not fact:
        return None, ""
    # Bounded length: 1 paragraph cap.
    if len(fact) > 600:
        fact = fact[:597].rstrip() + "..."
    return pair, fact


_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "so", "to", "of",
    "in", "on", "at", "for", "by", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "it", "its", "this", "that", "these", "those", "i", "you",
    "we", "they", "me", "my", "your", "our", "their", "am",
    # Contraction-collapsed forms (apostrophes are stripped before
    # keyword extraction, so "I'm" / "you're" / "we'll" become these).
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "its", "theyre", "theyve", "theyll", "theyd",
    "weve", "well", "wed",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "cant", "couldnt", "shouldnt",
    "wouldnt", "wont", "shant", "mustnt",
    # Interrogatives: "what is X" vs "what is Y" always share "what".
    "what", "whats", "whos", "whens", "wheres", "whys", "hows",
    "why", "how", "when", "where", "who", "whom", "whose",
    "tell", "give", "show", "find",
    # Modals / auxiliaries that over-match.
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "must", "ought", "let", "like",
    # Generic intensifiers / fillers.
    "very", "really", "just", "much", "more", "less", "some", "any",
    "all", "every", "each", "no", "not", "yes", "ok", "okay",
})


def _keywords(text: str) -> set[str]:
    """Return the set of content-bearing lowercase tokens in *text*.

    Apostrophes are stripped so ``"what's"`` and ``"whats"`` hash to
    the same keyword — otherwise a user typing without the apostrophe
    never produces a keyword overlap against the training prompts.
    """
    # Strip apostrophes first so contractions fold: "what's" → "whats".
    cleaned = text.lower().replace("'", "").replace("’", "")
    tokens = re.findall(r"[a-z]+", cleaned)
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation (including apostrophes), collapse
    whitespace.  Used for the exact-match fast path: ``"Hi!" -> "hi"``,
    ``"what's huawei" -> "whats huawei"``.  Two inputs that normalise
    to the same string are treated as semantically identical.
    """
    text = text.lower().strip().replace("’", "'")
    # Drop apostrophes OUTRIGHT (contraction-folding) before turning
    # other punctuation into whitespace — otherwise ``what's`` becomes
    # ``what s`` and the user's ``whats`` will never match.
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Low-quality-row filter (Fix 1 of the 2026-04-25 corpus quality overhaul)
# ---------------------------------------------------------------------------
# The 110-scenario audit (D:/tmp/conversational_audit.md) found a cluster
# of "movie-line / single-token" failures at retrieval=1.00 — for example
# user "CEO" → "Seven."  These come from Cornell-movies / OpenSubs rows
# that *look* like Q→A under the existing kind/looks_like_qa filter but
# actually carry zero substantive answer content.  We drop them BEFORE
# the embedding matrix is built so they can never be a top-1 cosine
# match.  Entity-tool routes (e.g. "when was Microsoft founded?" →
# "1975.") are handled by the tool layer above retrieval, not by this
# index, so removing year-only / single-word rows here doesn't lose
# correct answers — the curated entity dict still owns those.
# ---------------------------------------------------------------------------

_LOW_QUALITY_NUMERIC_RE = re.compile(r"^\s*\d+(?:[.,]\d+)?\.?\s*$")


def _is_low_quality_response(response: str) -> bool:
    """Return True if *response* looks like a movie-script or single-token row.

    Reasons to drop:
        - Length under 15 chars ("Seven.", "Three.", "Yes.")
        - 2-or-fewer word answers ("1975.", "Steve Jobs.")
        - Movie-script style: starts with "- " or has rapid-fire dashes
        - Pure numeric / year-only answers
    """
    r = (response or "").strip()
    if len(r) < 15:
        return True
    word_count = len(r.split())
    if word_count <= 2:
        return True
    if r.startswith("- ") or r.count(" - ") >= 2:
        return True
    if _LOW_QUALITY_NUMERIC_RE.match(r):
        return True
    return False


# ---------------------------------------------------------------------------
# Lightweight BM25 re-ranker (Fix 2)
# ---------------------------------------------------------------------------
# Plain BM25 (Robertson 1994) over the retrieval index's history
# keywords.  Lightweight: pre-computes IDF + average doc length once
# at init, scoring is one pass over the query terms per query.  We use
# this as a re-ranker on top of the cosine top-K so a query like
# "tell me about apple" doesn't accept a Japan-paragraph row that
# happens to share a 32-d embedding direction.
#
# Pure stdlib + torch.tensor for the score vector.  No rank_bm25.
# ---------------------------------------------------------------------------

class _BM25:
    """BM25 scorer over the per-document keyword bags of the retriever.

    Scoring uses Okapi BM25 with the standard k1 / b parameters.  The
    document corpus is fixed at construction time — documents are
    represented as ``set[str]`` of content keywords (not raw token
    counts), matching how the retriever already tokenises histories
    via :func:`_keywords`.  IDF and doc-length statistics are pre-
    computed once.

    Scoring is intentionally bounded: each per-document score is
    clamped to a finite range and any NaN / inf is replaced with zero
    so a downstream z-score normalisation cannot blow up.
    """

    def __init__(
        self,
        history_keywords: list[set[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._n = len(history_keywords)
        # Document keyword bags as sets — same representation the
        # retriever already maintains for the keyword-overlap boost.
        self._docs: list[set[str]] = list(history_keywords)
        # Per-doc length = number of unique content keywords.  Using
        # set cardinality (rather than raw token count) keeps the
        # scorer aligned with how the retriever tokenises histories.
        self._doc_len = [max(1, len(d)) for d in self._docs]
        if self._n > 0:
            self._avgdl = sum(self._doc_len) / float(self._n)
        else:
            self._avgdl = 1.0
        # IDF over the document frequency of each term.  We use the
        # standard BM25 IDF formulation, floored at zero so very-common
        # terms can't pull the score negative.
        df: Counter[str] = Counter()
        for doc in self._docs:
            for term in doc:
                df[term] += 1
        self._idf: dict[str, float] = {}
        for term, dfreq in df.items():
            # Robertson-Sparck Jones IDF, with the standard +0.5
            # smoothing.  Floor at 0 so over-frequent terms don't
            # subtract from the score.
            num = self._n - dfreq + 0.5
            den = dfreq + 0.5
            idf = math.log((num / den) + 1.0)
            self._idf[term] = max(0.0, idf)

    def score(self, query_keywords: set[str]) -> torch.Tensor:
        """Return a ``[N]`` float32 tensor of BM25 scores against the corpus.

        An empty *query_keywords* produces a zero tensor (no keyword
        signal → no re-ranking contribution).  Per-document scores are
        bounded and NaN-safe.
        """
        scores = torch.zeros(self._n, dtype=torch.float32)
        if not query_keywords or self._n == 0:
            return scores
        k1 = self.k1
        b = self.b
        avgdl = self._avgdl if self._avgdl > 0 else 1.0
        # Walk the query terms — the BM25 formula factorises over
        # query terms, so we accumulate per-term contributions.  For
        # set-based docs the term frequency is 1 if present else 0,
        # which collapses the BM25 numerator/denominator nicely.
        for term in query_keywords:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            # Build a contribution vector over docs in one Python loop.
            # N is bounded by I3_RETRIEVAL_MAX_ENTRIES (≈4 k typical),
            # so this is a few microseconds per query term on CPU.
            for j, doc in enumerate(self._docs):
                if term not in doc:
                    continue
                tf = 1.0  # set membership: term either present or not
                dl = self._doc_len[j]
                norm = 1.0 - b + b * (dl / avgdl)
                contrib = idf * ((tf * (k1 + 1.0)) / (tf + k1 * norm))
                scores[j] += contrib
        # Safety: replace any NaN/inf the math could in principle have
        # produced (very small avgdl, pathological IDF) with zero so
        # downstream z-scoring can't go off the rails.
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp to a sane range — BM25 in practice never exceeds ~50
        # per-doc on a corpus this size, but cap as a backstop so a
        # corrupted IDF map can't dominate the cosine signal entirely.
        scores = scores.clamp(min=0.0, max=50.0)
        return scores


class ResponseRetriever:
    """Token-embedding retrieval over the training corpus.

    Attributes:
        tokenizer: The same ``SimpleTokenizer`` the SLM uses.
        model: The ``AdaptiveSLM`` whose token embeddings we reuse.
        histories: List of training user-side prompts.
        responses: List of training assistant-side responses (aligned
            with *histories* by index).
        embeddings: ``[N, d_model]`` float tensor, unit-normalised,
            used for cosine-similarity search.
        history_keywords: Per-entry keyword bags used by the boost term.
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        triples: list[dict[str, Any]],
        *,
        device: torch.device | str | None = None,
        embeddings_cache_path: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        # Optional on-disk cache for the [N, d_model] embedding matrix.
        # Building 1.5 k embeddings on CPU through the v2 model's
        # embedding layer (768-d) takes ~2 minutes; caching the matrix
        # to checkpoints/slm_v2/retrieval_embeddings.pt cuts subsequent
        # boots to <100 ms. We key the cache on (vocab_size, d_model,
        # n_histories) — if any of those change (e.g. corpus growth)
        # the cache is invalidated and rebuilt automatically.
        self._embeddings_cache_path: Path | None = (
            Path(embeddings_cache_path) if embeddings_cache_path is not None else None
        )
        # Detect the model's actual device so token-embedding lookups
        # don't fall foul of the "tensor on cuda:0, index on cpu" crash
        # that you'd otherwise get on GPU runs.  Fall back to CPU only
        # when the caller explicitly asks for it.
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        # Keep prompts / responses parallel — dropping duplicates on
        # the (history, response) pair keeps the index small and
        # avoids the retriever being biased toward phrases that
        # appeared many times in the corpus.
        # Index *only* the curated Q→A entries.  The filler corpus
        # (Wikipedia article sentences, hand-written domain prose) is
        # there for tokenizer coverage and would otherwise let
        # retrieval return random encyclopedia snippets in response to
        # casual prompts (e.g. "test second" → a quadtree segmentation
        # sentence).  We accept anything tagged ``qa`` plus untagged
        # legacy entries that look like Q→A (short history, ends with
        # a question mark, or starts with a question word).
        QUESTION_STARTS = (
            "what", "who", "when", "where", "why", "how", "is",
            "are", "do", "does", "can", "could", "should", "would",
            "tell", "explain", "define", "give", "show", "find",
            "i ", "i'", "im ", "im", "thanks", "thank", "hello",
            "hi", "hey", "good", "bye", "ok", "okay", "yes", "no",
        )

        def looks_like_qa(history: str) -> bool:
            h = history.strip().lower()
            if not h:
                return False
            if len(h.split()) > 12:
                return False
            if h.endswith("?"):
                return True
            return h.startswith(QUESTION_STARTS)

        seen: set[tuple[str, str]] = set()
        histories: list[str] = []
        responses: list[str] = []
        emotions: list[int] = []
        skipped_filler = 0
        skipped_low_quality = 0
        for t in triples:
            h = str(t.get("history", "")).strip()
            r = str(t.get("response", "")).strip()
            if not h or not r:
                continue
            kind = str(t.get("kind", "")).lower()
            if kind == "filler":
                skipped_filler += 1
                continue
            if not kind and not looks_like_qa(h):
                # Untagged legacy entries: only accept if the history
                # *looks* like a user prompt.
                skipped_filler += 1
                continue
            # Quality filter (Fix 1, 2026-04-25 corpus quality overhaul):
            # purge movie-line / single-token / year-only rows so they
            # cannot become a top-1 cosine match for an unrelated query
            # (the "CEO" → "Seven." failure mode).  Entity-tool routes
            # still own the legitimate single-token answers (e.g. the
            # "1975." for "when was Microsoft founded?" comes from the
            # entity dict, not from the retrieval index).
            if _is_low_quality_response(r):
                skipped_low_quality += 1
                continue
            key = (h.lower(), r.lower())
            if key in seen:
                continue
            seen.add(key)
            histories.append(h)
            responses.append(r)
            emotions.append(int(t.get("emotion", 0)))
        logger.info(
            "Retrieval index: kept %d Q→A entries, skipped %d filler, "
            "purged %d low-quality (movie-line / single-token / year-only).",
            len(histories), skipped_filler, skipped_low_quality,
        )

        # Optional environment-variable cap with **stratified sampling**.
        # Building the embedding matrix is O(N) forward passes through
        # the SLM token-embedding layer; on a 726 k-entry corpus that's
        # ~10 minutes on CPU, which makes server startup unusable while
        # the GPU is held by an overnight training run.  Setting
        # ``I3_RETRIEVAL_MAX_ENTRIES=N`` caps the index at N.
        #
        # The sampling is stratified: we keep **every** entry from the
        # curated source (which is the hand-tuned HMI / demo set —
        # greetings, self-description, Huawei / accessibility / edge
        # responses), and only sub-sample the noisier sources (Cornell
        # movies, OpenSubtitles, Wikipedia, etc.).  Without this, the
        # random sample drowns the curated 5 k entries in 700 k+ movie
        # subtitles and the canonical demo prompts get nonsense matches
        # ("hello" → "Cynthia, John, meet me at my house...").
        import os, random as _random
        max_entries = int(os.environ.get("I3_RETRIEVAL_MAX_ENTRIES", "0") or "0")
        if max_entries and len(histories) > max_entries:
            # Re-derive source per entry from the original triples, in
            # the same order as histories (we only kept entries that
            # passed the kind/looks_like_qa filter, but we kept their
            # input order, so we can walk both in lock-step).
            sources_per_entry: list[str] = []
            j = 0
            for t in triples:
                h = str(t.get("history", "")).strip()
                r = str(t.get("response", "")).strip()
                if not h or not r:
                    continue
                kind = str(t.get("kind", "")).lower()
                if kind == "filler":
                    continue
                if not kind and not looks_like_qa(h):
                    continue
                # Mirror the low-quality filter from the first pass so
                # the source-per-entry list stays in lock-step with the
                # filtered ``histories`` / ``responses`` arrays.
                if _is_low_quality_response(r):
                    continue
                key = (h.lower(), r.lower())
                # Only count the first occurrence — duplicates were
                # dropped above.
                if j < len(histories) and histories[j] == h and responses[j] == r:
                    sources_per_entry.append(str(t.get("source", "")))
                    j += 1
            if len(sources_per_entry) != len(histories):
                # Defensive fallback: shape mismatch shouldn't happen,
                # but if it does, fall back to plain random sampling.
                logger.warning(
                    "stratified sample: source-list shape mismatch (%d vs %d), "
                    "falling back to plain random sample",
                    len(sources_per_entry), len(histories),
                )
                rng = _random.Random(17)
                keep_idx = sorted(rng.sample(range(len(histories)), max_entries))
            else:
                # Retrieval = the curated demo knowledge base, full stop.
                # All other sources (Cornell, OpenSubs, PersonaChat,
                # DailyDialog, SQuAD, WikiText) are training-corpus
                # signal for the SLM but make terrible retrieval
                # candidates: at this corpus scale + the crude mean-pool
                # embedding model, novel queries cosine-match arbitrary
                # crowd-chat fragments at high scores ("do something" →
                # "I would like to transfer some money", "whats apple?"
                # → "Motorola 68040").  Restricting the index to the
                # curated set means:
                #   * known demo prompts answer cleanly via retrieval
                #   * everything else falls through to the SLM which
                #     can at least try to generate something coherent,
                #     and to OOD if the SLM output isn't coherent.
                curated_idx = [
                    i for i, s in enumerate(sources_per_entry) if s == "curated"
                ]
                non_curated_idx: list[int] = []  # nothing else indexed
                budget = max(0, max_entries - len(curated_idx))
                rng = _random.Random(17)
                if budget > 0 and non_curated_idx:
                    sampled = rng.sample(
                        non_curated_idx, k=min(budget, len(non_curated_idx)),
                    )
                else:
                    sampled = []
                keep_idx = sorted(set(curated_idx) | set(sampled))
            histories = [histories[i] for i in keep_idx]
            responses = [responses[i] for i in keep_idx]
            emotions = [emotions[i] for i in keep_idx]
            logger.info(
                "Retrieval index capped at %d entries (stratified: %d curated + %d sampled)",
                len(histories),
                sum(1 for i in keep_idx if i < len(sources_per_entry) and sources_per_entry[i] == "curated"),
                sum(1 for i in keep_idx if i < len(sources_per_entry) and sources_per_entry[i] != "curated"),
            )

        self.histories = histories
        self.responses = responses
        self.emotions = emotions
        self.history_keywords = [_keywords(h) for h in histories]
        # Build a case/punctuation-insensitive exact-match index so
        # common queries ("hi", "how are you", "hello") return their
        # canonical response in O(1) without going through the
        # embedding search.  Multiple history entries may normalise to
        # the same key — keep the first response seen.
        self._exact_index: dict[str, int] = {}
        # Iter 47 (2026-04-26): the normaliser strips ALL non-
        # alphanumeric characters, so emoji-only ("🤔") and
        # punctuation-only ("?") histories normalise to empty and
        # become unreachable via the fast path.  We keep a parallel
        # raw-exact index keyed on a lighter normalisation
        # (lowercase + collapse whitespace) so emoji and short symbol
        # queries can still hit their curated entries.
        self._exact_raw_index: dict[str, int] = {}
        for i, h in enumerate(histories):
            key = _normalise(h)
            if key and key not in self._exact_index:
                self._exact_index[key] = i
            raw_key = " ".join((h or "").lower().split())
            if raw_key and raw_key not in self._exact_raw_index:
                self._exact_raw_index[raw_key] = i

        logger.info(
            "Retrieval index built: %d unique (history, response) pairs",
            len(self.histories),
        )

        # Pre-compute embeddings for every history prompt.  We use the
        # SLM's token-embedding matrix (mean-pooled over the tokens of
        # the prompt) rather than running the full transformer — this
        # gives a perfectly reasonable sentence-level representation at
        # almost zero cost.
        self.embeddings = self._load_or_build_embeddings(histories)
        logger.info(
            "Retrieval embedding matrix: shape=%s dtype=%s",
            tuple(self.embeddings.shape),
            self.embeddings.dtype,
        )

        # Build the BM25 lexical scorer over the history keyword bags.
        # Cosine alone happily picks ``"Japan is an island nation..."``
        # for ``"tell me about apple"`` because both contain country/
        # tech-noun directions in the embedding space; BM25 over
        # keyword overlap penalises that match (no shared content
        # words) and gives the cosine signal a sanity check at
        # ranking time.  See ``_BM25`` above for the maths.
        self._bm25 = _BM25(self.history_keywords)
        logger.info(
            "BM25 re-ranker built: N=%d, avgdl=%.2f, vocab=%d terms",
            self._bm25._n, self._bm25._avgdl, len(self._bm25._idf),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _token_embeddings(self, ids: list[int]) -> torch.Tensor:
        """Return the ``[L, d_model]`` token embeddings for *ids*."""
        token_emb = self.model.embedding.token_embedding.embedding
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)
        return token_emb(idx)

    def _is_bpe_tokenizer(self) -> bool:
        """True when wrapped tokenizer is the byte-level BPE flavour."""
        return hasattr(self.tokenizer, "merges") and hasattr(
            self.tokenizer, "token_bytes"
        )

    def _encode_no_special(self, text: str) -> list[int]:
        """Tokenizer-agnostic encode that adds *no* BOS/EOS specials."""
        if self._is_bpe_tokenizer():
            # BPE keyword API: ``add_bos`` / ``add_eos`` (both default False).
            return self.tokenizer.encode(text)
        # Word-level SimpleTokenizer uses ``add_special``.
        return self.tokenizer.encode(text, add_special=False)

    @torch.no_grad()
    def _embed_one(self, text: str) -> torch.Tensor:
        """Return a single unit-normalised ``[d_model]`` vector for *text*."""
        ids = self._encode_no_special(text)
        if not ids:
            ids = [self.tokenizer.UNK_ID]
        emb = self._token_embeddings(ids).mean(dim=0)
        norm = torch.linalg.vector_norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb

    @torch.no_grad()
    def _embed_all(self, texts: list[str]) -> torch.Tensor:
        """Return an ``[N, d_model]`` unit-normalised matrix for *texts*."""
        vecs = [self._embed_one(t) for t in texts]
        return torch.stack(vecs, dim=0)

    def _load_or_build_embeddings(self, histories: list[str]) -> torch.Tensor:
        """Return the ``[N, d_model]`` embedding matrix, using a disk cache.

        Cache file format (when ``self._embeddings_cache_path`` is set):

        .. code-block:: python

            {
                "version": 1,
                "vocab_size": int,
                "d_model": int,
                "n_histories": int,
                "embeddings": Tensor[N, d_model],
            }

        On cache miss (file absent, or any of the size keys mismatch the
        current model/tokenizer/corpus), we rebuild from scratch and save.
        """
        cache_path = self._embeddings_cache_path
        try:
            d_model = int(self.model.embedding.token_embedding.embedding.embedding_dim)
        except AttributeError:
            d_model = int(getattr(self.model, "d_model", 0)) or 0
        try:
            vocab_size = int(self.model.embedding.token_embedding.embedding.num_embeddings)
        except AttributeError:
            vocab_size = int(getattr(self.tokenizer, "vocab_size", 0)) or 0

        if cache_path is not None and cache_path.is_file():
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(blob, dict)
                    and int(blob.get("vocab_size", -1)) == vocab_size
                    and int(blob.get("d_model", -1)) == d_model
                    and int(blob.get("n_histories", -1)) == len(histories)
                    and "embeddings" in blob
                ):
                    emb = blob["embeddings"].to(self.device)
                    logger.info(
                        "Retrieval embeddings cache HIT: %s (shape=%s)",
                        cache_path, tuple(emb.shape),
                    )
                    return emb
                logger.info(
                    "Retrieval embeddings cache stale at %s (rebuilding)",
                    cache_path,
                )
            except Exception:
                logger.exception(
                    "Failed to load retrieval embeddings cache %s; rebuilding.",
                    cache_path,
                )

        emb = self._embed_all(histories)
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "version": 1,
                        "vocab_size": vocab_size,
                        "d_model": d_model,
                        "n_histories": len(histories),
                        "embeddings": emb.detach().cpu(),
                    },
                    cache_path,
                )
                logger.info(
                    "Retrieval embeddings cache WRITE: %s (shape=%s)",
                    cache_path, tuple(emb.shape),
                )
            except Exception:
                logger.exception(
                    "Failed to write retrieval embeddings cache %s",
                    cache_path,
                )
        return emb

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @torch.no_grad()
    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        adaptation: dict[str, float] | None = None,
        query_for_embedding: str | None = None,
    ) -> list[tuple[float, str, str]]:
        """Return up to *top_k* ``(score, history, response)`` matches.

        *adaptation* is an optional dict of scalar adaptation axes
        (``formality``, ``verbosity``, ``cognitive_load`` etc.) that
        biases the ranking toward responses whose length matches the
        requested verbosity.  This is the hook that makes the
        cross-attention conditioning visible at retrieval time.

        *query_for_embedding* (optional) is the string used for the
        cosine-similarity path only.  The keyword-overlap boost still
        runs against the raw ``text`` so that prepending multi-turn
        conversation history to the query for context cannot pollute
        the keyword-overlap signal that gates the 0.85+ confidence
        chip on curated demo prompts.  Default ``None`` preserves the
        single-turn behaviour exactly: cosine and keyword paths both
        see the same string.
        """
        if not self.histories:
            return []

        # Cosine path uses the contextualised query when supplied;
        # keyword path always uses the raw single-turn text.
        embed_text = query_for_embedding if query_for_embedding else text
        q = self._embed_one(embed_text)
        # Cosine similarity reduces to a single matmul because both
        # query and index are unit-normalised.  Do the matmul on the
        # model's device, then pull the result back to CPU for scoring
        # (keyword boost + verbosity bias live on CPU).
        cosine = (self.embeddings @ q).detach().to(torch.device("cpu"))
        # Preserve the raw cosine for the displayed score: callers
        # show this on the UI chip, so we do NOT mix in the BM25 +
        # verbosity terms that drive ranking.  The cosine is bounded
        # to [-1, 1] by construction (both vectors are unit-normalised).
        cosine = torch.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=-1.0)

        # Keyword-overlap boost keeps exact-match prompts ("hello" ==
        # "hello") at the top even if the embedding layer hasn't
        # learned to separate them well.  Folded into the *ranking*
        # signal only.
        query_kw = _keywords(text)
        sims = cosine.clone()
        if query_kw:
            boosts = torch.tensor(
                [
                    (
                        len(query_kw & kw) / max(1, len(query_kw | kw))
                        if kw else 0.0
                    )
                    for kw in self.history_keywords
                ],
                dtype=torch.float32,
            )
            sims = sims + 0.35 * boosts

        # BM25 lexical re-rank (Fix 2, 2026-04-25 corpus-quality overhaul).
        # Cosine + Jaccard alone confidently mis-routes
        #   "tell me about apple" → "Japan is an island..."
        # because the query keyword "apple" overlaps zero rows but the
        # cosine direction lands on a Japan-paragraph row.  BM25 over
        # the same keyword bags penalises rows with no shared content
        # words.  We z-score normalise both signals (subtract mean,
        # divide by std) and combine with weights 0.55/0.45 so the
        # final ranking reflects both semantic and lexical agreement.
        # The displayed cosine on the chip is unchanged — only the
        # *order* of the top-k is re-shuffled.
        if query_kw:
            try:
                bm25 = self._bm25.score(query_kw)
            except Exception:  # pragma: no cover - defensive
                bm25 = torch.zeros_like(cosine)

            def _z(v: torch.Tensor) -> torch.Tensor:
                if v.numel() == 0:
                    return v
                std = v.std()
                if not torch.isfinite(std) or std.item() < 1e-6:
                    return torch.zeros_like(v)
                return (v - v.mean()) / std

            cos_z = _z(cosine)
            bm_z = _z(bm25)
            # Final ranking score: weighted sum of z-scored signals.
            # Stored separately from ``sims`` (which still carries the
            # Jaccard boost for the legacy keyword-overlap signal so
            # exact-match prompts stay sticky) and combined into the
            # topk pick below.  Bound so a degenerate std couldn't
            # produce a NaN that escapes into the ranking.
            rank_score = 0.55 * cos_z + 0.45 * bm_z
            rank_score = torch.nan_to_num(
                rank_score, nan=0.0, posinf=0.0, neginf=0.0,
            )
            # Fold the rank_score into ``sims`` so the verbosity bias
            # below still applies on top.  Scale the rank_score by 0.5
            # so it doesn't dominate the existing Jaccard boost on
            # canonical demo prompts.
            sims = sims + 0.5 * rank_score

        # Verbosity bias: if the user's current adaptation requests a
        # short / concise response, boost shorter paired responses and
        # vice versa.
        if adaptation:
            verbosity = float(adaptation.get("verbosity", 0.5))
            lengths = torch.tensor(
                [len(r.split()) for r in self.responses], dtype=torch.float32
            )
            # Normalise lengths to [0, 1] via a soft log-scale.
            norm_len = (lengths.log1p() / lengths.log1p().max().clamp_min(1.0))
            target = torch.full_like(norm_len, verbosity)
            length_fit = 1.0 - (norm_len - target).abs()
            sims = sims + 0.15 * length_fit

        # Rank by combined ``sims``, but report the raw cosine on the
        # chip — UI invariant: displayed score is bounded [0,1] and
        # represents semantic similarity, NOT the internal ranking.
        top = torch.topk(sims, k=min(top_k, sims.shape[0]))
        out: list[tuple[float, str, str]] = []
        for idx in top.indices.tolist():
            display_score = float(cosine[idx].item())
            out.append(
                (display_score, self.histories[idx], self.responses[idx])
            )
        return out

    @torch.no_grad()
    def consistency_check(
        self,
        text: str,
        *,
        adaptation: dict[str, float] | None = None,
        query_for_embedding: str | None = None,
    ) -> dict[str, Any]:
        """Self-consistency check on the top-3 retrieval candidates.

        Phase B.6 (2026-04-25): when the borderline retrieval band
        (0.65–0.85) lands on a candidate, retrieve TOP-3 and check
        whether at least 2 of them have similar response text using
        token overlap > 0.4 as the agreement threshold.

        Returns a dict with shape::

            {"top_k": int,
             "top_scores": [float, ...],
             "agreement_pairs": int,         # how many of C(3,2)=3 pairs agreed
             "consistent": bool,             # True if ≥2 candidates agree
             "winning_response": str | None, # canonical of the agreeing cluster
             "winning_score": float | None}

        Used by the engine's borderline-fallback path: when
        ``consistent=True`` we elevate the borderline candidate to a
        confident retrieval answer; when ``consistent=False`` we surface
        the uncertainty as a clarification.
        """
        out: dict[str, Any] = {
            "top_k": 0,
            "top_scores": [],
            "agreement_pairs": 0,
            "consistent": False,
            "winning_response": None,
            "winning_score": None,
        }
        if not self.histories:
            return out
        try:
            matches = self.query(
                text,
                top_k=3,
                adaptation=adaptation,
                query_for_embedding=query_for_embedding,
            )
        except Exception:
            return out
        if not matches:
            return out
        out["top_k"] = len(matches)
        out["top_scores"] = [
            float(max(0.0, min(1.0, s))) for s, _, _ in matches
        ]
        # Pairwise token-overlap agreement.  We use Jaccard over content
        # keywords; agreement threshold is 0.4 (mid-overlap).
        kw_lists = [_keywords(resp) for _, _, resp in matches]
        pairs = 0
        agreeing_indices: set[int] = set()
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                if not kw_lists[i] or not kw_lists[j]:
                    continue
                overlap = (
                    len(kw_lists[i] & kw_lists[j])
                    / max(1, len(kw_lists[i] | kw_lists[j]))
                )
                if overlap > 0.4:
                    pairs += 1
                    agreeing_indices.add(i)
                    agreeing_indices.add(j)
        out["agreement_pairs"] = pairs
        if pairs >= 1 and len(agreeing_indices) >= 2:
            out["consistent"] = True
            # Pick the highest-scoring response in the agreeing cluster.
            best_idx = min(agreeing_indices, key=lambda k: -matches[k][0])
            out["winning_response"] = matches[best_idx][2]
            out["winning_score"] = float(
                max(0.0, min(1.0, matches[best_idx][0]))
            )
        return out

    def best(
        self,
        text: str,
        *,
        adaptation: dict[str, float] | None = None,
        min_score: float = 0.40,
        tool_route: bool = True,
        query_for_embedding: str | None = None,
        compare_fallback_pair: tuple[str, str] | None = None,
    ) -> tuple[str, float] | None:
        """Return ``(response, score)`` of the single best match, or None.

        Retrieval is intentionally strict: at this corpus size a false
        match is worse than no match because an OOD fallback tells the
        user honestly "that's outside my training", while a false match
        produces "Seven." in reply to "what's apple?".  The gate
        enforces five cumulative conditions:

        1. **Tool routes.**  Arithmetic expressions and overt
           hostility are answered deterministically — cosine retrieval
           is skipped entirely so the chip shows ``tool: math`` or
           ``tool: refuse``, not a bogus ``conf 1.37``.
        2. **Exact-match fast path.**
        3. **Empty-keyword guard.**  Queries with zero content words
           (digits, symbols, pure stopwords) cannot produce a
           meaningful match — noise in, noise out.  Refuse.
        4. **OOV guard.**  >50 % of content words missing from the
           tokenizer vocab → mean-pooled query is UNK noise → refuse.
        5. **Keyword-overlap veto.**  Every query with content words
           must share at least one content word with the matched
           history, or cross a higher cosine bar (0.85+).

        Scores are clamped to [0, 1] — the internal ranking score
        includes Jaccard and verbosity boosts that can push the sum
        above 1.0, but the chip we return should look like a cosine.

        *query_for_embedding* (optional) is forwarded to
        :meth:`query` so callers that maintain multi-turn context
        can have prior turns prepended to the cosine query without
        polluting the exact-match / keyword-overlap / OOV gates,
        which all run against the raw single-turn ``text``.  Default
        ``None`` preserves the historical behaviour byte-for-byte.
        """
        # ── 0. Tool routes: math solver, hostility refusal, entity facts ──
        # These stamp the last-invoked tool name on the retriever so
        # the engine can surface it as a chip (``tool: math`` etc.)
        # instead of ``retrieval``.
        self._last_tool: str | None = None
        if tool_route:
            if _is_math_expr(text):
                answer = _eval_math(text)
                if answer is not None:
                    self._last_tool = "math"
                    # 0.99 (not 1.0) so we can distinguish tool answers
                    # from exact-match retrievals on the UI side.
                    return answer, 0.99
            if _is_hostility(text):
                self._last_tool = "refuse"
                return _hostility_reply(text), 0.99
            # Entity knowledge tool — answers
            #   "where is huawei located?" → "Shenzhen, China."
            #   "who founded apple?"        → "Steve Jobs ..."
            # The engine runs this against the *co-reference-resolved*
            # query, so multi-turn follow-ups like "where are they?"
            # land here once the resolver has rewritten the pronoun.
            entity, fact = _entity_lookup(text)
            if entity is not None and fact:
                self._last_tool = "entity"
                return fact, 0.99
            # Country-attribute tool (Phase 14, 2026-04-25).  Powers
            # follow-ups like "language of japan?" / "capital of
            # germany?" — the bare-noun rewriter in coref.py +
            # engine.py rewrites bare "language?" after a country
            # topic into the full question shape this matches.
            country, country_fact = _country_lookup(text)
            if country is not None and country_fact:
                self._last_tool = "entity"
                return country_fact, 0.99
            # Graph-compose tool — multi-triple composed answers (KG).
            # Handles "what does microsoft own?", "who are apple's
            # competitors?", and overview queries the single-fact entity
            # tool above doesn't cover.  Returns a 0.98-confidence stamp
            # (slightly below the entity tool's 0.99 so the chip
            # priority is unambiguous).
            graph_subj, graph_ans = _graph_compose_lookup(text)
            if graph_subj is not None and graph_ans:
                self._last_tool = "graph_compose"
                return graph_ans, 0.98
            # Comparison tool — answers
            #   "python vs rust"           → curated paragraph
            #   "is apple bigger than huawei?" → curated paragraph
            #   "which one is bigger?"     → curated paragraph if the
            #                                engine supplied two recent
            #                                entities via fallback_pair.
            # Pairs not in the curated table return (None, "") and we
            # fall through to retrieval — never fabricate.
            cmp_pair, cmp_fact = _compare_lookup(
                text, fallback_pair=compare_fallback_pair,
            )
            if cmp_pair is not None and cmp_fact:
                self._last_tool = "compare"
                return cmp_fact, 0.99

        # ── 1. Exact-match fast path ────────────────────────────
        key = _normalise(text)
        if key and key in self._exact_index:
            idx = self._exact_index[key]
            return self.responses[idx], 1.0

        # ── 1b. Raw-exact-match fast path (iter 47) ─────────────
        # For emoji-only and punctuation-only queries — "🤔", "?",
        # "..." — that normalise to empty.  Keyed on lowercase +
        # whitespace-collapsed form so they still reach the curated
        # entry.
        raw_key = " ".join((text or "").lower().split())
        if raw_key and raw_key in self._exact_raw_index:
            idx = self._exact_raw_index[raw_key]
            return self.responses[idx], 1.0

        # ── 2. Empty-keyword guard ──────────────────────────────
        query_kw = _keywords(text)
        if not query_kw:
            # No content keywords — numbers-only, symbols-only, or
            # pure stopwords ("ok", "yes").  Pure-stopword queries
            # should have matched the exact-match table above; if
            # they reached here, retrieval will pick noise.
            return None

        # ── 3. OOV guard ────────────────────────────────────────
        if self._oov_rate(query_kw) > 0.5:
            return None

        # ── 4. Embedding-based nearest neighbour with keyword veto ──
        matches = self.query(
            text,
            top_k=5,
            adaptation=adaptation,
            query_for_embedding=query_for_embedding,
        )
        if not matches:
            return None
        # Low-information keywords don't disambiguate topic — "describe
        # a sunset in one line" and "That would be two." share "one"
        # but have nothing in common semantically.  Strip these from
        # the overlap calculation so a spurious match on a numeric/
        # quantifier/generic-verb token alone never commits retrieval.
        _LOW_INFO_KW = {
            # Generic verbs — don't carry topic
            "describe", "explain", "tell", "show", "give", "make",
            "name", "list", "say", "speak", "talk", "ask",
            "want", "need", "like", "think", "feel", "know",
            # Numeric / quantifier — too cheap to license a match
            "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "few", "many", "lot", "lots",
            "some", "any", "all", "every", "each", "more", "less",
            # Generic frame nouns — don't carry topic
            "thing", "things", "stuff", "way", "ways", "kind", "type",
            "sort", "line", "lines", "piece", "part", "side",
            "question", "answer", "reply", "response",
            # Comparison frame — "difference between X and Y" must
            # match on X or Y, not just on the comparison word itself
            "difference", "differences", "compare", "comparison",
            "vs", "versus", "between",
        }
        for raw_score, history, response in matches:
            # Display score is clamped to [0, 1] — the raw score is
            # cosine plus rank-boost terms that can exceed 1.0, but
            # the UI chip must stay interpretable.
            score = max(0.0, min(1.0, float(raw_score)))
            if score < min_score:
                break
            match_kw = _keywords(history)
            overlap = query_kw & match_kw
            content_overlap = overlap - _LOW_INFO_KW
            content_query = query_kw - _LOW_INFO_KW
            # If the query has ANY substantive content keyword, the
            # overlap MUST include at least one — a generic-verb or
            # numeric collision alone is not enough.  This kills:
            #   "describe a sunset in one line" → "That would be two."
            # caught in the 2026-04-26 user-emulation audit.
            if len(content_query) >= 1 and not content_overlap:
                continue
            # And for genuinely substantive queries (3+ content words),
            # require the overlap to cover at least 1/3 of those words
            # so a single fluke noun doesn't commit a wildly-off-topic
            # paragraph.
            if (
                len(content_query) >= 3
                and len(content_overlap) < max(1, len(content_query) // 3)
            ):
                continue
            if overlap:
                return response, score
            # No overlap but cosine is very high — single-keyword
            # paraphrase ("help me" ↔ "help"); still accept.
            if len(query_kw) == 1 and score >= 0.85:
                return response, score
        return None

    def _oov_rate(self, keywords: set[str]) -> float:
        """Fraction of *keywords* that don't have a token id.

        ``SimpleTokenizer.token_to_id`` is the authoritative vocab
        mapping; anything not in it becomes ``[UNK]`` at encode time
        and is therefore useless for the retrieval cosine.

        For the byte-level BPE tokenizer (v2) this gate is moot — every
        possible UTF-8 byte is in the vocab, so the encoder can never
        emit ``[UNK]``.  We short-circuit to 0.0 to keep the OOV gate
        from rejecting legitimate emotional / small-talk prompts whose
        word-level keywords don't appear as standalone surface forms in
        the BPE merge table (they're encoded as multi-piece sequences).
        """
        if not keywords:
            return 0.0
        # BPE: every byte is encodable, so no concept of OOV applies.
        if self._is_bpe_tokenizer():
            return 0.0
        vocab = getattr(self.tokenizer, "token_to_id", {}) or {}
        missing = sum(1 for kw in keywords if kw not in vocab)
        return missing / len(keywords)


# ---------------------------------------------------------------------------
# Module-level smoke tests for the math word-form exponent fix
# (Phase 14, 2026-04-25).  These are executed only when this file is
# run as a script — `python -m i3.slm.retrieval` — and exist as a
# fast-feedback regression net for the `_normalise_math` /
# `_is_math_expr` / `_eval_math` chain.
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    _MATH_SMOKE_CASES: tuple[tuple[str, str], ...] = (
        ("99 squared", "9801."),
        ("nine squared", "81."),
        ("eleven cubed", "1331."),
        ("what is 99 squared", "9801."),
        ("what's 5 cubed", "125."),
        ("two to the power of three", "8."),
        ("4 to the 3", "64."),
    )
    for prompt, expected in _MATH_SMOKE_CASES:
        got = _eval_math(prompt)
        assert got == expected, f"_eval_math({prompt!r}) = {got!r}, expected {expected!r}"
        assert _is_math_expr(prompt), f"_is_math_expr({prompt!r}) returned False"
    print("retrieval.py math smoke tests passed.")

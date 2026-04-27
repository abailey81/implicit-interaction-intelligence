"""Lightweight per-session entity tracking + pronoun resolution.

This module exists to fix the multi-turn understanding gap in the I3
chat: previously a follow-up like ``"where are they located?"`` after
``"tell me about huawei"`` would pick a high-cosine but semantically
wrong retrieval match (``"Nice. Favourite genre?"``) because the
pronoun-laden short query carries no useful keyword signal on its own.

We solve it with two coupled rule-based components, both pure Python
with no external NER / parsing dependency:

1.  :class:`EntityTracker` keeps a bounded recency stack of named
    entities seen on each turn (extracted from BOTH the user message
    and the assistant response, so an answer mentioning *Huawei* lands
    the entity even if the user only said *"tell me about it"*).

2.  :meth:`EntityTracker.resolve` detects pronouns / referring
    expressions in the *current* user message and rewrites the query
    using the most recent compatible entity from the stack.

The design draws on Hobbs (1978) "Resolving pronoun references" and
the centring-theory tradition (Grosz, Joshi & Weinstein, 1995).  We
don't replicate full centring — instead we keep a recency stack and
pick the most recent compatible entity when a pronoun is detected,
which is sufficient for the I3 demo's two-three-turn conversational
horizon while being O(1) per turn.

Entity extraction is rule-based:
    - Curated entity list (Huawei, Apple, Microsoft, Google, OpenAI,
      ChatGPT, Python, Rust, Linux, Einstein, Darwin, Shakespeare,
      Berlin, Paris, London, Shenzhen, etc.).
    - Multi-word allowlist for orgs/places like ``"Steve Jobs"`` /
      ``"New York"`` / ``"San Francisco"``.
    - Capitalised-token fallback for tokens that aren't sentence-
      initial articles/pronouns.

Pronoun resolution is rule-based:
    - Personal pronouns (``it``, ``they``, ``them``, ``their``,
      ``its``) → most recent ORG / PLACE / TOPIC.
    - Demonstratives (``this``, ``that``, ``these``, ``those``) →
      most recent topic.
    - Definite descriptions (``the company``, ``the team``,
      ``the place``) → recent compatible entity.

The tracker is bounded:
    - Per-session entity stack capped at ``max_entities_per_session``
      (default 16).
    - Total tracked sessions LRU-capped at 1000.

Every public call is wrapped at the call-site (in the engine) in
try/except so a tracker bug can never block a turn — the worst case
degenerates to single-turn behaviour.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntityFrame:
    """A single entity observed in the conversation.

    Attributes:
        text: The surface form as observed (e.g. ``"Huawei"``).
        canonical: Lowercase canonical form used for retrieval lookups
            (e.g. ``"huawei"``).
        kind: One of ``'org'`` / ``'person'`` / ``'place'`` /
            ``'topic'`` / ``'unknown'``.
        last_turn_idx: Turn index when this entity was last mentioned
            (used to age out stale references).
        gender_pronoun: For ``kind="person"``, one of ``"he"`` /
            ``"she"`` / ``"they"`` driving he/him/his vs she/her/hers
            resolution.  ``None`` for non-person frames.
    """

    text: str
    canonical: str
    kind: str
    last_turn_idx: int
    gender_pronoun: str | None = None
    # Iteration 18 (2026-04-26): user-anchor turn — set when the
    # USER explicitly named this topic in their message (e.g. typed
    # "what is transformer").  Used to give user-anchored topics
    # extra stickiness in the priority walk: when an anchored
    # topic is within ``ANCHOR_TURNS`` of the current turn, it
    # outranks any topic the ASSISTANT mentioned later, even if the
    # assistant mention is more recent.  Without this, after asking
    # "what is transformer" then 3 turns of explanation that mention
    # "neural network", a bare "explain that" would resolve to neural
    # network instead of transformer.  None = not user-anchored.
    user_anchor_turn: int | None = None
    # Iter 31 (2026-04-26): first-ever user-anchor turn — set only on
    # the FIRST user mention and never updated.  Used by the "back to
    # the start" handler to find the topic the user first established
    # in this session, regardless of subsequent re-anchors.
    first_anchor_turn: int | None = None


@dataclass
class ResolutionResult:
    """Outcome of attempting to resolve pronouns in a user query.

    Attributes:
        original_query: The raw user text as received.
        resolved_query: Rewritten query with the pronoun substituted
            for the resolved entity (or unchanged when no resolution
            occurred).
        used_entity: The :class:`EntityFrame` that won the
            resolution, or ``None``.
        used_pronoun: The pronoun (or referring expression) that
            triggered the resolution, or ``None``.
        confidence: Resolution confidence in [0, 1].  ``1.0`` for
            no-op (no pronoun detected).  Pronoun-substitution
            confidence is a function of (a) recency of the
            referent and (b) whether other compatible entities
            recently competed for the slot.
        reasoning: Short human-readable explanation, suitable for
            the WS reasoning trace.
    """

    original_query: str
    resolved_query: str
    used_entity: EntityFrame | None
    used_pronoun: str | None
    confidence: float
    reasoning: str


def resolution_to_dict(res: ResolutionResult | None) -> dict | None:
    """Serialise a :class:`ResolutionResult` to a JSON-safe dict.

    Returns ``None`` when *res* is itself ``None`` or when the
    resolution was a no-op (no entity used) — the WS layer treats the
    field as optional and we don't want to ship empty noise.
    """
    if res is None:
        return None
    if res.used_entity is None and res.used_pronoun is None:
        return None
    ent = res.used_entity
    return {
        "original_query": res.original_query,
        "resolved_query": res.resolved_query,
        "used_entity": (
            None
            if ent is None
            else {
                "text": ent.text,
                "canonical": ent.canonical,
                "kind": ent.kind,
                "last_turn_idx": int(ent.last_turn_idx),
                "gender_pronoun": ent.gender_pronoun,
            }
        ),
        "used_pronoun": res.used_pronoun,
        "confidence": float(round(res.confidence, 3)),
        "reasoning": res.reasoning,
    }


# ---------------------------------------------------------------------------
# Curated entity catalogue
# ---------------------------------------------------------------------------
# Each entry is ``{canonical: {"kind": str, "aliases": list[str]}}``.
# Aliases are matched case-insensitively against the surface tokens.
# Multi-word aliases (``"Steve Jobs"``, ``"New York"``) are matched
# *before* the single-token pass so they don't get split.
# ---------------------------------------------------------------------------

_ENTITY_CATALOGUE: dict[str, dict] = {
    # Orgs ---------------------------------------------------------------
    "huawei": {"kind": "org", "aliases": ["huawei", "huawei R&D"]},
    "apple": {"kind": "org", "aliases": ["apple", "apple inc"]},
    "microsoft": {"kind": "org", "aliases": ["microsoft", "msft"]},
    "google": {"kind": "org", "aliases": ["google", "alphabet"]},
    "openai": {"kind": "org", "aliases": ["openai", "open ai"]},
    "anthropic": {"kind": "org", "aliases": ["anthropic"]},
    "meta": {"kind": "org", "aliases": ["meta", "facebook", "fb"]},
    "amazon": {"kind": "org", "aliases": ["amazon", "aws"]},
    "ibm": {"kind": "org", "aliases": ["ibm"]},
    "samsung": {"kind": "org", "aliases": ["samsung"]},
    "tesla": {"kind": "org", "aliases": ["tesla"]},
    "nvidia": {"kind": "org", "aliases": ["nvidia"]},
    "intel": {"kind": "org", "aliases": ["intel"]},
    "sony": {"kind": "org", "aliases": ["sony"]},
    # Tech / topics — programming languages, OSes, products --------------
    "python": {"kind": "topic", "aliases": ["python"]},
    "rust": {"kind": "topic", "aliases": ["rust"]},
    "javascript": {"kind": "topic", "aliases": ["javascript", "js"]},
    "java": {"kind": "topic", "aliases": ["java"]},
    "c++": {"kind": "topic", "aliases": ["c++", "cpp"]},
    "go": {"kind": "topic", "aliases": ["golang"]},
    "linux": {"kind": "topic", "aliases": ["linux"]},
    "windows": {"kind": "topic", "aliases": ["windows"]},
    "macos": {"kind": "topic", "aliases": ["macos", "mac os"]},
    "android": {"kind": "topic", "aliases": ["android"]},
    "ios": {"kind": "topic", "aliases": ["ios"]},
    "chatgpt": {"kind": "topic", "aliases": ["chatgpt", "chat gpt"]},
    "gpt": {"kind": "topic", "aliases": ["gpt"]},
    "claude": {"kind": "topic", "aliases": ["claude"]},
    "gemini": {"kind": "topic", "aliases": ["gemini"]},
    "bert": {"kind": "topic", "aliases": ["bert"]},
    # Tech / topics — concepts -------------------------------------------
    "machine learning": {"kind": "topic", "aliases": ["machine learning", "ml"]},
    "deep learning": {"kind": "topic", "aliases": ["deep learning"]},
    "neural network": {"kind": "topic", "aliases": ["neural network", "neural networks"]},
    "transformer": {"kind": "topic", "aliases": ["transformer", "transformers"]},
    "attention mechanism": {"kind": "topic", "aliases": ["attention mechanism"]},
    "reinforcement learning": {"kind": "topic", "aliases": ["reinforcement learning", "rl"]},
    "edge computing": {"kind": "topic", "aliases": ["edge computing"]},
    "cloud computing": {"kind": "topic", "aliases": ["cloud computing"]},
    "encryption": {"kind": "topic", "aliases": ["encryption", "cryptography"]},
    "blockchain": {"kind": "topic", "aliases": ["blockchain"]},
    "internet": {"kind": "topic", "aliases": ["internet", "the internet"]},
    "websocket": {"kind": "topic", "aliases": ["websocket", "websockets"]},
    "http": {"kind": "topic", "aliases": ["http", "https"]},
    "tcp": {"kind": "topic", "aliases": ["tcp", "tcp/ip"]},
    "dns": {"kind": "topic", "aliases": ["dns"]},
    "machine vision": {"kind": "topic", "aliases": ["machine vision", "computer vision"]},
    "natural language processing": {"kind": "topic", "aliases": ["natural language processing", "nlp"]},
    "ai": {"kind": "topic", "aliases": ["ai", "artificial intelligence"]},
    # History / events --------------------------------------------------
    "world war 2": {"kind": "topic", "aliases": [
        "world war 2", "world war ii", "wwii", "ww2", "second world war",
    ]},
    "world war 1": {"kind": "topic", "aliases": [
        "world war 1", "world war i", "wwi", "ww1", "first world war",
    ]},
    "roman empire": {"kind": "topic", "aliases": [
        "the roman empire", "roman empire", "rome",
    ]},
    "cold war": {"kind": "topic", "aliases": ["the cold war", "cold war"]},
    # People (historical) ----------------------------------------------
    "einstein": {"kind": "person", "aliases": ["albert einstein", "einstein"]},
    "newton": {"kind": "person", "aliases": ["isaac newton", "newton"]},
    "darwin": {"kind": "person", "aliases": ["charles darwin", "darwin"]},
    # Tech / topics — science -------------------------------------------
    "photosynthesis": {"kind": "topic", "aliases": ["photosynthesis"]},
    "mitosis": {"kind": "topic", "aliases": ["mitosis"]},
    "gravity": {"kind": "topic", "aliases": ["gravity"]},
    "evolution": {"kind": "topic", "aliases": ["evolution"]},
    "dna": {"kind": "topic", "aliases": ["dna"]},
    "rna": {"kind": "topic", "aliases": ["rna"]},
    "atoms": {"kind": "topic", "aliases": ["atoms", "atom"]},
    "electrons": {"kind": "topic", "aliases": ["electrons", "electron"]},
    "quantum mechanics": {"kind": "topic", "aliases": ["quantum mechanics", "quantum physics"]},
    "relativity": {"kind": "topic", "aliases": ["relativity"]},
    "climate change": {"kind": "topic", "aliases": ["climate change", "global warming"]},
    "ecosystems": {"kind": "topic", "aliases": ["ecosystems", "ecosystem"]},
    "neurons": {"kind": "topic", "aliases": ["neurons", "neuron"]},
    "mitochondria": {"kind": "topic", "aliases": ["mitochondria", "mitochondrion"]},
    "respiration": {"kind": "topic", "aliases": ["respiration", "cellular respiration"]},
    "chlorophyll": {"kind": "topic", "aliases": ["chlorophyll"]},
    # Iter 32 (2026-04-26): science topics that have curated overlay
    # entries but were missing from the catalog, so explicit pivot
    # phrases ("and dark matter", "what about black holes") couldn't
    # force-anchor on them.
    "dark matter": {"kind": "topic", "aliases": ["dark matter"]},
    "dark energy": {"kind": "topic", "aliases": ["dark energy"]},
    "string theory": {"kind": "topic", "aliases": ["string theory", "string theories"]},
    "quantum entanglement": {"kind": "topic", "aliases": ["quantum entanglement", "entanglement"]},
    "black hole": {"kind": "topic", "aliases": ["black hole", "black holes"]},
    "consciousness": {"kind": "topic", "aliases": ["consciousness"]},
    "universe": {"kind": "topic", "aliases": ["universe", "cosmos"]},
    "time": {"kind": "topic", "aliases": ["time"]},
    "space": {"kind": "topic", "aliases": ["space", "outer space"]},
    "virus": {"kind": "topic", "aliases": ["virus", "viruses"]},
    "cell": {"kind": "topic", "aliases": ["cell", "cells"]},
    "brain": {"kind": "topic", "aliases": ["brain", "brains"]},
    "magnet": {"kind": "topic", "aliases": ["magnet", "magnets"]},
    "neural network": {"kind": "topic", "aliases": ["neural network", "neural networks", "nn"]},
    "deep learning": {"kind": "topic", "aliases": ["deep learning", "dl"]},
    "internet": {"kind": "topic", "aliases": ["internet", "the internet", "the web"]},
    "database": {"kind": "topic", "aliases": ["database", "databases", "db"]},
    "computer": {"kind": "topic", "aliases": ["computer", "computers"]},
    # Iter 34 (2026-04-26): more tech/cloud/ML aliases that recruiters
    # commonly probe.
    "cnn": {"kind": "topic", "aliases": ["cnn", "convolutional neural network", "convolutional neural networks"]},
    "rnn": {"kind": "topic", "aliases": ["rnn", "recurrent neural network", "recurrent neural networks"]},
    "lstm": {"kind": "topic", "aliases": ["lstm", "long short term memory"]},
    "gan": {"kind": "topic", "aliases": ["gan", "gans", "generative adversarial network"]},
    "aws": {"kind": "org", "aliases": ["aws", "amazon web services"]},
    "azure": {"kind": "org", "aliases": ["azure", "microsoft azure"]},
    "gcp": {"kind": "org", "aliases": ["gcp", "google cloud", "google cloud platform"]},
    "kubernetes": {"kind": "topic", "aliases": ["kubernetes", "k8s"]},
    "docker": {"kind": "topic", "aliases": ["docker"]},
    "git": {"kind": "topic", "aliases": ["git"]},
    "github": {"kind": "org", "aliases": ["github"]},
    "javascript": {"kind": "topic", "aliases": ["javascript", "js", "ecmascript"]},
    "react": {"kind": "topic", "aliases": ["react", "react.js", "reactjs"]},
    "pytorch": {"kind": "topic", "aliases": ["pytorch"]},
    "tensorflow": {"kind": "topic", "aliases": ["tensorflow", "tf"]},
    "huggingface": {"kind": "org", "aliases": ["huggingface", "hugging face"]},
    "openai": {"kind": "org", "aliases": ["openai", "open ai"]},
    "anthropic": {"kind": "org", "aliases": ["anthropic"]},
    "claude": {"kind": "topic", "aliases": ["claude"]},
    "gpt": {"kind": "topic", "aliases": ["gpt", "chatgpt"]},
    "llm": {"kind": "topic", "aliases": ["llm", "llms", "large language model", "large language models"]},
    "rag": {"kind": "topic", "aliases": ["rag", "retrieval augmented generation"]},
    "fine tuning": {"kind": "topic", "aliases": ["fine tuning", "fine-tuning", "finetuning"]},
    "lora": {"kind": "topic", "aliases": ["lora", "low rank adaptation"]},
    "moe": {"kind": "topic", "aliases": ["moe", "mixture of experts"]},
    "act": {"kind": "topic", "aliases": ["act", "adaptive computation time"]},
    "transformer architecture": {"kind": "topic", "aliases": ["transformer architecture"]},
    "self attention": {"kind": "topic", "aliases": ["self attention", "self-attention"]},
    "tokenizer": {"kind": "topic", "aliases": ["tokenizer", "tokenization", "tokenizing"]},
    "embedding": {"kind": "topic", "aliases": ["embedding", "embeddings"]},
    # Iter 36 (2026-04-26): more abstract topics that recruiters and
    # casual users probe with single-word queries.  Adding them here
    # lets the bare-word auto-anchor rewrite "love" / "happiness" /
    # "intelligence" / "creativity" / "beauty" / "truth" → "what is
    # love" etc. rather than falling to OOD.
    "love": {"kind": "topic", "aliases": ["love"]},
    "happiness": {"kind": "topic", "aliases": ["happiness"]},
    "intelligence": {"kind": "topic", "aliases": ["intelligence"]},
    "creativity": {"kind": "topic", "aliases": ["creativity"]},
    "beauty": {"kind": "topic", "aliases": ["beauty"]},
    "truth": {"kind": "topic", "aliases": ["truth"]},
    "freedom": {"kind": "topic", "aliases": ["freedom"]},
    "justice": {"kind": "topic", "aliases": ["justice"]},
    "knowledge": {"kind": "topic", "aliases": ["knowledge"]},
    "wisdom": {"kind": "topic", "aliases": ["wisdom"]},
    "art": {"kind": "topic", "aliases": ["art"]},
    "music": {"kind": "topic", "aliases": ["music"]},
    "philosophy": {"kind": "topic", "aliases": ["philosophy"]},
    "religion": {"kind": "topic", "aliases": ["religion"]},
    "psychology": {"kind": "topic", "aliases": ["psychology"]},
    "sociology": {"kind": "topic", "aliases": ["sociology"]},
    "economics": {"kind": "topic", "aliases": ["economics"]},
    "biology": {"kind": "topic", "aliases": ["biology"]},
    "physics": {"kind": "topic", "aliases": ["physics"]},
    "chemistry": {"kind": "topic", "aliases": ["chemistry"]},
    "mathematics": {"kind": "topic", "aliases": ["mathematics", "maths", "math"]},
    "geography": {"kind": "topic", "aliases": ["geography"]},
    "history": {"kind": "topic", "aliases": ["history"]},
    # Iter 37 (2026-04-26): programming language + concept aliases.
    "go": {"kind": "topic", "aliases": ["go", "golang"]},
    "c++": {"kind": "topic", "aliases": ["c++", "cpp", "cplusplus"]},
    "c": {"kind": "topic", "aliases": ["c language"]},
    "java": {"kind": "topic", "aliases": ["java"]},
    "typescript": {"kind": "topic", "aliases": ["typescript", "ts"]},
    "sql": {"kind": "topic", "aliases": ["sql"]},
    "bash": {"kind": "topic", "aliases": ["bash"]},
    "shell": {"kind": "topic", "aliases": ["shell", "the shell"]},
    "functional programming": {"kind": "topic", "aliases": ["functional programming", "fp"]},
    "object oriented programming": {"kind": "topic", "aliases": ["object oriented programming", "oop"]},
    "closure": {"kind": "topic", "aliases": ["closure", "closures"]},
    "pointer": {"kind": "topic", "aliases": ["pointer", "pointers"]},
    "memory management": {"kind": "topic", "aliases": ["memory management"]},
    "async": {"kind": "topic", "aliases": ["async", "asynchronous"]},
    "concurrency": {"kind": "topic", "aliases": ["concurrency"]},
    "parallelism": {"kind": "topic", "aliases": ["parallelism"]},
    "multithreading": {"kind": "topic", "aliases": ["multithreading", "multi-threading"]},
    "thread": {"kind": "topic", "aliases": ["thread", "threads"]},
    "process": {"kind": "topic", "aliases": ["process", "processes"]},
    "algorithm": {"kind": "topic", "aliases": ["algorithm", "algorithms"]},
    "big o": {"kind": "topic", "aliases": ["big o", "big o notation"]},
    "data structure": {"kind": "topic", "aliases": ["data structure", "data structures"]},
    "hash table": {"kind": "topic", "aliases": ["hash table", "hash tables", "hashmap", "hashtable"]},
    "binary tree": {"kind": "topic", "aliases": ["binary tree", "binary trees"]},
    "graph": {"kind": "topic", "aliases": ["graph"]},
    "dynamic programming": {"kind": "topic", "aliases": ["dynamic programming", "dp"]},
    "api": {"kind": "topic", "aliases": ["api", "apis"]},
    "rest": {"kind": "topic", "aliases": ["rest", "rest api", "rest apis"]},
    "graphql": {"kind": "topic", "aliases": ["graphql"]},
    "json": {"kind": "topic", "aliases": ["json"]},
    # Iter 44 (2026-04-26): ML/training concepts that appear as
    # short-text user topics.  Without these in the catalog, capitalised-
    # token fallback marks them "unknown", and the prefer_kinds walk
    # skips them in favour of any catalog-topic word that the assistant
    # incidentally mentioned (e.g. "algorithm" from a gradient-descent
    # response hijacking "how do we prevent it" → algorithm).
    "overfitting": {"kind": "topic", "aliases": ["overfitting", "over fitting", "over-fitting"]},
    "underfitting": {"kind": "topic", "aliases": ["underfitting", "under fitting", "under-fitting"]},
    "backpropagation": {"kind": "topic", "aliases": ["backpropagation", "backprop", "back propagation", "back-propagation"]},
    "gradient descent": {"kind": "topic", "aliases": ["gradient descent", "sgd", "stochastic gradient descent"]},
    "regularisation": {"kind": "topic", "aliases": ["regularisation", "regularization", "l1 regularisation", "l2 regularisation", "weight decay"]},
    "dropout": {"kind": "topic", "aliases": ["dropout"]},
    "loss function": {"kind": "topic", "aliases": ["loss function", "loss"]},
    "activation function": {"kind": "topic", "aliases": ["activation function", "activation", "relu", "sigmoid", "tanh", "gelu"]},
    "optimizer": {"kind": "topic", "aliases": ["optimizer", "optimiser", "adam", "adamw", "rmsprop"]},
    "training data": {"kind": "topic", "aliases": ["training data", "training set"]},
    "validation": {"kind": "topic", "aliases": ["validation", "validation set", "cross validation", "cross-validation"]},
    "recursion": {"kind": "topic", "aliases": ["recursion", "recursive function"]},
    "call stack": {"kind": "topic", "aliases": ["call stack", "the call stack"]},
    # Iter 38 (2026-04-26): web/security/devops aliases.  These cover
    # the curated overlay entries added in the same iteration so single-
    # word probes ("oauth", "jwt", "ci/cd", "devops") trigger auto-anchor
    # and pivot rewrites resolve correctly.
    "https": {"kind": "topic", "aliases": ["https"]},
    "oauth": {"kind": "topic", "aliases": ["oauth", "oauth2", "oauth 2.0"]},
    "jwt": {"kind": "topic", "aliases": ["jwt", "json web token", "json web tokens"]},
    "hashing": {"kind": "topic", "aliases": ["hashing", "hash function", "hash functions"]},
    "ci/cd": {"kind": "topic", "aliases": ["ci/cd", "ci cd", "cicd"]},
    "continuous integration": {"kind": "topic", "aliases": ["continuous integration", "ci"]},
    "devops": {"kind": "topic", "aliases": ["devops", "dev ops"]},
    "microservices": {"kind": "topic", "aliases": ["microservices", "microservice", "micro services"]},
    "monolith": {"kind": "topic", "aliases": ["monolith", "monolithic", "monolithic architecture"]},
    "load balancer": {"kind": "topic", "aliases": ["load balancer", "load balancing", "load balancers"]},
    "cdn": {"kind": "topic", "aliases": ["cdn", "content delivery network"]},
    "firewall": {"kind": "topic", "aliases": ["firewall", "firewalls"]},
    "vpn": {"kind": "topic", "aliases": ["vpn", "vpns", "virtual private network"]},
    "udp": {"kind": "topic", "aliases": ["udp"]},
    "ip": {"kind": "topic", "aliases": ["ip address", "ip addresses", "ip protocol"]},
    "cookie": {"kind": "topic", "aliases": ["cookie", "cookies", "http cookie", "browser cookie"]},
    "xss": {"kind": "topic", "aliases": ["xss", "cross site scripting", "cross-site scripting"]},
    "sql injection": {"kind": "topic", "aliases": ["sql injection", "sqli"]},
    "csrf": {"kind": "topic", "aliases": ["csrf", "cross site request forgery", "cross-site request forgery"]},
    "zero trust": {"kind": "topic", "aliases": ["zero trust", "zero-trust"]},
    "vulnerability": {"kind": "topic", "aliases": ["vulnerability", "vulnerabilities", "cve", "cves"]},
    "observability": {"kind": "topic", "aliases": ["observability"]},
    "logging": {"kind": "topic", "aliases": ["logging", "logs"]},
    "metrics": {"kind": "topic", "aliases": ["metrics", "metric"]},
    "trace": {"kind": "topic", "aliases": ["trace", "traces", "distributed tracing", "distributed trace"]},
    "infrastructure as code": {"kind": "topic", "aliases": ["infrastructure as code", "iac"]},
    "terraform": {"kind": "topic", "aliases": ["terraform", "opentofu"]},
    "html": {"kind": "topic", "aliases": ["html", "html5"]},
    "css": {"kind": "topic", "aliases": ["css", "stylesheet", "stylesheets"]},
    "tls": {"kind": "topic", "aliases": ["tls", "ssl", "ssl/tls"]},
    # Iter 39 (2026-04-26): business / finance / medicine / sports /
    # geography aliases for casual-conversation depth + recruiter probes.
    "roi": {"kind": "topic", "aliases": ["roi", "return on investment"]},
    "kpi": {"kind": "topic", "aliases": ["kpi", "kpis", "key performance indicator"]},
    "marketing": {"kind": "topic", "aliases": ["marketing"]},
    "supply chain": {"kind": "topic", "aliases": ["supply chain", "supply chains"]},
    "ipo": {"kind": "topic", "aliases": ["ipo", "initial public offering"]},
    "startup": {"kind": "topic", "aliases": ["startup", "startups"]},
    "venture capital": {"kind": "topic", "aliases": ["venture capital", "vc"]},
    "product market fit": {"kind": "topic", "aliases": ["product market fit", "pmf", "product-market fit"]},
    "churn": {"kind": "topic", "aliases": ["churn"]},
    "okr": {"kind": "topic", "aliases": ["okr", "okrs", "objectives and key results"]},
    "stock": {"kind": "topic", "aliases": ["stock", "stocks"]},
    "bond": {"kind": "topic", "aliases": ["bond", "bonds"]},
    "etf": {"kind": "topic", "aliases": ["etf", "etfs", "exchange traded fund"]},
    "inflation": {"kind": "topic", "aliases": ["inflation"]},
    "recession": {"kind": "topic", "aliases": ["recession", "recessions"]},
    "gdp": {"kind": "topic", "aliases": ["gdp", "gross domestic product"]},
    "federal reserve": {"kind": "org", "aliases": ["federal reserve", "the fed", "fed", "the federal reserve"]},
    "compound interest": {"kind": "topic", "aliases": ["compound interest"]},
    "credit score": {"kind": "topic", "aliases": ["credit score", "credit scores", "fico", "fico score"]},
    "vitamin": {"kind": "topic", "aliases": ["vitamin", "vitamins"]},
    "antibiotic": {"kind": "topic", "aliases": ["antibiotic", "antibiotics"]},
    "cancer": {"kind": "topic", "aliases": ["cancer"]},
    "diabetes": {"kind": "topic", "aliases": ["diabetes"]},
    "blood pressure": {"kind": "topic", "aliases": ["blood pressure", "hypertension"]},
    "cholesterol": {"kind": "topic", "aliases": ["cholesterol"]},
    "pandemic": {"kind": "topic", "aliases": ["pandemic", "pandemics"]},
    "depression": {"kind": "topic", "aliases": ["depression"]},
    "anxiety": {"kind": "topic", "aliases": ["anxiety"]},
    "adhd": {"kind": "topic", "aliases": ["adhd", "attention deficit hyperactivity disorder"]},
    "metabolism": {"kind": "topic", "aliases": ["metabolism"]},
    "football": {"kind": "topic", "aliases": ["football", "soccer"]},
    "olympics": {"kind": "topic", "aliases": ["olympics", "the olympics", "olympic games"]},
    "chess": {"kind": "topic", "aliases": ["chess"]},
    "everest": {"kind": "topic", "aliases": ["everest", "mount everest", "chomolungma", "sagarmatha"]},
    "amazon rainforest": {"kind": "topic", "aliases": [
        "amazon rainforest", "the amazon rainforest", "amazon jungle",
        "the amazon jungle",
    ]},
    "amazon river": {"kind": "topic", "aliases": [
        "amazon river", "the amazon river",
    ]},
    "sahara": {"kind": "topic", "aliases": ["sahara", "the sahara", "sahara desert"]},
    # Tech / topics — everyday concepts ----------------------------------
    "battery": {"kind": "topic", "aliases": ["battery", "batteries"]},
    "electricity": {"kind": "topic", "aliases": ["electricity"]},
    "magnetism": {"kind": "topic", "aliases": ["magnetism"]},
    "light": {"kind": "topic", "aliases": ["light"]},
    "sound": {"kind": "topic", "aliases": ["sound"]},
    "sleep": {"kind": "topic", "aliases": ["sleep"]},
    "dreams": {"kind": "topic", "aliases": ["dreams", "dream"]},
    "exercise": {"kind": "topic", "aliases": ["exercise"]},
    "nutrition": {"kind": "topic", "aliases": ["nutrition"]},
    "immune system": {"kind": "topic", "aliases": ["immune system", "immunity"]},
    "vaccines": {"kind": "topic", "aliases": ["vaccines", "vaccine", "vaccination"]},
    # People — scientists -------------------------------------------------
    "einstein": {
        "kind": "person",
        "aliases": ["einstein", "albert einstein"],
        "gender_pronoun": "he",
    },
    "darwin": {
        "kind": "person",
        "aliases": ["darwin", "charles darwin"],
        "gender_pronoun": "he",
    },
    "newton": {
        "kind": "person",
        "aliases": ["newton", "isaac newton"],
        "gender_pronoun": "he",
    },
    "marie curie": {
        "kind": "person",
        "aliases": ["marie curie", "curie", "madame curie"],
        "gender_pronoun": "she",
    },
    "alan turing": {
        "kind": "person",
        "aliases": ["alan turing", "turing"],
        "gender_pronoun": "he",
    },
    "nikola tesla": {
        "kind": "person",
        "aliases": ["nikola tesla", "tesla the inventor"],
        "gender_pronoun": "he",
    },
    "stephen hawking": {
        "kind": "person",
        "aliases": ["stephen hawking", "hawking"],
        "gender_pronoun": "he",
    },
    # People — writers / artists -----------------------------------------
    "shakespeare": {
        "kind": "person",
        "aliases": ["shakespeare", "william shakespeare"],
        "gender_pronoun": "he",
    },
    "mozart": {
        "kind": "person",
        "aliases": ["mozart", "wolfgang mozart", "wolfgang amadeus mozart"],
        "gender_pronoun": "he",
    },
    "beethoven": {
        "kind": "person",
        "aliases": ["beethoven", "ludwig van beethoven"],
        "gender_pronoun": "he",
    },
    "leonardo da vinci": {
        "kind": "person",
        "aliases": ["leonardo da vinci", "da vinci", "leonardo"],
        "gender_pronoun": "he",
    },
    "dostoevsky": {
        "kind": "person",
        "aliases": ["dostoevsky", "fyodor dostoevsky"],
        "gender_pronoun": "he",
    },
    "tolstoy": {
        "kind": "person",
        "aliases": ["tolstoy", "leo tolstoy"],
        "gender_pronoun": "he",
    },
    # People — tech founders / leaders -----------------------------------
    "steve jobs": {
        "kind": "person",
        "aliases": ["steve jobs", "jobs"],
        "gender_pronoun": "he",
    },
    "bill gates": {
        "kind": "person",
        "aliases": ["bill gates", "gates"],
        "gender_pronoun": "he",
    },
    "elon musk": {
        "kind": "person",
        "aliases": ["elon musk", "musk"],
        "gender_pronoun": "he",
    },
    "tim berners-lee": {
        "kind": "person",
        "aliases": ["tim berners-lee", "berners-lee", "tim berners lee"],
        "gender_pronoun": "he",
    },
    "linus torvalds": {
        "kind": "person",
        "aliases": ["linus torvalds", "torvalds"],
        "gender_pronoun": "he",
    },
    "tim cook": {
        "kind": "person",
        "aliases": ["tim cook"],
        "gender_pronoun": "he",
    },
    "satya nadella": {
        "kind": "person",
        "aliases": ["satya nadella", "nadella"],
        "gender_pronoun": "he",
    },
    "sundar pichai": {
        "kind": "person",
        "aliases": ["sundar pichai", "pichai"],
        "gender_pronoun": "he",
    },
    "mark zuckerberg": {
        "kind": "person",
        "aliases": ["mark zuckerberg", "zuckerberg"],
        "gender_pronoun": "he",
    },
    "jeff bezos": {
        "kind": "person",
        "aliases": ["jeff bezos", "bezos"],
        "gender_pronoun": "he",
    },
    "ada lovelace": {
        "kind": "person",
        "aliases": ["ada lovelace", "lovelace"],
        "gender_pronoun": "she",
    },
    "grace hopper": {
        "kind": "person",
        "aliases": ["grace hopper"],
        "gender_pronoun": "she",
    },
    # Places -------------------------------------------------------------
    "berlin": {"kind": "place", "aliases": ["berlin"]},
    "paris": {"kind": "place", "aliases": ["paris"]},
    "london": {"kind": "place", "aliases": ["london"]},
    "tokyo": {"kind": "place", "aliases": ["tokyo"]},
    "shenzhen": {"kind": "place", "aliases": ["shenzhen"]},
    "new york": {"kind": "place", "aliases": ["new york", "nyc"]},
    "san francisco": {"kind": "place", "aliases": ["san francisco", "sf"]},
    "beijing": {"kind": "place", "aliases": ["beijing"]},
    "moscow": {"kind": "place", "aliases": ["moscow"]},
    "cupertino": {"kind": "place", "aliases": ["cupertino"]},
    "redmond": {"kind": "place", "aliases": ["redmond"]},
    # Countries (Phase 14, 2026-04-25) — kind="place" so the existing
    # "the country" definite-description trigger picks them up, plus
    # the bare-noun rewriter in engine.py can map "language?" /
    # "currency?" / "capital?" / "population?" follow-ups onto the
    # country fact lookup.
    "japan": {"kind": "place", "aliases": ["japan"]},
    "germany": {"kind": "place", "aliases": ["germany", "deutschland"]},
    "france": {"kind": "place", "aliases": ["france"]},
    "china": {"kind": "place", "aliases": ["china"]},
    "uk": {"kind": "place", "aliases": [
        "uk", "u.k.", "united kingdom", "britain", "great britain",
    ]},
    "usa": {"kind": "place", "aliases": [
        "usa", "u.s.a.", "united states", "america",
    ]},
    "spain": {"kind": "place", "aliases": ["spain"]},
    "italy": {"kind": "place", "aliases": ["italy"]},
    "russia": {"kind": "place", "aliases": ["russia"]},
    "brazil": {"kind": "place", "aliases": ["brazil"]},
    "india": {"kind": "place", "aliases": ["india"]},
    "australia": {"kind": "place", "aliases": ["australia"]},
    "canada": {"kind": "place", "aliases": ["canada"]},
}


# Set of country canonicals (subset of place kind) — used by the
# engine's bare-noun rewriter to detect when a "language?" /
# "currency?" / "capital?" follow-up should rewrite into a country-
# attribute query.
COUNTRY_CANONICALS: frozenset[str] = frozenset({
    "japan", "germany", "france", "china", "uk", "usa", "spain", "italy",
    "russia", "brazil", "india", "australia", "canada",
})


# ---------------------------------------------------------------------------
# Pre-computed alias lookup tables.
# ---------------------------------------------------------------------------
# We build two dicts up-front so observe() runs in O(L) over message
# length rather than O(L * |catalogue|) per turn.

_ALIAS_TO_CANONICAL: dict[str, str] = {}
_MULTIWORD_ALIASES: list[tuple[str, str]] = []  # (alias_lower, canonical)
_GENDER_PRONOUN_FOR_CANON: dict[str, str] = {}
for _canon, _spec in _ENTITY_CATALOGUE.items():
    for _alias in _spec["aliases"]:
        a = _alias.lower().strip()
        # Don't clobber an existing single-token alias for a different
        # canonical (e.g. "tesla" already maps to the org; the person
        # alias for Nikola Tesla uses "nikola tesla" / "tesla the
        # inventor" instead so we preserve the original mapping).
        if a not in _ALIAS_TO_CANONICAL:
            _ALIAS_TO_CANONICAL[a] = _canon
        if " " in a:
            _MULTIWORD_ALIASES.append((a, _canon))
    if _spec.get("kind") == "person":
        gp = _spec.get("gender_pronoun")
        if gp in {"he", "she", "they"}:
            _GENDER_PRONOUN_FOR_CANON[_canon] = gp
# Longest multi-word aliases first so ``"new york"`` isn't pre-empted
# by a future ``"new"`` alias.
_MULTIWORD_ALIASES.sort(key=lambda kv: len(kv[0]), reverse=True)


# Pronouns we attempt to resolve.  Each maps to an *ordered* tuple of
# entity kinds — earlier kinds are preferred over later ones.  This
# ordering matters: a follow-up like "who founded them?" after
# Huawei's founding location was mentioned should bind "them" to the
# ORG (Huawei) rather than the PLACE (Shenzhen) even though both are
# on the recency stack.  We don't claim this is general anaphora
# resolution; it's a deliberate heuristic that matches the dominant
# conversational pattern in the I3 demo (asking follow-ups about a
# company / topic the user has been discussing).
_PRONOUN_TARGETS: dict[str, tuple[str, ...]] = {
    "it": ("org", "topic", "place"),
    "its": ("org", "topic", "place"),
    "they": ("org", "topic", "place", "person"),
    "them": ("org", "topic", "place", "person"),
    "their": ("org", "topic", "place", "person"),
    "theirs": ("org", "topic", "place", "person"),
    "this": ("topic", "org", "place"),
    "that": ("topic", "org", "place"),
    "these": ("topic", "org", "place"),
    "those": ("topic", "org", "place"),
    "he": ("person",),
    "him": ("person",),
    "his": ("person",),
    "she": ("person",),
    "her": ("person",),
    "hers": ("person",),
}

# Gender hint for he/him/his vs she/her/hers — used by
# ``_pick_referent`` to filter ``person`` frames by their
# ``gender_pronoun`` attribute.  Plural / neutral pronouns map to
# ``None`` (no filter), and singular-they fallback is handled
# separately at the call-site.
_PRONOUN_GENDER: dict[str, str | None] = {
    "he": "he", "him": "he", "his": "he",
    "she": "she", "her": "she", "hers": "she",
    "they": None, "them": None, "their": None, "theirs": None,
    "it": None, "its": None,
    "this": None, "that": None, "these": None, "those": None,
}

# Definite-description triggers that re-anchor on the most recent
# entity of a compatible kind.  Iter 23 (2026-04-26) extended with
# more conversational definite descriptions ("this topic", "that
# topic", "the field", "the technology", "the org") so a recruiter's
# natural follow-ups bind to the active subject.
_DEFINITE_DESCRIPTIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    # Org / firm / brand
    (re.compile(r"\bthe\s+company\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+firm\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+team\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+org(?:anization|anisation)?\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+brand\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+business\b", re.I), ("org",)),
    (re.compile(r"\bthe\s+(?:vendor|maker)\b", re.I), ("org",)),
    # Place
    (re.compile(r"\bthe\s+place\b", re.I), ("place",)),
    (re.compile(r"\bthe\s+city\b", re.I), ("place",)),
    (re.compile(r"\bthe\s+country\b", re.I), ("place",)),
    (re.compile(r"\bthe\s+location\b", re.I), ("place",)),
    # Person
    (re.compile(r"\bthe\s+person\b", re.I), ("person",)),
    (re.compile(r"\bthe\s+(?:founder|ceo|leader)\b", re.I), ("person",)),
    # Topic / concept
    (re.compile(r"\bthe\s+language\b", re.I), ("topic",)),
    (re.compile(r"\bthe\s+(?:topic|subject|concept)\b", re.I), ("topic", "org")),
    (re.compile(r"\bthe\s+(?:field|area|domain)\b", re.I), ("topic",)),
    (re.compile(r"\bthe\s+(?:technology|tech|tool|technique)\b", re.I), ("topic", "org")),
    (re.compile(r"\bthe\s+(?:method|approach|algorithm|model|architecture)\b", re.I), ("topic",)),
    (re.compile(r"\bthis\s+(?:topic|thing|one|subject|stuff)\b", re.I), ("topic", "org", "person")),
    (re.compile(r"\bthat\s+(?:topic|thing|one|subject|stuff)\b", re.I), ("topic", "org", "person")),
)

# Words we should NEVER treat as named entities even if title-cased
# (sentence-initial articles, common pronouns, English filler).
_NEVER_ENTITY: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of",
    "in", "on", "at", "for", "by", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "it", "its", "this", "that", "these", "those", "i", "you",
    "we", "they", "me", "my", "your", "our", "their", "am", "he",
    "him", "his", "she", "her", "hers", "what", "who", "when", "where",
    "why", "how", "tell", "give", "show", "find", "yes", "no", "ok",
    "okay", "hi", "hello", "hey", "thanks", "thank", "please",
})


# ---------------------------------------------------------------------------
# EntityTracker
# ---------------------------------------------------------------------------

class EntityTracker:
    """Per-(user_id, session_id) entity memory + lightweight anaphora resolution.

    Bounded, per-session, in-memory.  Public methods never raise — any
    internal failure degrades to a no-op resolution so the engine's
    response path is unaffected.
    """

    def __init__(
        self,
        max_entities_per_session: int = 16,
        max_sessions: int = 1000,
    ) -> None:
        self._max_entities = max(1, int(max_entities_per_session))
        self._max_sessions = max(1, int(max_sessions))
        # Most-recently-mentioned entities first.  Each value is a list
        # of EntityFrame ordered newest first.
        self._stacks: OrderedDict[str, list[EntityFrame]] = OrderedDict()

    # ------------------------------------------------------------------
    # Session-key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id}"

    def _get_stack(self, user_id: str, session_id: str) -> list[EntityFrame]:
        if not user_id or not session_id:
            return []
        k = self._key(user_id, session_id)
        stack = self._stacks.get(k)
        if stack is None:
            return []
        # LRU touch.
        self._stacks.move_to_end(k)
        return stack

    def _ensure_stack(self, user_id: str, session_id: str) -> list[EntityFrame]:
        k = self._key(user_id, session_id)
        stack = self._stacks.get(k)
        if stack is None:
            stack = []
            self._stacks[k] = stack
            self._evict_if_over_cap()
        else:
            self._stacks.move_to_end(k)
        return stack

    def _evict_if_over_cap(self) -> None:
        while len(self._stacks) > self._max_sessions:
            self._stacks.popitem(last=False)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities(text: str) -> list[tuple[str, str, str]]:
        """Return a list of ``(surface_text, canonical, kind)`` from *text*.

        Order in the returned list is left-to-right occurrence order.
        Duplicates within the same call are collapsed to first
        occurrence.

        Iter 41 (2026-04-26): the multi-word pass previously returned
        results in alias-dict-iteration order, not text-source order,
        so e.g. ``"Apple's CEO is Tim Cook, who took over from Steve
        Jobs"`` returned ``[Steve Jobs, Tim Cook]`` whenever
        ``steve jobs`` was registered in the catalog before
        ``tim cook`` — which broke recursive person-coref ("his
        salary" resolved to the historical figure, not the slot
        answer).  Now the multi-word pass records
        ``(idx, surface, canonical, kind)`` and is sorted by idx
        before being merged with the single-token pass.
        """
        if not text:
            return []
        results: list[tuple[str, str, str]] = []
        seen_canon: set[str] = set()

        # 1) Multi-word aliases first (greedy, longest-first).
        # Collect (idx, surface, canonical, kind) so we can sort by
        # text-position before appending — alias-iteration order is
        # NOT source order.
        lowered = text.lower()
        masked = list(text)
        mw_hits: list[tuple[int, str, str, str]] = []
        # We mark consumed character ranges with NUL so the
        # single-token pass below skips them.
        for alias, canonical in _MULTIWORD_ALIASES:
            start = 0
            while True:
                idx = lowered.find(alias, start)
                if idx < 0:
                    break
                # Word-boundary check on both ends.
                left_ok = idx == 0 or not lowered[idx - 1].isalnum()
                end = idx + len(alias)
                right_ok = end >= len(lowered) or not lowered[end].isalnum()
                if left_ok and right_ok:
                    if canonical not in seen_canon:
                        seen_canon.add(canonical)
                        kind = _ENTITY_CATALOGUE[canonical]["kind"]
                        # Surface form = the original-case substring.
                        surface = text[idx:end]
                        mw_hits.append((idx, surface, canonical, kind))
                    # Mask consumed chars so the single-token pass
                    # doesn't double-count (e.g. ``"new york"`` won't
                    # then trigger a fallback ``"york"`` capture).
                    for j in range(idx, end):
                        masked[j] = "\x00"
                    start = end
                else:
                    start = idx + 1
        # Sort by text-position so the recency stack reflects the
        # actual order in which entities appear (the FIRST-mentioned
        # entity ends up at position 0 after observe()'s reversed walk).
        mw_hits.sort(key=lambda h: h[0])
        for _idx, surface, canonical, kind in mw_hits:
            results.append((surface, canonical, kind))

        masked_text = "".join(masked)

        # 2) Single-token pass against the alias dict.
        for match in re.finditer(r"[A-Za-z][A-Za-z&]*", masked_text):
            tok = match.group(0)
            tok_lower = tok.lower()
            if tok_lower in _NEVER_ENTITY:
                continue
            canon = _ALIAS_TO_CANONICAL.get(tok_lower)
            if canon is not None:
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
                kind = _ENTITY_CATALOGUE[canon]["kind"]
                results.append((tok, canon, kind))

        # 3) Capitalised-token fallback — title-cased tokens that
        #    aren't sentence-initial filler are treated as `unknown`
        #    entities.  We tag them so future ``it``/``they`` can
        #    resolve against them, even when they aren't in the
        #    curated catalogue.
        sentence_starts: set[int] = {0}
        for sep_match in re.finditer(r"[.!?]\s+", masked_text):
            sentence_starts.add(sep_match.end())
        for match in re.finditer(r"[A-Z][a-zA-Z]{2,}", masked_text):
            tok = match.group(0)
            tok_lower = tok.lower()
            if tok_lower in _NEVER_ENTITY:
                continue
            if tok_lower in _ALIAS_TO_CANONICAL:
                continue  # Already captured via the alias pass.
            # Skip tokens that are merely sentence-initial.
            if match.start() in sentence_starts and tok_lower in {"the", "i", "a"}:
                continue
            if tok_lower in seen_canon:
                continue
            seen_canon.add(tok_lower)
            results.append((tok, tok_lower, "unknown"))

        return results

    # ------------------------------------------------------------------
    # observe()
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_idx: int,
        user_text: str,
        assistant_text: str,
        priority_canonical: str | None = None,
    ) -> None:
        """Record entities mentioned in this turn.

        Pulls entities from BOTH the user message and the assistant
        response so a turn like ``"tell me about it"`` followed by
        ``"Huawei is a global..."`` correctly anchors *Huawei* on the
        recency stack even though the user never said the word.

        ``priority_canonical`` (optional): when set, the engine has
        already resolved a pronoun on this turn (the user's subject
        was inferred from prior context).  We promote that entity to
        the top of the stack AFTER assistant-side extraction, so a
        rambly assistant response can't steal anaphor priority for
        the next turn.  This is the fix for: ``"python"`` →
        ``"who created it"`` → assistant rambles about "transformer"
        → next turn ``"why is it popular"`` would otherwise resolve
        to *transformer* instead of *python*.
        """
        try:
            stack = self._ensure_stack(user_id, session_id)
            # Extraction order matters: we want the FIRST-mentioned
            # entity (typically the conversational subject) to end up
            # at position 0 (most recent / top of stack) after this
            # turn is observed.  We achieve that by walking the
            # extraction list in reverse and inserting each at
            # position 0 — so the originally-first entity goes in
            # last and ends up on top.
            #
            # Concrete example: assistant says
            #   "Huawei is a global technology company headquartered
            #    in Shenzhen."
            # Extraction yields [Huawei, Shenzhen] in source order.
            # Walking in reverse and inserting at index 0 produces
            # the stack [Huawei, Shenzhen] (Huawei on top), which
            # correctly anchors a follow-up "they" on Huawei (the
            # subject) rather than Shenzhen (the location).
            # Iter 18: tag entities by side so user-side mentions can
            # be marked as "anchors" for sticky topic resolution.
            user_extracted = self._extract_entities(user_text or "")
            assistant_extracted = self._extract_entities(assistant_text or "")
            user_canonicals = {c for _, c, _ in user_extracted}
            extracted = user_extracted + assistant_extracted
            for surface, canonical, kind in reversed(extracted):
                # If we already have this entity, refresh its turn idx
                # and bump it to the top of the stack.
                idx = next(
                    (i for i, e in enumerate(stack) if e.canonical == canonical),
                    -1,
                )
                if idx >= 0:
                    stack[idx].last_turn_idx = int(turn_idx)
                    stack[idx].text = surface
                    # Refresh gender_pronoun in case the catalogue
                    # entry was added after the original observation.
                    if kind == "person" and stack[idx].gender_pronoun is None:
                        stack[idx].gender_pronoun = (
                            _GENDER_PRONOUN_FOR_CANON.get(canonical)
                        )
                    # Iter 18: user mention re-anchors this entity.
                    if canonical in user_canonicals:
                        stack[idx].user_anchor_turn = int(turn_idx)
                        # Iter 31: first-anchor is set ONCE and never
                        # refreshed, so "back to the start" can find
                        # the original session topic.
                        if stack[idx].first_anchor_turn is None:
                            stack[idx].first_anchor_turn = int(turn_idx)
                    # Move to position 0 (most recent).
                    frame = stack.pop(idx)
                    stack.insert(0, frame)
                else:
                    gender = (
                        _GENDER_PRONOUN_FOR_CANON.get(canonical)
                        if kind == "person"
                        else None
                    )
                    is_user_anchor = canonical in user_canonicals
                    stack.insert(
                        0,
                        EntityFrame(
                            text=surface,
                            canonical=canonical,
                            kind=kind,
                            last_turn_idx=int(turn_idx),
                            gender_pronoun=gender,
                            user_anchor_turn=(
                                int(turn_idx) if is_user_anchor else None
                            ),
                            first_anchor_turn=(
                                int(turn_idx) if is_user_anchor else None
                            ),
                        ),
                    )
            # Coref-resolved priority entity: bump to top of stack
            # so a rambly assistant response can't steal the anaphor
            # priority for the next turn.
            if priority_canonical:
                idx = next(
                    (i for i, e in enumerate(stack) if e.canonical == priority_canonical),
                    -1,
                )
                if idx > 0:
                    frame = stack.pop(idx)
                    frame.last_turn_idx = int(turn_idx)
                    stack.insert(0, frame)
                elif idx == 0:
                    # Already on top — just refresh the turn idx.
                    stack[0].last_turn_idx = int(turn_idx)
            # Cap (iter 26): anchored topics get eviction immunity.
            # Without this, after a few "tell me about X" turns the
            # 16-slot stack fills with incidental places/products
            # (Cupertino, Windows, Redmond, mountain, view, ...) and
            # earlier user-anchored topics (Apple, Microsoft) get
            # evicted even though the user might pivot back to them.
            # Strategy: when the stack overflows, first keep all
            # user-anchored frames (in their current order) plus
            # enough non-anchored frames to fill the remaining slots
            # (preserving stack order).
            if len(stack) > self._max_entities:
                anchored = [f for f in stack if f.user_anchor_turn is not None]
                non_anchored = [f for f in stack if f.user_anchor_turn is None]
                # Keep all anchored (they're rare — at most 1 per turn)
                # plus the most-recent non-anchored frames to fill.
                room = max(0, self._max_entities - len(anchored))
                kept_non_anchored = non_anchored[:room]
                # Rebuild the stack preserving the original positional
                # order of the kept frames so recency is intact.
                kept_set = {id(f) for f in anchored} | {id(f) for f in kept_non_anchored}
                stack[:] = [f for f in stack if id(f) in kept_set]
                # Hard truncate as a safety net (should never fire).
                if len(stack) > self._max_entities * 2:
                    del stack[self._max_entities * 2 :]
        except Exception:  # pragma: no cover - defensive, never blocks
            pass

    # ------------------------------------------------------------------
    # resolve()
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_pronoun(text: str) -> str | None:
        """Return the first resolvable pronoun in *text*, lowercased.

        Scans left-to-right so the leading anaphor wins (matches
        natural conversation flow: ``"where are they?"`` → ``they``).

        Iter 44 (2026-04-26): skip dummy/idiomatic ``it`` — phrases
        where ``it`` has no antecedent and English requires a
        placeholder ("what time is it", "what is it like", "is it
        raining", "how is it going").  Without this, coref rewrites
        ``"what time is it"`` → ``"what time is overfitting"`` after
        an overfitting thread, then retrieval lands on the wrong
        curated entry.
        """
        if not text:
            return None
        # Reject dummy-it idioms first.
        _DUMMY_IT_PATTERNS = (
            re.compile(r"\bwhat\s+(?:time|date|day|year|month)\s+is\s+it\b", re.I),
            re.compile(r"\bhow['']?s?\s+(?:it|the\s+weather)\s+(?:going|hanging)\b", re.I),
            re.compile(r"\bwhat['']?s?\s+(?:it|the\s+weather)\s+like\b", re.I),
            re.compile(r"\bis\s+it\s+(?:raining|snowing|hot|cold|cloudy|sunny|warm|chilly|windy)\b", re.I),
            re.compile(r"\bwhat['']?s?\s+up\b", re.I),
            re.compile(r"\bhow['']?s?\s+(?:it|things)\s+(?:going|been)\b", re.I),
        )
        for pat in _DUMMY_IT_PATTERNS:
            if pat.search(text):
                return None
        # Iter 48: plural-comparison shapes — "are they X" / "do they
        # relate" / "are they connected/similar/different/linked/the
        # same" — refer to the discourse plural (both recent topics)
        # rather than a single referent.  Coref shouldn't substitute
        # one of them and lose the comparison meaning.
        _COMPARE_SHAPE_RE = re.compile(
            r"^\s*(?:are\s+(?:they|these|those)|"
            r"do\s+(?:they|these|those)|"
            r"how\s+(?:do|are)\s+(?:they|these|those))\s+"
            r"(?:related|connected|linked|similar|different|"
            r"the\s+same|alike|comparable|relate|differ|"
            r"compare|interact)\b",
            re.I,
        )
        if _COMPARE_SHAPE_RE.match(text.strip()):
            return None
        # Tokenise (apostrophes preserved so ``"it's"`` doesn't match
        # ``it`` here — that's fine, we only care about actual
        # standalone pronouns).
        for match in re.finditer(r"[A-Za-z]+", text):
            tok = match.group(0).lower()
            if tok in _PRONOUN_TARGETS:
                return tok
        return None

    @staticmethod
    def _detect_definite(text: str) -> tuple[str, tuple[str, ...]] | None:
        """Return ``(matched_text, allowed_kinds)`` if a definite
        description like ``"the company"`` is present, else ``None``.
        """
        if not text:
            return None
        for pattern, kinds in _DEFINITE_DESCRIPTIONS:
            m = pattern.search(text)
            if m:
                return m.group(0), kinds
        return None

    def _pick_referent(
        self,
        stack: list[EntityFrame],
        allowed_kinds: Iterable[str],
        turn_idx: int,
        gender: str | None = None,
    ) -> EntityFrame | None:
        """Return the most-recent entity matching any kind in *allowed_kinds*.

        Resolution order: walk the stack from most-recent to least-
        recent and return the first frame whose kind is in
        ``allowed_kinds``.  We deliberately do NOT honour the order
        within ``allowed_kinds`` — the recency stack already orders
        entities by user-mentioned-then-assistant-mentioned, so the
        topmost compatible frame is the conversational subject.

        Honouring the kind-order would mean a follow-up "it" after
        ``"chatgpt"`` (kind=topic) → ``"who made it"`` → assistant
        names ``"OpenAI"`` (kind=org) would steal the next "it" from
        chatgpt to openai, which is not what users mean.

        ``unknown``-kind entities (capitalised-token fallback) are
        accepted as a last resort.

        ``gender`` (optional, one of ``"he"`` / ``"she"`` / ``None``):
        when set, ``person``-kind frames are filtered by their
        ``gender_pronoun`` attribute.  ``None`` means no gender
        filter, used by plural/neutral pronouns and definite
        descriptions.
        """
        allowed_set = set(allowed_kinds)
        for frame in stack:
            if turn_idx - frame.last_turn_idx > 5:
                continue
            if frame.kind not in allowed_set:
                continue
            if (
                frame.kind == "person"
                and gender is not None
                and frame.gender_pronoun
                and frame.gender_pronoun != gender
            ):
                # Wrong-gender person — skip and keep walking.
                continue
            return frame
        # Fallback: accept ``unknown`` if any of org/topic/place is in
        # the allowed set (the typical demonstrative-pronoun case).
        if {"org", "topic", "place"} & allowed_set:
            for frame in stack:
                if frame.kind == "unknown" and turn_idx - frame.last_turn_idx <= 5:
                    return frame
        return None

    def resolve(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_idx: int,
        user_text: str,
    ) -> ResolutionResult:
        """Identify pronouns / referring expressions in *user_text* and
        rewrite the query using the most recent compatible entity.

        Returns a :class:`ResolutionResult`.  When no resolution is
        applicable (no pronoun detected, no compatible entity in
        scope, or any internal failure) the result has
        ``original_query == resolved_query`` with ``confidence=1.0``.
        """
        no_op = ResolutionResult(
            original_query=user_text or "",
            resolved_query=user_text or "",
            used_entity=None,
            used_pronoun=None,
            confidence=1.0,
            reasoning="No pronoun detected; query passed through unchanged.",
        )
        if not user_text:
            return no_op
        # Iter 35 (2026-04-26): comparison-shape veto.  When the user
        # types a compare-shape ("compare them" / "how do they compare"
        # / "which one is bigger"), the pronoun "them" / "they" must
        # stay literal so the compare tool's zero-arg pattern matches
        # and uses the engine-supplied fallback_pair.  Without this
        # veto, "compare them" gets rewritten to "compare microsoft"
        # by the resolver, hiding the comparison shape.
        _COMPARE_SHAPES = re.compile(
            r"^\s*(?:compare\s+(?:them|these|those|the\s+two|both)|"
            r"how\s+do\s+(?:they|these|those)\s+compare|"
            r"which\s+(?:one\s+)?(?:is\s+)?(?:better|bigger|larger|"
            r"smaller|faster|slower|cheaper|more\s+popular)\b)",
            re.I,
        )
        if _COMPARE_SHAPES.match(user_text):
            return no_op
        try:
            stack = self._get_stack(user_id, session_id)
            if not stack:
                return ResolutionResult(
                    original_query=user_text,
                    resolved_query=user_text,
                    used_entity=None,
                    used_pronoun=None,
                    confidence=1.0,
                    reasoning="No prior entity in scope; query unchanged.",
                )

            # 1) Definite descriptions ("the company", "the place").
            definite = self._detect_definite(user_text)
            if definite is not None:
                phrase, kinds = definite
                referent = self._pick_referent(stack, kinds, turn_idx)
                if referent is not None:
                    rewritten = self._substitute_phrase(
                        user_text, phrase, referent.canonical
                    )
                    age = max(0, turn_idx - referent.last_turn_idx)
                    confidence = max(0.55, 0.85 - 0.1 * age)
                    return ResolutionResult(
                        original_query=user_text,
                        resolved_query=rewritten,
                        used_entity=referent,
                        used_pronoun=phrase,
                        confidence=confidence,
                        reasoning=(
                            f"Resolved '{phrase}' to '{referent.canonical}' "
                            f"(most recent {referent.kind} mentioned "
                            f"{age} turn{'' if age == 1 else 's'} ago)."
                        ),
                    )

            # 2) Plain pronouns.
            pronoun = self._detect_pronoun(user_text)
            if pronoun is None:
                return no_op
            allowed_kinds = _PRONOUN_TARGETS[pronoun]
            gender = _PRONOUN_GENDER.get(pronoun)
            referent = self._pick_referent(
                stack, allowed_kinds, turn_idx, gender=gender,
            )
            # Singular-they fallback: if "they/them/their" found
            # nothing among org/topic/place, accept a person frame as
            # the referent (lower confidence, since true singular-they
            # is ambiguous with neutral pronouns for non-personhood).
            singular_they_fallback = False
            if referent is None and pronoun in {"they", "them", "their", "theirs"}:
                referent = self._pick_referent(
                    stack, ("person",), turn_idx, gender=None,
                )
                if referent is not None:
                    singular_they_fallback = True
            if referent is None:
                return ResolutionResult(
                    original_query=user_text,
                    resolved_query=user_text,
                    used_entity=None,
                    used_pronoun=pronoun,
                    confidence=0.0,
                    reasoning=(
                        f"Detected pronoun '{pronoun}' but no compatible "
                        f"entity ({'/'.join(allowed_kinds)}) on the recency stack."
                    ),
                )
            rewritten = self._substitute_pronoun(
                user_text, pronoun, referent
            )
            age = max(0, turn_idx - referent.last_turn_idx)
            # Confidence: closest referent with single competing
            # candidate is high; faraway referent is medium.
            same_kind_count = sum(
                1 for e in stack if e.kind == referent.kind
                and turn_idx - e.last_turn_idx <= 5
            )
            ambiguity_penalty = 0.0 if same_kind_count <= 1 else 0.1 * (same_kind_count - 1)
            if singular_they_fallback:
                # Pin the singular-they branch to ~0.65 max so the
                # confidence chip honestly reflects the uncertainty.
                confidence = max(0.50, 0.65 - 0.05 * age - ambiguity_penalty)
            elif gender is not None and referent.kind == "person":
                # Explicit gender match → high baseline.
                confidence = max(0.55, 0.85 - 0.1 * age - ambiguity_penalty)
            else:
                confidence = max(0.50, 0.95 - 0.1 * age - ambiguity_penalty)
            return ResolutionResult(
                original_query=user_text,
                resolved_query=rewritten,
                used_entity=referent,
                used_pronoun=pronoun,
                confidence=confidence,
                reasoning=(
                    f"Resolved '{pronoun}' to '{referent.canonical}' "
                    f"(most recent {referent.kind} mentioned "
                    f"{age} turn{'' if age == 1 else 's'} ago"
                    + (", singular-they fallback" if singular_they_fallback else "")
                    + ")."
                ),
            )
        except Exception:  # pragma: no cover - defensive
            return no_op

    # ------------------------------------------------------------------
    # Pronoun → entity surface substitution
    # ------------------------------------------------------------------

    @staticmethod
    def _substitute_pronoun(
        text: str, pronoun: str, entity: EntityFrame
    ) -> str:
        """Substitute the first occurrence of *pronoun* in *text* with the
        entity's canonical name, fixing up subject/verb agreement so the
        rewritten query looks natural.

        Substitutions handled:
            ``where are they located?`` → ``where is huawei located?``
            ``what do they sell?``       → ``what does huawei sell?``
            ``who founded them?``        → ``who founded huawei?``
            ``what about it?``           → ``what about huawei?``
        """
        canonical = entity.canonical
        # Verb agreement fix-ups: when the pronoun is plural
        # ('they'/'them'/'their') but we're substituting a singular
        # entity, demote the auxiliary verb.
        plural_pronouns = {"they", "them", "their", "theirs"}
        plural = pronoun in plural_pronouns

        # Replace the pronoun (case-insensitive, first occurrence,
        # whole word).  We use a function so we can preserve the
        # original capitalisation.
        pat = re.compile(rf"\b{re.escape(pronoun)}\b", re.IGNORECASE)
        rewritten, n = pat.subn(canonical, text, count=1)
        if n == 0:
            return text

        if plural:
            # ``are <entity>`` → ``is <entity>``.  Conservative:
            # only flip when the auxiliary directly precedes the
            # substituted entity name (we know its canonical form).
            rewritten = re.sub(
                rf"\bare\s+{re.escape(canonical)}\b",
                f"is {canonical}",
                rewritten,
                count=1,
                flags=re.IGNORECASE,
            )
            rewritten = re.sub(
                rf"\bdo\s+{re.escape(canonical)}\b",
                f"does {canonical}",
                rewritten,
                count=1,
                flags=re.IGNORECASE,
            )
            rewritten = re.sub(
                rf"\bwere\s+{re.escape(canonical)}\b",
                f"was {canonical}",
                rewritten,
                count=1,
                flags=re.IGNORECASE,
            )
        return rewritten

    @staticmethod
    def _substitute_phrase(text: str, phrase: str, canonical: str) -> str:
        """Replace a definite-description phrase like ``"the company"``
        with the canonical entity name."""
        pat = re.compile(re.escape(phrase), re.IGNORECASE)
        rewritten, _ = pat.subn(canonical, text, count=1)
        return rewritten

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def end_session(self, user_id: str, session_id: str) -> None:
        """Drop the per-session entity stack."""
        if not user_id or not session_id:
            return
        try:
            self._stacks.pop(self._key(user_id, session_id), None)
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Introspection (optional, used by tests / smoke)
    # ------------------------------------------------------------------

    def snapshot(self, user_id: str, session_id: str) -> list[EntityFrame]:
        """Return a shallow copy of the recency stack (newest first)."""
        return list(self._get_stack(user_id, session_id))

    def get_recent_entity(
        self,
        user_id: str,
        session_id: str,
        *,
        max_age_turns: int = 5,
        current_turn: int | None = None,
        prefer_kinds: tuple[str, ...] | None = None,
    ) -> EntityFrame | None:
        """Return the topmost (most recent) entity on the recency stack.

        This is the entry point used by the engine's *short-prompt
        topic carryover* heuristic (Fix 3 of the 2026-04-25 corpus-
        quality overhaul) — when a user types a 1–6 token follow-up
        with no resolvable pronoun (e.g. ``"most famous product"``
        right after ``"tell me about apple"``), the engine prepends
        this entity's canonical surface form to the embedding query
        so the cosine retrieval lands on apple rows instead of
        whatever short-query mean-pool noise looks like to the model.

        Args:
            user_id: Session user id.
            session_id: Session id.
            max_age_turns: Drop entries older than this many turns.
                When ``current_turn`` is None the age check is
                skipped (we just return the topmost frame, if any).
            current_turn: The current turn index, used for the age
                check.  When None, no age check is applied.
            prefer_kinds: Optional tuple of entity kinds to prefer.
                When set, the most recent frame of any preferred kind
                is returned even if a more recent frame of a different
                kind sits on top.  This is the fix for "Steve Jobs
                hijacks the Apple thread" — bare entity-slot probes
                ("ceo", "products", "headquarters") want the most
                recent ORG, not whatever PERSON the assistant
                mentioned in the founder answer.  When no preferred
                kind is found, falls back to the topmost frame.

        Returns:
            The matching :class:`EntityFrame`, or ``None`` if the
            stack is empty / no frame meets the age check.  Never
            raises.
        """
        try:
            stack = self._get_stack(user_id, session_id)
            if not stack:
                return None
            ageok = lambda f: (
                current_turn is None
                or current_turn - f.last_turn_idx <= max_age_turns
            )
            if prefer_kinds:
                # Iter 18: user-anchored topics outrank assistant-only
                # mentions inside an anchor window.  When the user
                # explicitly named a topic recently (within
                # ``ANCHOR_TURNS=4``), keep that topic active even if
                # the assistant later mentioned a different topic.
                # This kills "transformer thread → drift to neural
                # network" style failures where the assistant text
                # incidentally pushes an adjacent topic.
                ANCHOR_TURNS = 4
                for kind in prefer_kinds:
                    for f in stack:
                        if not ageok(f):
                            continue
                        if f.kind != kind:
                            continue
                        if f.user_anchor_turn is None:
                            continue
                        if (
                            current_turn is None
                            or current_turn - f.user_anchor_turn <= ANCHOR_TURNS
                        ):
                            return f
                # Fallback: walk preferred kinds in priority order —
                # return the topmost frame of the highest-priority
                # kind that exists on the stack.  This is what makes
                # "prefer ORG over PERSON" actually mean something:
                # even when a PERSON frame sits on top of the stack,
                # we walk past it to find the topmost ORG.
                for kind in prefer_kinds:
                    for f in stack:
                        if not ageok(f):
                            continue
                        if f.kind == kind:
                            return f
            top = stack[0]
            if not ageok(top):
                return None
            return top
        except Exception:  # pragma: no cover - defensive
            return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    tracker = EntityTracker()

    print("=" * 60)
    print("Smoke test: Huawei follow-up")
    print("=" * 60)

    # Turn 1
    tracker.observe(
        user_id="u",
        session_id="s",
        turn_idx=1,
        user_text="tell me about huawei",
        assistant_text="Huawei is a global technology company headquartered in Shenzhen.",
    )
    snap = tracker.snapshot("u", "s")
    print("Stack after turn 1:")
    for f in snap:
        print(f"  - {f.canonical:15s} ({f.kind}, last_turn={f.last_turn_idx})")
    assert any(f.canonical == "huawei" for f in snap), "huawei should be tracked"

    # Turn 2
    res = tracker.resolve(
        user_id="u",
        session_id="s",
        turn_idx=2,
        user_text="where are they located?",
    )
    print(f"\nTurn 2 resolution:")
    print(f"  original:  {res.original_query!r}")
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  pronoun:   {res.used_pronoun!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    print(f"  confidence:{res.confidence:.2f}")
    print(f"  reasoning: {res.reasoning}")
    assert res.used_entity is not None and res.used_entity.canonical == "huawei", \
        "Should resolve 'they' -> huawei"
    assert "huawei" in res.resolved_query.lower(), "Query should mention huawei"
    assert res.confidence >= 0.8, f"confidence too low: {res.confidence}"

    # Turn 3 — switch entity
    tracker.observe(
        user_id="u",
        session_id="s",
        turn_idx=3,
        user_text="what about apple?",
        assistant_text="Apple Inc. is a US technology company.",
    )
    snap = tracker.snapshot("u", "s")
    print(f"\nStack after turn 3 (most-recent first):")
    for f in snap:
        print(f"  - {f.canonical:15s} ({f.kind}, last_turn={f.last_turn_idx})")
    assert snap[0].canonical == "apple", "apple should be most recent"

    # Turn 4 — pronoun should bind to apple now
    res = tracker.resolve(
        user_id="u",
        session_id="s",
        turn_idx=4,
        user_text="what do they sell?",
    )
    print(f"\nTurn 4 resolution:")
    print(f"  original:  {res.original_query!r}")
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    print(f"  reasoning: {res.reasoning}")
    assert res.used_entity is not None and res.used_entity.canonical == "apple", \
        "Should resolve 'they' -> apple (most recent)"
    assert "apple" in res.resolved_query.lower(), "Query should mention apple"

    # Turn 5 — no pronoun, no resolution
    res = tracker.resolve(
        user_id="u",
        session_id="s",
        turn_idx=5,
        user_text="how is the weather?",
    )
    print(f"\nTurn 5 (no pronoun): used_entity={res.used_entity}")
    assert res.used_entity is None, "Should not resolve when no pronoun"
    assert res.original_query == res.resolved_query

    # Empty-stack negative case
    tracker2 = EntityTracker()
    res = tracker2.resolve(
        user_id="x", session_id="y", turn_idx=1,
        user_text="what do you mean?",
    )
    print(f"\nEmpty-stack: used_entity={res.used_entity}, "
          f"reasoning={res.reasoning!r}")
    assert res.used_entity is None

    # Person pronoun smoke: einstein -> "when did he live"
    print()
    print("=" * 60)
    print("Smoke test: Einstein person pronoun")
    print("=" * 60)
    tracker3 = EntityTracker()
    tracker3.observe(
        user_id="u", session_id="s", turn_idx=1,
        user_text="tell me about einstein",
        assistant_text="Albert Einstein was a German-born physicist who developed relativity.",
    )
    snap = tracker3.snapshot("u", "s")
    print("Stack after einstein turn 1:")
    for f in snap:
        print(f"  - {f.canonical:15s} ({f.kind}, gender={f.gender_pronoun})")
    res = tracker3.resolve(
        user_id="u", session_id="s", turn_idx=2,
        user_text="when did he live",
    )
    print(f"\nTurn 2 resolution ('when did he live'):")
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    print(f"  pronoun:   {res.used_pronoun!r}")
    print(f"  conf:      {res.confidence:.2f}")
    assert res.used_entity is not None and res.used_entity.canonical == "einstein", \
        "Should resolve 'he' -> einstein"
    assert "einstein" in res.resolved_query.lower()
    assert res.confidence >= 0.7, f"confidence too low: {res.confidence}"

    # 'was he German' should also resolve
    res = tracker3.resolve(
        user_id="u", session_id="s", turn_idx=3,
        user_text="was he German",
    )
    print(f"\nTurn 3 resolution ('was he German'):")
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    assert res.used_entity is not None and res.used_entity.canonical == "einstein"

    # Female pronoun smoke
    print()
    print("=" * 60)
    print("Smoke test: female-pronoun resolution")
    print("=" * 60)
    tracker4 = EntityTracker()
    tracker4.observe(
        user_id="u", session_id="s", turn_idx=1,
        user_text="tell me about marie curie",
        assistant_text="Marie Curie was a Polish-French physicist who pioneered work on radioactivity.",
    )
    res = tracker4.resolve(
        user_id="u", session_id="s", turn_idx=2,
        user_text="what did she discover",
    )
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    assert res.used_entity is not None and res.used_entity.canonical == "marie curie"

    # Topic 'it' carryover smoke
    print()
    print("=" * 60)
    print("Smoke test: photosynthesis topic carryover")
    print("=" * 60)
    tracker5 = EntityTracker()
    tracker5.observe(
        user_id="u", session_id="s", turn_idx=1,
        user_text="what is photosynthesis",
        assistant_text="Photosynthesis is how plants convert sunlight into glucose.",
    )
    snap = tracker5.snapshot("u", "s")
    print("Stack:", [(f.canonical, f.kind) for f in snap])
    res = tracker5.resolve(
        user_id="u", session_id="s", turn_idx=2,
        user_text="why is it important",
    )
    print(f"  resolved:  {res.resolved_query!r}")
    print(f"  entity:    {res.used_entity.canonical if res.used_entity else None!r}")
    assert res.used_entity is not None and res.used_entity.canonical == "photosynthesis"

    print("\nAll smoke tests passed.")

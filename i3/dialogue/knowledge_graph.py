"""In-memory curated knowledge graph for the I3 demo.

This is a lightweight relational store over a hand-curated catalogue
of ~200 (subject, predicate, object) triples covering the top ~30
entities the demo discusses (orgs, places, people, concepts).

The graph powers two new tool routes added in the 2026-04-25
"final smartness" upgrade:

1. **Entity-attribute lookup** — replaces the flat ``_ENTITY_KNOWLEDGE``
   dict in :mod:`i3.slm.retrieval` with a richer set of slots
   (``founded_by``, ``founded_in``, ``competitor_of``, ``acquired``,
   ``owns``, ``famous_for`` and so on).  The retriever's
   ``tool: entity`` route consults this graph first and falls back to
   the original flat dict only if the graph has no matching triple.

2. **Composed answers** — an open question like
   "what does microsoft own?" or "who are apple's competitors?" is
   composed from all triples sharing a subject + predicate; the
   pipeline tags this as ``tool: graph_compose``.

The module is **pure stdlib** (no NetworkX / Neo4j) so it stays inside
the from-scratch I3 stack constraints, and it is **graceful** — if
the catalogue file is missing or malformed, every public method
returns the empty result and the engine falls back to the legacy
behaviour.

Bounds:
    - At most ~500 triples after deduping (the catalogue is small by
      design).
    - ``find_path`` is bounded at ``max_depth`` (default 3) to keep
      the BFS finite even on cyclic graphs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KGRelation:
    """Single ``(subject, predicate, object, confidence)`` triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    year: int | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
        }
        if self.year is not None:
            d["year"] = self.year
        return d


# ---------------------------------------------------------------------------
# Phrase resolver — maps user query → (subject, predicate)
# ---------------------------------------------------------------------------
# Keys are the canonical predicate names; values are the regex patterns
# we expect a question to match.  Each regex must capture the entity
# surface form in group 1.
_PHRASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # founded_by / created_by
    ("founded_by", re.compile(r"^who\s+(?:founded|started|created|invented)\s+(.+?)\s*\??\s*$", re.I)),
    ("founded_by", re.compile(r"^(.+?)\s+founders?\s*\??\s*$", re.I)),
    # founded_in
    ("founded_in", re.compile(r"^when\s+was\s+(.+?)\s+(?:founded|started|created|established)\s*\??\s*$", re.I)),
    ("founded_in", re.compile(r"^when\s+(?:did|was)\s+(.+?)\s+(?:start|begin|established|founded)\s*\??\s*$", re.I)),
    ("founded_in", re.compile(r"^when\s+did\s+(.+?)\s+start\s*\??\s*$", re.I)),
    # headquartered_in
    ("headquartered_in", re.compile(r"^where\s+(?:is|are)\s+(.+?)\s+(?:located|based|headquartered)\s*\??\s*$", re.I)),
    ("headquartered_in", re.compile(r"^where\s+(?:is|are)\s+(.+?)\s*\??\s*$", re.I)),
    # ceo
    ("ceo", re.compile(r"^who\s+(?:runs|leads|heads)\s+(.+?)\s*\??\s*$", re.I)),
    ("ceo", re.compile(r"^who\s+is\s+the\s+ceo\s+of\s+(.+?)\s*\??\s*$", re.I)),
    # famous_for
    ("famous_for", re.compile(r"^what\s+(?:is|are)\s+(.+?)\s+famous\s+for\s*\??\s*$", re.I)),
    ("famous_for", re.compile(r"^what\s+(?:is|are)\s+(.+?)\s+known\s+for\s*\??\s*$", re.I)),
    # competitor_of
    ("competitor_of", re.compile(r"^who\s+are\s+(.+?)['’]?s?\s+competitors?\s*\??\s*$", re.I)),
    ("competitor_of", re.compile(r"^(.+?)['’]?s?\s+competitors?\s*\??\s*$", re.I)),
    ("competitor_of", re.compile(r"^who\s+competes?\s+with\s+(.+?)\s*\??\s*$", re.I)),
    # discovered_by / proposed
    ("discovered_by", re.compile(r"^who\s+(?:proposed|discovered)\s+(.+?)\s*\??\s*$", re.I)),
    # fell_in (events)
    ("fell_in", re.compile(r"^when\s+did\s+(.+?)\s+fall\s*\??\s*$", re.I)),
    ("ended_in", re.compile(r"^when\s+did\s+(.+?)\s+end\s*\??\s*$", re.I)),
    ("won_by", re.compile(r"^who\s+won\s+(.+?)\s*\??\s*$", re.I)),
    # acquired / owns
    ("acquired", re.compile(r"^what\s+(?:has|did)\s+(.+?)\s+(?:bought|acquired|purchased)\s*\??\s*$", re.I)),
    ("acquired", re.compile(r"^what\s+did\s+(.+?)\s+buy\s*\??\s*$", re.I)),
    ("owns", re.compile(r"^what\s+does\s+(.+?)\s+own\s*\??\s*$", re.I)),
    # discovered / wrote / born / died
    ("discovered", re.compile(r"^what\s+did\s+(.+?)\s+discover\s*\??\s*$", re.I)),
    ("wrote", re.compile(r"^what\s+did\s+(.+?)\s+write\s*\??\s*$", re.I)),
    ("born_in", re.compile(r"^where\s+was\s+(.+?)\s+born\s*\??\s*$", re.I)),
    ("born_year", re.compile(r"^when\s+was\s+(.+?)\s+born\s*\??\s*$", re.I)),
    ("died_in", re.compile(r"^where\s+did\s+(.+?)\s+die\s*\??\s*$", re.I)),
    ("died_year", re.compile(r"^when\s+did\s+(.+?)\s+die\s*\??\s*$", re.I)),
)


# Aliases — surface form → canonical entity key in the catalogue.
# Mirrors :data:`i3.slm.retrieval._ENTITY_TOOL_ALIASES` so a rewrite
# that lands in either place finds the same entity.
_ENTITY_ALIASES: dict[str, str] = {
    # orgs
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
    # tech
    "github": "github",
    "youtube": "youtube",
    "android": "android",
    "iphone": "iphone",
    "windows": "windows",
    "linux": "linux",
    "python": "python",
    "rust": "rust",
    # people
    "einstein": "einstein", "albert einstein": "einstein",
    "newton": "newton", "isaac newton": "newton",
    "darwin": "darwin", "charles darwin": "darwin",
    # concepts / events
    "the roman empire": "roman empire", "roman empire": "roman empire",
    "rome": "roman empire",
    "world war 2": "world war 2", "wwii": "world war 2",
    "world war ii": "world war 2", "ww2": "world war 2",
    "evolution": "evolution",
    "photosynthesis": "photosynthesis",
    "quantum mechanics": "quantum mechanics", "quantum physics": "quantum mechanics",
    "relativity": "relativity", "general relativity": "relativity",
    "special relativity": "relativity",
}


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    """Tiny in-memory relational graph backed by a curated JSON file."""

    def __init__(self, catalogue_path: Path | str | None = None):
        self._triples: list[KGRelation] = []
        self._by_subject: dict[str, list[KGRelation]] = {}
        self._loaded = False
        self._path: Path | None = None
        if catalogue_path is None:
            # Default: data/knowledge_graph.json next to repo root.
            here = Path(__file__).resolve().parent.parent.parent
            catalogue_path = here / "data" / "knowledge_graph.json"
        self._path = Path(catalogue_path)
        try:
            self._load(self._path)
        except Exception:
            logger.exception(
                "KnowledgeGraph: failed to load catalogue from %s; "
                "graph will return empty results.",
                self._path,
            )

    # -- loading ---------------------------------------------------------
    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("KnowledgeGraph: catalogue %s not found", path)
            return
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        rels = data.get("relations") or []
        seen: set[tuple[str, str, str]] = set()
        for r in rels:
            try:
                s = str(r.get("s", "")).strip().lower()
                p = str(r.get("p", "")).strip().lower()
                o = str(r.get("o", "")).strip()
                c = float(r.get("c", 1.0))
                y = r.get("y")
                if not s or not p or not o:
                    continue
                key = (s, p, o.lower())
                if key in seen:
                    continue
                seen.add(key)
                rel = KGRelation(s, p, o, c, int(y) if y is not None else None)
                self._triples.append(rel)
                self._by_subject.setdefault(s, []).append(rel)
            except Exception:
                logger.debug("KG: skipping malformed triple %r", r)
                continue
        self._loaded = True
        logger.info(
            "KnowledgeGraph loaded %d triples across %d subjects",
            len(self._triples),
            len(self._by_subject),
        )

    # -- queries ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded and bool(self._triples)

    def get_facts(self, subject: str) -> list[KGRelation]:
        """All triples where *subject* is the subject, sorted by confidence."""
        if not subject:
            return []
        canonical = self._canonical(subject)
        if canonical is None:
            return []
        return sorted(
            self._by_subject.get(canonical, []),
            key=lambda r: r.confidence,
            reverse=True,
        )

    def get_related(
        self, subject: str, predicate: str | None = None
    ) -> list[KGRelation]:
        """All triples for *subject*, optionally filtered by predicate."""
        facts = self.get_facts(subject)
        if predicate:
            pred = predicate.lower()
            return [r for r in facts if r.predicate == pred]
        return facts

    def find_path(
        self, subject_a: str, subject_b: str, max_depth: int = 3
    ) -> list[KGRelation]:
        """Return a short relation path between two subjects, or [].

        Bounded BFS over predicates whose object also names a subject in
        the graph.  Returns the first path found within *max_depth* hops;
        empty list when no path exists.
        """
        a = self._canonical(subject_a)
        b = self._canonical(subject_b)
        if a is None or b is None:
            return []
        if a == b:
            return []
        # BFS
        from collections import deque
        seen: set[str] = {a}
        queue: deque[tuple[str, list[KGRelation]]] = deque()
        queue.append((a, []))
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for rel in self._by_subject.get(node, []):
                # Try to canonicalise the object as a subject.
                obj_lower = rel.object.lower()
                # take first word to handle "github (2018)" etc.
                obj_head = re.split(r"[\s,(]", obj_lower, maxsplit=1)[0]
                cand = (
                    _ENTITY_ALIASES.get(obj_lower)
                    or _ENTITY_ALIASES.get(obj_head)
                )
                if cand and cand == b:
                    return path + [rel]
                if cand and cand not in seen:
                    seen.add(cand)
                    queue.append((cand, path + [rel]))
        return []

    def resolve_phrase(
        self, text: str, hint_entity: str | None = None
    ) -> tuple[str, str] | None:
        """Try to extract ``(canonical_entity, predicate)`` from *text*.

        Returns ``None`` when no pattern matches.  *hint_entity* is the
        most-recent entity from the EntityTracker (used when the user
        types a bare follow-up like "and the founder?" without naming
        the entity).
        """
        if not text:
            return None
        cleaned = re.sub(r"\s+", " ", text.strip())
        # Strip trailing punctuation
        cleaned = cleaned.rstrip(".!?,")
        # Strip lead-ins like "now back to apple — " or "and " so the
        # downstream patterns see a clean question.  Keep a copy of
        # the unstripped form so we can fall back if both fail.
        original = cleaned
        cleaned = re.sub(
            r"^(?:now\s+(?:back\s+to|about)\s+\S+\s*[—\-,:;]+\s*)",
            "", cleaned, flags=re.I,
        )
        cleaned = re.sub(r"^(?:and|but|or|so|ok|okay)\s+", "", cleaned, flags=re.I)
        for pred, pat in _PHRASE_PATTERNS:
            m = pat.match(cleaned)
            if not m:
                continue
            try:
                surface = m.group(1)
            except IndexError:
                continue
            if not surface:
                continue
            surface_lower = surface.strip().lower()
            surface_lower = re.sub(r"^(?:the|a|an)\s+", "", surface_lower)
            canonical = _ENTITY_ALIASES.get(surface_lower)
            if canonical and canonical in self._by_subject:
                return canonical, pred
        # Inline-entity scan: when the cleaned text looks like a
        # question shape ("who founded it?", "where is it?", ...) but
        # the entity is "it"/"them" and the original phrasing has a
        # known entity name embedded in it (e.g. "now back to apple —
        # who founded it?"), pick up that entity.
        if hint_entity is None:
            # First check the explicit "back to X" / "about X" / "X —"
            # lead-in phrasings — these pin the entity unambiguously
            # over a stray mention later in the sentence.
            lead_m = re.match(
                r"^(?:now\s+back\s+to|now\s+about|back\s+to|about|talk\s+about)\s+([a-z][a-z\s]{1,30}?)\s*(?:[—\-,:;]|—|$)",
                original, re.I,
            )
            if lead_m:
                surface = lead_m.group(1).strip().lower()
                if surface in _ENTITY_ALIASES:
                    hint_entity = _ENTITY_ALIASES[surface]
            if hint_entity is None:
                # Fallback: longest-alias-first scan so "the roman
                # empire" beats "rome" and multi-word entities aren't
                # shadowed by short aliases.
                for alias, canon in sorted(
                    _ENTITY_ALIASES.items(), key=lambda kv: -len(kv[0])
                ):
                    if re.search(rf"\b{re.escape(alias)}\b", original, re.I):
                        hint_entity = canon
                        break
        if hint_entity:
            # Re-run the phrase patterns but allow "it"/"them" as the entity.
            it_text = re.sub(
                r"\b(?:it|them|they)\b", hint_entity, cleaned, flags=re.I,
            )
            for pred, pat in _PHRASE_PATTERNS:
                m = pat.match(it_text)
                if not m:
                    continue
                try:
                    surface = m.group(1)
                except IndexError:
                    continue
                if surface and _ENTITY_ALIASES.get(surface.strip().lower()) == hint_entity:
                    return hint_entity, pred
        # Bare-hint fallback: short follow-ups like "competitors?" or
        # "founder?" with a hint entity from the tracker.
        if hint_entity:
            hint_canonical = _ENTITY_ALIASES.get(hint_entity.lower(), hint_entity.lower())
            if hint_canonical in self._by_subject:
                lower = cleaned.lower()
                bare_predicates = {
                    "founder": "founded_by",
                    "founders": "founded_by",
                    "ceo": "ceo",
                    "competitors": "competitor_of",
                    "location": "headquartered_in",
                    "headquarters": "headquartered_in",
                    "owners": "owned_by",
                    "products": "famous_for",
                }
                for k, p in bare_predicates.items():
                    if lower.startswith(k) or lower == k or lower == k + "?":
                        return hint_canonical, p
        return None

    # -- composed answers -----------------------------------------------
    def compose_answer(
        self, subject: str, predicate: str
    ) -> str:
        """Build a short, well-formed paragraph from the triples for
        ``(subject, predicate)``.  Returns an empty string when nothing
        matches.
        """
        # Predicate aliasing: a question like "who discovered evolution?"
        # resolves to predicate=discovered_by, but the catalogue stores
        # the same fact as "founded_by" (since evolution-as-theory was
        # proposed/founded by Darwin).  Try the requested predicate
        # first, then a small alias chain.
        _PRED_ALIASES: dict[str, tuple[str, ...]] = {
            "discovered_by": ("discovered_by", "founded_by", "created_by"),
            "founded_by": ("founded_by", "discovered_by", "created_by"),
            "created_by": ("created_by", "founded_by"),
            "founded_in": ("founded_in", "released_in"),
            "won_by": ("won_by", "ended_in"),
            "owns": ("owns", "acquired"),
            "acquired": ("acquired", "owns"),
        }
        rels = self.get_related(subject, predicate)
        if not rels:
            for alt in _PRED_ALIASES.get(predicate, ()):
                rels = self.get_related(subject, alt)
                if rels:
                    predicate = alt  # display under the alias slot
                    break
        if not rels:
            return ""
        # Stable display name from the catalogue's first triple.
        display = self._display_name(subject)
        objs = [r.object for r in rels]
        if predicate == "founded_by":
            if len(objs) == 1:
                year = rels[0].year
                yr = f" in {year}" if year else ""
                return f"{display} was founded by {objs[0]}{yr}."
            year = next((r.year for r in rels if r.year), None)
            yr = f" in {year}" if year else ""
            return f"{display} was founded by {self._and_join(objs)}{yr}."
        if predicate == "founded_in":
            return f"{display} was founded in {objs[0]}."
        if predicate == "headquartered_in":
            return f"{display} is headquartered in {objs[0]}."
        if predicate == "ceo":
            return f"{display}'s CEO is {objs[0]}."
        if predicate == "famous_for":
            return f"{display} is famous for {objs[0]}."
        if predicate == "competitor_of":
            return (
                f"{display} competes with {self._and_join(objs)}."
            )
        if predicate == "acquired":
            return f"{display} has acquired {self._and_join(objs)}."
        if predicate == "owns":
            return f"{display} owns {self._and_join(objs)}."
        if predicate == "discovered":
            return f"{display} discovered {self._and_join(objs)}."
        if predicate == "discovered_by":
            return f"{display} was proposed by {self._and_join(objs)}."
        if predicate == "created_by":
            return f"{display} was created by {self._and_join(objs)}."
        if predicate == "wrote":
            return f"{display} wrote {self._and_join(objs)}."
        if predicate == "born_in":
            return f"{display} was born in {objs[0]}."
        if predicate == "born_year":
            return f"{display} was born in {objs[0]}."
        if predicate == "died_in":
            return f"{display} died in {objs[0]}."
        if predicate == "died_year":
            return f"{display} died in {objs[0]}."
        if predicate == "owned_by":
            return f"{display} is owned by {objs[0]}."
        if predicate == "fell_in":
            return f"{display} fell in {objs[0]}."
        if predicate == "ended_in":
            return f"{display} ended in {objs[0]}."
        if predicate == "won_by":
            return f"{display} was won by {objs[0]}."
        # Generic fallback
        return f"{display} {predicate.replace('_', ' ')}: {self._and_join(objs)}."

    def overview(self, subject: str) -> str:
        """Compose a 2-3 sentence prominent-triples overview.

        Iter 51 (2026-04-27): added a pairwise-overlap dedupe pass so
        a ``founded_by`` sentence that already names the year (e.g.
        "Python was founded by Guido van Rossum in 1991.") doesn't get
        followed by a redundant ``founded_in`` sentence ("Python was
        founded in 1991.").  Detection: if a candidate sentence shares
        ≥ 60 % of its content tokens with an already-selected sentence,
        skip it.  Cheaper and more robust than slot-pair-specific
        rules and handles future predicate additions automatically.
        """
        rels = self.get_facts(subject)
        if not rels:
            return ""
        display = self._display_name(subject)
        # Pick the most-prominent slots in priority order.
        slots = ("famous_for", "headquartered_in", "founded_by", "founded_in",
                 "ceo", "competitor_of")
        sentences: list[str] = []
        seen_preds: set[str] = set()
        # Pre-tokenise selected sentences for the overlap dedupe.
        _STOP = {"is", "the", "a", "an", "and", "of", "in", "by", "to",
                 "for", "with", "on", "was", "are", "be", "as", "at",
                 "from", "or", "it", "its"}

        def _content_tokens(s: str) -> set[str]:
            import re as _re
            toks = _re.findall(r"[a-zA-Z][a-zA-Z]+|\d+", s.lower())
            return {t for t in toks if t not in _STOP and len(t) > 1}

        for slot in slots:
            if slot in seen_preds:
                continue
            sent = self.compose_answer(subject, slot)
            if not sent:
                continue
            new_tokens = _content_tokens(sent)
            # Skip if a selected sentence already covers >= 60% of these
            # content tokens (year-mention overlap is the typical case).
            redundant = False
            for prev in sentences:
                prev_tokens = _content_tokens(prev)
                if not new_tokens:
                    break
                overlap = new_tokens & prev_tokens
                if len(overlap) / max(len(new_tokens), 1) >= 0.6:
                    redundant = True
                    break
            if redundant:
                seen_preds.add(slot)
                continue
            sentences.append(sent)
            seen_preds.add(slot)
            if len(sentences) >= 3:
                break
        if not sentences:
            return ""
        return " ".join(sentences)

    # -- helpers ---------------------------------------------------------
    def _canonical(self, surface: str | None) -> str | None:
        if not surface:
            return None
        s = surface.strip().lower()
        s = re.sub(r"^(?:the|a|an)\s+", "", s)
        s = s.rstrip(".!?,")
        if s in self._by_subject:
            return s
        return _ENTITY_ALIASES.get(s)

    # Special-cased display names where stock title-casing is wrong
    # (acronyms, proper nouns with non-standard capitalisation).
    _DISPLAY_OVERRIDES: dict[str, str] = {
        "huawei": "Huawei",
        "apple": "Apple",
        "microsoft": "Microsoft",
        "google": "Google",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "meta": "Meta",
        "amazon": "Amazon",
        "ibm": "IBM",
        "samsung": "Samsung",
        "tesla": "Tesla",
        "nvidia": "NVIDIA",
        "intel": "Intel",
        "sony": "Sony",
        "github": "GitHub",
        "youtube": "YouTube",
        "android": "Android",
        "iphone": "iPhone",
        "windows": "Windows",
        "linux": "Linux",
        "python": "Python",
        "rust": "Rust",
        "einstein": "Einstein",
        "newton": "Newton",
        "darwin": "Darwin",
        "roman empire": "The Roman Empire",
        "world war 2": "World War 2",
        "evolution": "Evolution",
        "photosynthesis": "Photosynthesis",
        "quantum mechanics": "Quantum mechanics",
        "relativity": "Relativity",
    }

    def _display_name(self, subject: str) -> str:
        """Title-case a subject for display, with hand-curated overrides."""
        canonical = self._canonical(subject) or subject
        return self._DISPLAY_OVERRIDES.get(canonical, canonical.title())

    @staticmethod
    def _and_join(items: Iterable[str]) -> str:
        items = list(items)
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"


# ---------------------------------------------------------------------------
# Module-level singleton — graceful fallback if the catalogue is missing
# ---------------------------------------------------------------------------
_GLOBAL_KG: KnowledgeGraph | None = None


def get_global_kg() -> KnowledgeGraph:
    """Return a process-wide :class:`KnowledgeGraph` (lazy init)."""
    global _GLOBAL_KG
    if _GLOBAL_KG is None:
        _GLOBAL_KG = KnowledgeGraph()
    return _GLOBAL_KG

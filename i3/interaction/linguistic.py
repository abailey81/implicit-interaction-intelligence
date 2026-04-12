"""From-scratch linguistic analysis utilities for I3.

Every method is implemented without external NLP libraries (no NLTK, no spaCy,
no TextBlob).  The module provides type-token ratio, readability metrics,
formality scoring, emoji counting, and a comprehensive lexicon-based sentiment
analyser with negation handling.

Typical usage::

    analyser = LinguisticAnalyzer()
    features = analyser.compute_all("I really love this new feature!")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ====================================================================
# Constants
# ====================================================================

# --- Contractions (~35) ------------------------------------------------
CONTRACTIONS: set[str] = {
    "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "here's",
    "i'd", "i'll", "i'm", "i've", "isn't", "it'd", "it'll", "it's",
    "let's", "mightn't", "mustn't", "shan't", "she'd", "she'll", "she's",
    "shouldn't", "that's", "there's", "they'd", "they'll", "they're",
    "they've", "wasn't", "we'd", "we'll", "we're", "we've", "weren't",
    "what's", "where's", "who'd", "who'll", "who's", "won't", "wouldn't",
    "you'd", "you'll", "you're", "you've",
}

# --- Slang markers (~45) -----------------------------------------------
SLANG_MARKERS: set[str] = {
    "lol", "lmao", "rofl", "omg", "brb", "tbh", "ngl", "imo", "imho",
    "smh", "fwiw", "afaik", "idk", "ikr", "ftw", "yolo", "fomo", "goat",
    "tl;dr", "tldr", "rn", "nvm", "ty", "thx", "pls", "plz", "bc",
    "cuz", "cos", "gonna", "gotta", "wanna", "kinda", "sorta", "dunno",
    "ya", "yep", "nah", "nope", "sup", "yo", "dude", "bro", "sis",
    "lit", "slay", "vibe", "vibes", "lowkey", "highkey", "fr", "ong",
    "srsly", "rly", "tho", "tho'",
}

# --- Abbreviations that end with a period (for sentence splitting) ------
_ABBREVIATIONS: set[str] = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "ave.",
    "blvd.", "dept.", "est.", "vol.", "vs.", "etc.", "approx.", "govt.",
    "inc.", "ltd.", "corp.", "co.", "no.", "gen.", "sgt.", "pvt.",
    "e.g.", "i.e.", "a.m.", "p.m.", "u.s.", "u.k.", "u.n.",
}

# --- Positive sentiment lexicon (~200 words) ----------------------------
POSITIVE_LEXICON: dict[str, float] = {
    # Strong positive (0.8 - 1.0)
    "love": 0.90, "amazing": 0.90, "wonderful": 0.85, "fantastic": 0.90,
    "excellent": 0.90, "brilliant": 0.88, "outstanding": 0.92, "perfect": 0.95,
    "incredible": 0.88, "magnificent": 0.90, "superb": 0.88, "exceptional": 0.90,
    "phenomenal": 0.90, "marvelous": 0.85, "spectacular": 0.88, "terrific": 0.85,
    "awesome": 0.85, "fabulous": 0.85, "remarkable": 0.82, "sublime": 0.88,
    "delightful": 0.85, "exquisite": 0.85, "glorious": 0.82, "blissful": 0.88,
    "ecstatic": 0.90, "thrilled": 0.85, "overjoyed": 0.90, "elated": 0.88,
    "euphoric": 0.90, "triumphant": 0.82, "magnificent": 0.85, "divine": 0.82,

    # Moderate positive (0.5 - 0.79)
    "good": 0.60, "great": 0.70, "nice": 0.55, "fine": 0.50, "happy": 0.75,
    "glad": 0.65, "pleased": 0.65, "enjoy": 0.70, "enjoyable": 0.70,
    "lovely": 0.75, "beautiful": 0.78, "pretty": 0.55, "cool": 0.60,
    "fun": 0.65, "exciting": 0.72, "interesting": 0.60, "impressive": 0.72,
    "helpful": 0.70, "useful": 0.65, "valuable": 0.68, "worthy": 0.60,
    "positive": 0.60, "favorable": 0.62, "pleasant": 0.65, "satisfying": 0.68,
    "comfortable": 0.60, "confident": 0.65, "grateful": 0.72, "thankful": 0.70,
    "appreciate": 0.72, "admire": 0.70, "respect": 0.65, "trust": 0.68,
    "hope": 0.60, "hopeful": 0.65, "optimistic": 0.68, "inspired": 0.72,
    "inspiring": 0.72, "motivating": 0.68, "encouraging": 0.68, "uplifting": 0.72,
    "warm": 0.58, "friendly": 0.65, "kind": 0.68, "generous": 0.70,
    "caring": 0.70, "compassionate": 0.72, "gentle": 0.60, "tender": 0.62,
    "sweet": 0.65, "charming": 0.65, "graceful": 0.65, "elegant": 0.62,
    "smart": 0.65, "clever": 0.62, "wise": 0.68, "brilliant": 0.75,
    "talented": 0.70, "skilled": 0.62, "capable": 0.58, "competent": 0.55,
    "efficient": 0.60, "effective": 0.62, "productive": 0.60, "successful": 0.72,
    "accomplished": 0.68, "achieved": 0.65, "victory": 0.72, "winning": 0.68,

    # Mild positive (0.3 - 0.49)
    "okay": 0.35, "ok": 0.35, "alright": 0.38, "decent": 0.42,
    "fair": 0.40, "adequate": 0.38, "acceptable": 0.40, "reasonable": 0.42,
    "solid": 0.48, "steady": 0.42, "stable": 0.42, "reliable": 0.50,
    "consistent": 0.45, "clear": 0.48, "clean": 0.45, "smooth": 0.50,
    "easy": 0.48, "simple": 0.42, "straightforward": 0.48, "intuitive": 0.55,
    "responsive": 0.55, "fast": 0.50, "quick": 0.48, "prompt": 0.45,
    "ready": 0.40, "prepared": 0.42, "willing": 0.45, "eager": 0.55,
    "keen": 0.50, "enthusiastic": 0.65, "passionate": 0.70, "devoted": 0.65,
    "loyal": 0.62, "faithful": 0.60, "honest": 0.60, "sincere": 0.62,
    "genuine": 0.60, "authentic": 0.58, "natural": 0.45, "organic": 0.42,

    # Conversational AI interaction positive
    "understand": 0.55, "understood": 0.55, "helpful": 0.70, "thanks": 0.65,
    "thank": 0.65, "assist": 0.55, "support": 0.58, "solve": 0.60,
    "resolved": 0.65, "fixed": 0.60, "works": 0.55, "working": 0.50,
    "correct": 0.58, "accurate": 0.60, "precise": 0.58, "relevant": 0.55,
    "insightful": 0.72, "informative": 0.65, "educational": 0.60,
    "enlightening": 0.70, "empowering": 0.68, "transformative": 0.72,
    "innovative": 0.68, "creative": 0.65, "imaginative": 0.65,
    "thoughtful": 0.68, "considerate": 0.65, "attentive": 0.62,
    "responsive": 0.60, "proactive": 0.58, "initiative": 0.55,
    "progress": 0.55, "improvement": 0.58, "growth": 0.55, "learn": 0.50,
    "learned": 0.55, "discover": 0.58, "discovered": 0.60,
    "breakthrough": 0.75, "solution": 0.60, "answered": 0.58,
    "clarity": 0.62, "coherent": 0.55, "logical": 0.52,
    "recommend": 0.58, "endorsed": 0.60, "approved": 0.58,
    "celebrate": 0.72, "congrats": 0.70, "congratulations": 0.72,
    "bravo": 0.70, "kudos": 0.68, "cheers": 0.60, "hooray": 0.72,
    "wow": 0.65, "yay": 0.70, "hurrah": 0.72,
    "peaceful": 0.65, "calm": 0.55, "serene": 0.65, "tranquil": 0.62,
    "refreshing": 0.60, "rejuvenating": 0.62, "invigorating": 0.65,
    "vibrant": 0.62, "lively": 0.60, "energetic": 0.62,
    "prosper": 0.65, "thrive": 0.68, "flourish": 0.70,
    "reward": 0.60, "rewarding": 0.68, "fulfilling": 0.72,
    "meaningful": 0.65, "purposeful": 0.62, "worthwhile": 0.65,
}

# --- Negative sentiment lexicon (~200 words) ----------------------------
NEGATIVE_LEXICON: dict[str, float] = {
    # Strong negative (-0.8 to -1.0)
    "hate": -0.90, "terrible": -0.90, "horrible": -0.88, "awful": -0.85,
    "dreadful": -0.85, "atrocious": -0.90, "abysmal": -0.92, "appalling": -0.88,
    "disgusting": -0.88, "revolting": -0.85, "repulsive": -0.85, "vile": -0.88,
    "wretched": -0.82, "miserable": -0.85, "devastating": -0.88, "catastrophic": -0.90,
    "disastrous": -0.88, "ruinous": -0.82, "destructive": -0.80, "horrendous": -0.88,
    "abominable": -0.85, "despicable": -0.88, "loathsome": -0.85, "detestable": -0.85,
    "outrageous": -0.80, "unbearable": -0.82, "intolerable": -0.82, "excruciating": -0.85,
    "agonizing": -0.82, "torturous": -0.82, "nightmarish": -0.85, "hellish": -0.82,

    # Moderate negative (-0.5 to -0.79)
    "bad": -0.60, "poor": -0.58, "wrong": -0.62, "ugly": -0.65,
    "sad": -0.70, "unhappy": -0.68, "upset": -0.65, "angry": -0.72,
    "furious": -0.78, "annoyed": -0.62, "irritated": -0.60, "frustrated": -0.68,
    "disappointing": -0.65, "disappointed": -0.68, "dissatisfied": -0.62,
    "unsatisfied": -0.58, "displeased": -0.60, "unhelpful": -0.65,
    "useless": -0.72, "worthless": -0.78, "pointless": -0.68, "meaningless": -0.65,
    "hopeless": -0.75, "helpless": -0.72, "powerless": -0.68,
    "pathetic": -0.72, "pitiful": -0.68, "lame": -0.55, "mediocre": -0.50,
    "inferior": -0.60, "subpar": -0.55, "inadequate": -0.58, "insufficient": -0.55,
    "flawed": -0.55, "defective": -0.62, "broken": -0.65, "damaged": -0.60,
    "harmful": -0.70, "dangerous": -0.72, "threatening": -0.68, "hostile": -0.72,
    "aggressive": -0.65, "violent": -0.78, "cruel": -0.78, "harsh": -0.60,
    "painful": -0.68, "hurtful": -0.70, "offensive": -0.72, "insulting": -0.70,
    "rude": -0.65, "disrespectful": -0.68, "arrogant": -0.62, "selfish": -0.65,
    "greedy": -0.62, "corrupt": -0.72, "dishonest": -0.68, "deceitful": -0.70,
    "manipulative": -0.68, "toxic": -0.75, "malicious": -0.78,

    # Mild negative (-0.3 to -0.49)
    "boring": -0.48, "dull": -0.45, "tedious": -0.50, "tiresome": -0.48,
    "mundane": -0.40, "monotonous": -0.45, "repetitive": -0.42,
    "slow": -0.38, "sluggish": -0.42, "delayed": -0.40, "late": -0.35,
    "complicated": -0.42, "complex": -0.35, "confusing": -0.50, "unclear": -0.48,
    "vague": -0.42, "ambiguous": -0.40, "inconsistent": -0.45,
    "awkward": -0.42, "clumsy": -0.45, "sloppy": -0.50, "messy": -0.45,
    "noisy": -0.38, "chaotic": -0.50, "disorganized": -0.48,
    "stressful": -0.55, "anxious": -0.58, "nervous": -0.50, "worried": -0.52,
    "afraid": -0.58, "scared": -0.60, "fearful": -0.58, "terrified": -0.72,
    "lonely": -0.60, "isolated": -0.55, "abandoned": -0.65, "neglected": -0.60,
    "ignored": -0.55, "overlooked": -0.48, "forgotten": -0.52,
    "tired": -0.42, "exhausted": -0.55, "drained": -0.52, "burnt": -0.50,
    "overwhelmed": -0.55, "overloaded": -0.50, "burdened": -0.52,
    "stuck": -0.45, "trapped": -0.58, "blocked": -0.42, "stalled": -0.42,

    # Conversational AI interaction negative
    "error": -0.55, "bug": -0.55, "crash": -0.65, "fail": -0.62,
    "failed": -0.65, "failure": -0.68, "broken": -0.65, "glitch": -0.55,
    "laggy": -0.50, "unresponsive": -0.58, "hang": -0.50, "freeze": -0.55,
    "spam": -0.55, "scam": -0.72, "fake": -0.60, "misleading": -0.62,
    "inaccurate": -0.58, "incorrect": -0.55, "irrelevant": -0.50,
    "incompetent": -0.65, "clueless": -0.58, "ignorant": -0.60,
    "nonsense": -0.60, "gibberish": -0.62, "rubbish": -0.65, "garbage": -0.68,
    "waste": -0.55, "wasted": -0.58, "lost": -0.45, "missing": -0.42,
    "problem": -0.50, "issue": -0.42, "trouble": -0.48, "difficulty": -0.45,
    "struggle": -0.50, "suffering": -0.65, "pain": -0.60,
    "regret": -0.62, "sorry": -0.45, "apologize": -0.40, "blame": -0.55,
    "fault": -0.50, "guilt": -0.58, "shame": -0.62,
    "doubt": -0.45, "skeptical": -0.42, "suspicious": -0.48, "distrust": -0.55,
    "reject": -0.60, "rejected": -0.65, "denied": -0.58, "refused": -0.55,
    "complaint": -0.55, "complain": -0.52, "criticize": -0.50, "condemn": -0.68,
    "ridiculous": -0.60, "absurd": -0.58, "stupid": -0.65, "idiotic": -0.72,
    "annoying": -0.58, "irritating": -0.55, "infuriating": -0.72,
    "depressing": -0.68, "gloomy": -0.55, "bleak": -0.58, "grim": -0.60,
    "disturbing": -0.62, "troubling": -0.55, "alarming": -0.60, "shocking": -0.62,
}

# Negation words that flip sentiment polarity
_NEGATION_WORDS: set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "wouldn't", "shouldn't",
    "couldn't", "hardly", "barely", "scarcely", "seldom", "rarely",
}

# Pre-compiled emoji pattern covering the most common Unicode emoji ranges.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # enclosed characters
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero width joiner
    "]+",
    flags=re.UNICODE,
)

# Sentence-ending punctuation (simple)
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

# Word tokeniser (simple)
_WORD_RE = re.compile(r"[a-zA-Z'\u2019]+")


# ====================================================================
# LinguisticAnalyzer
# ====================================================================

class LinguisticAnalyzer:
    """From-scratch linguistic feature extractor.

    All methods are stateless and operate on raw text.  No external NLP
    libraries are used.

    Example::

        la = LinguisticAnalyzer()
        features = la.compute_all("That was really great, thanks!")
        print(features["sentiment_valence"])  # ~0.27
    """

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def type_token_ratio(self, text: str) -> float:
        """Vocabulary richness: unique words / total words.

        Returns 0.0 for empty text.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def mean_word_length(self, text: str) -> float:
        """Average number of characters per word.

        Returns 0.0 for empty text.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def count_syllables(self, word: str) -> int:
        """Estimate syllable count using the vowel-group heuristic.

        Algorithm:
        1. Lower-case the word.
        2. Count groups of consecutive vowels (a, e, i, o, u).
        3. Subtract 1 if the word ends with a silent 'e' (single trailing e
           that is not the only vowel group).
        4. Return max(1, count).
        """
        word = word.lower().strip()
        if not word:
            return 1
        vowels = set("aeiou")
        count = 0
        prev_vowel = False
        for ch in word:
            if ch in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        # Silent-e rule: if word ends with 'e' and there is more than one
        # vowel group, subtract one.
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def flesch_kincaid_grade(self, text: str) -> float:
        """Flesch-Kincaid Grade Level.

        ``0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59``

        Returns 0.0 when there are no words or no sentences.
        """
        sentences = self.sentence_split(text)
        words = self._tokenize(text)
        if not words or not sentences:
            return 0.0
        total_syllables = sum(self.count_syllables(w) for w in words)
        n_words = len(words)
        n_sentences = len(sentences)
        grade = 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
        return max(0.0, grade)

    def question_ratio(self, text: str) -> float:
        """Fraction of sentences that end with a question mark.

        Returns 0.0 for empty text.
        """
        sentences = self.sentence_split(text)
        if not sentences:
            return 0.0
        q_count = sum(1 for s in sentences if s.rstrip().endswith("?"))
        return q_count / len(sentences)

    def exclamation_ratio(self, text: str) -> float:
        """Fraction of sentences that contain an exclamation mark.

        Returns 0.0 for empty text.
        """
        sentences = self.sentence_split(text)
        if not sentences:
            return 0.0
        e_count = sum(1 for s in sentences if "!" in s)
        return e_count / len(sentences)

    def formality_score(self, text: str) -> float:
        """Language formality score in [0, 1].

        ``1.0 - (contractions + slang_markers) / total_words``

        A score of 1.0 means fully formal; lower values indicate more
        informal language.  Returns 1.0 for empty text.
        """
        words = self._tokenize(text)
        if not words:
            return 1.0
        lower_words = [w.lower() for w in words]
        informal_count = 0
        for w in lower_words:
            if w in CONTRACTIONS or w in SLANG_MARKERS:
                informal_count += 1
        score = 1.0 - (informal_count / len(lower_words))
        return max(0.0, min(1.0, score))

    def emoji_count(self, text: str) -> int:
        """Count emoji characters in *text* using Unicode range matching."""
        # Each match may contain multiple emoji in a row (due to '+' in
        # the regex).  Count individual characters that are actually emoji.
        count = 0
        for ch in text:
            if _EMOJI_RE.match(ch):
                count += 1
        return count

    def sentiment_valence(self, text: str) -> float:
        """Lexicon-based sentiment score in [-1, 1].

        Algorithm:
        1. Tokenize and lowercase.
        2. Walk through tokens.  If the previous token is a negation word,
           flip the valence of the current token.
        3. Sum matched valences, divide by total word count.
        4. Clamp to [-1, 1].
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        lower_words = [w.lower() for w in words]
        total_valence = 0.0
        for i, w in enumerate(lower_words):
            valence = POSITIVE_LEXICON.get(w, 0.0) + NEGATIVE_LEXICON.get(w, 0.0)
            if valence != 0.0:
                # Check for negation in preceding 1-3 words
                negated = False
                window_start = max(0, i - 3)
                for j in range(window_start, i):
                    if lower_words[j] in _NEGATION_WORDS:
                        negated = True
                        break
                if negated:
                    valence = -valence
            total_valence += valence
        # Normalize by word count
        score = total_valence / len(lower_words)
        return max(-1.0, min(1.0, score))

    def sentence_split(self, text: str) -> list[str]:
        """Split *text* into sentences on ``.``, ``!``, ``?``.

        Handles common abbreviations (Mr., Dr., e.g., etc.) to avoid
        false splits.  Returns a list with at least one element for
        non-empty input.
        """
        if not text or not text.strip():
            return []

        # Protect abbreviations by temporarily replacing their periods.
        protected = text
        for abbr in _ABBREVIATIONS:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            replacement = abbr.replace(".", "\x00")
            protected = pattern.sub(replacement, protected)

        # Also protect decimal numbers (e.g. "3.14")
        protected = re.sub(r'(\d)\.(\d)', r'\1\x00\2', protected)

        # Split on sentence-ending punctuation
        parts: list[str] = []
        current: list[str] = []
        for ch in protected:
            current.append(ch)
            if ch in ".!?":
                sentence = "".join(current).strip().replace("\x00", ".")
                if sentence:
                    parts.append(sentence)
                current = []
        # Remaining text (no trailing punctuation)
        remainder = "".join(current).strip().replace("\x00", ".")
        if remainder:
            parts.append(remainder)

        return parts if parts else [text.strip()]

    def compute_all(self, text: str) -> dict[str, float]:
        """Compute all linguistic features and return them as a dict.

        Keys match the message-content feature names in
        :class:`InteractionFeatureVector`.

        Returns:
            Dictionary with keys: ``type_token_ratio``, ``mean_word_length``,
            ``flesch_kincaid``, ``question_ratio``, ``exclamation_ratio``,
            ``formality``, ``emoji_density``, ``sentiment_valence``,
            ``message_length``.
        """
        words = self._tokenize(text)
        n_words = len(words) if words else 0

        emoji_cnt = self.emoji_count(text)
        emoji_density = emoji_cnt / max(1, n_words)

        return {
            "type_token_ratio": self.type_token_ratio(text),
            "mean_word_length": self.mean_word_length(text),
            "flesch_kincaid": self.flesch_kincaid_grade(text),
            "question_ratio": self.question_ratio(text),
            "exclamation_ratio": self.exclamation_ratio(text),
            "formality": self.formality_score(text),
            "emoji_density": emoji_density,
            "sentiment_valence": self.sentiment_valence(text),
            "message_length": float(n_words),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize *text* into a list of word strings.

        Uses a simple regex that matches sequences of alphabetic characters
        and apostrophes.  Numbers and punctuation are excluded.
        """
        if not text:
            return []
        return _WORD_RE.findall(text.lower())

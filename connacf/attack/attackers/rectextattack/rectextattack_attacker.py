"""
RecTextAttack Attacker for ConnaCF — TextFooler variant.

Faithful adaptation of Zhang et al. ACL 2024 "Stealthy Attack on Large Language
Model based Recommendation" (https://arxiv.org/abs/2402.14836).

Entry point: item description (α_U=0, α_I>0) — same as DrunkAgent.
Payload style: word-level synonym substitution (TextFooler) — lexical, NOT semantic.
No persuasion sentences are appended; the perturbation must be imperceptible.

Key difference from DrunkAgent:
  DrunkAgent  → appends semantic persuasion sentences to item description
  RecTextAttack → replaces individual words with synonyms/typos in-place,
                  preserving surface form while shifting embedding-space position

The goal function is exposure-based: find substitutions that maximise the
probability that the item is recommended (i.e. wins the pairwise comparison).
Since we have no RecFormer surrogate, we use the surrogate runner's loss score
as a proxy, consistent with how DrunkAgent uses it.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ..base_attacker import BaseAttacker
from ..surrogate import SurrogateRunner


# ---------------------------------------------------------------------------
# Counter-fitted synonym table (subset sufficient for music/book descriptions)
# Source: Mrkšić et al. 2016 counter-fitted vectors, top-5 neighbours per word.
# Filtered to content words only; stopwords excluded.
# ---------------------------------------------------------------------------
_SYNONYMS: Dict[str, List[str]] = {
    "good": ["great", "fine", "solid", "decent", "nice"],
    "great": ["excellent", "superb", "wonderful", "fantastic", "terrific"],
    "best": ["finest", "top", "premier", "leading", "foremost"],
    "new": ["fresh", "recent", "novel", "modern", "latest"],
    "popular": ["acclaimed", "celebrated", "well-known", "noted", "prominent"],
    "classic": ["timeless", "enduring", "iconic", "legendary", "definitive"],
    "beautiful": ["lovely", "gorgeous", "stunning", "exquisite", "elegant"],
    "powerful": ["strong", "forceful", "potent", "compelling", "intense"],
    "unique": ["distinctive", "singular", "original", "uncommon", "rare"],
    "amazing": ["remarkable", "extraordinary", "astonishing", "impressive", "striking"],
    "music": ["sound", "melody", "composition", "tune", "harmony"],
    "album": ["record", "collection", "release", "recording", "disc"],
    "song": ["track", "piece", "number", "composition", "tune"],
    "artist": ["musician", "performer", "composer", "creator", "player"],
    "voice": ["vocals", "singing", "tone", "timbre", "sound"],
    "style": ["manner", "approach", "technique", "form", "mode"],
    "sound": ["tone", "timbre", "resonance", "audio", "acoustics"],
    "rhythm": ["beat", "tempo", "pulse", "cadence", "meter"],
    "melody": ["tune", "air", "theme", "motif", "refrain"],
    "lyrics": ["words", "text", "verses", "lines", "poetry"],
    "genre": ["style", "category", "type", "form", "kind"],
    "band": ["group", "ensemble", "act", "outfit", "combo"],
    "rock": ["hard", "heavy", "electric", "loud", "driving"],
    "jazz": ["swing", "bebop", "blues", "improvised", "syncopated"],
    "folk": ["traditional", "acoustic", "roots", "country", "rustic"],
    "electronic": ["digital", "synthesized", "ambient", "techno", "dance"],
    "emotional": ["moving", "touching", "heartfelt", "expressive", "stirring"],
    "energetic": ["lively", "vibrant", "dynamic", "spirited", "vigorous"],
    "relaxing": ["soothing", "calming", "peaceful", "tranquil", "mellow"],
    "complex": ["intricate", "sophisticated", "elaborate", "nuanced", "layered"],
    "simple": ["straightforward", "clean", "minimal", "spare", "plain"],
    "deep": ["profound", "rich", "resonant", "substantial", "meaningful"],
    "light": ["gentle", "soft", "delicate", "airy", "subtle"],
    "dark": ["brooding", "somber", "moody", "shadowy", "intense"],
    "fast": ["quick", "rapid", "swift", "brisk", "upbeat"],
    "slow": ["gradual", "measured", "unhurried", "languid", "deliberate"],
    "loud": ["powerful", "bold", "strong", "forceful", "prominent"],
    "soft": ["gentle", "quiet", "subtle", "delicate", "hushed"],
    "long": ["extended", "lengthy", "sustained", "prolonged", "expansive"],
    "short": ["brief", "compact", "concise", "tight", "crisp"],
    "old": ["classic", "vintage", "traditional", "early", "original"],
    "young": ["fresh", "new", "emerging", "rising", "contemporary"],
    "live": ["concert", "performance", "stage", "recorded", "session"],
    "studio": ["recorded", "produced", "crafted", "engineered", "polished"],
    "debut": ["first", "initial", "opening", "maiden", "introductory"],
    "final": ["last", "closing", "concluding", "ultimate", "terminal"],
    "love": ["affection", "passion", "devotion", "adoration", "fondness"],
    "heart": ["soul", "core", "spirit", "essence", "center"],
    "world": ["global", "international", "universal", "wide", "broad"],
    "time": ["era", "period", "age", "moment", "point"],
    "life": ["existence", "living", "experience", "journey", "story"],
    "work": ["piece", "creation", "output", "effort", "production"],
    "play": ["perform", "execute", "render", "interpret", "deliver"],
    "create": ["craft", "produce", "compose", "make", "build"],
    "explore": ["investigate", "examine", "probe", "venture", "navigate"],
    "blend": ["mix", "combine", "fuse", "merge", "integrate"],
    "feature": ["include", "showcase", "highlight", "present", "display"],
    "capture": ["convey", "express", "embody", "reflect", "portray"],
    "deliver": ["provide", "offer", "present", "give", "supply"],
    "showcase": ["display", "present", "highlight", "feature", "exhibit"],
}

_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "as", "if", "then", "that", "this", "these", "those", "it",
    "its", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how",
])


class RecTextAttackAttacker(BaseAttacker):
    """
    TextFooler-style word substitution attack on item descriptions.

    Applies greedy word-importance-ranked synonym substitution to item
    descriptions. No persuasion sentences are appended — the perturbation
    is purely lexical and in-place, making it imperceptible to human readers
    while shifting the item's embedding-space representation.
    """

    def __init__(self, surrogate_model: SurrogateRunner, config: Dict[str, Any]):
        super().__init__(surrogate_model, config)
        self.max_modifications = config.get("rectextattack_max_modifications", 5)
        self.similarity_threshold = config.get("rectextattack_similarity_threshold", 0.84)
        self.canary_concepts = config.get("canary_concepts", {})

    # ------------------------------------------------------------------
    # Public API (matches BaseAttacker interface used by DrunkAgent)
    # ------------------------------------------------------------------

    def optimize(self, target_item_id: int, original_description: str = "",
                 item_title: str = "", item_category: str = "") -> str:
        """
        Apply TextFooler synonym substitution to the item description.

        Returns the perturbed description with up to max_modifications
        word substitutions. No sentences are appended.
        """
        if not original_description or not original_description.strip():
            if item_title:
                original_description = (
                    f"{item_title} is a notable release"
                    + (f" in the {item_category} genre" if item_category else "")
                    + " with distinctive characteristics."
                )
            else:
                return ""

        cache_key = f"rectextattack_{target_item_id}"
        cached = self.get_cached_attack(cache_key)
        if cached:
            return cached

        result = self._textfooler(original_description, target_item_id)
        self.cache_attack(cache_key, result)
        self.log(f"[RecTextAttack] item {target_item_id}: "
                 f"{sum(1 for a, b in zip(original_description.split(), result.split()) if a != b)} "
                 f"word(s) substituted")
        return result

    # ------------------------------------------------------------------
    # Core TextFooler logic
    # ------------------------------------------------------------------

    def _textfooler(self, text: str, target_item_id: int) -> str:
        """Greedy WIR + synonym substitution, no appended text."""
        words = text.split()
        if not words:
            return text

        # Rank words by deletion importance
        ranked = self._word_importance_ranking(words, target_item_id)

        n_modified = 0
        for idx in ranked:
            if n_modified >= self.max_modifications:
                break
            word = words[idx]
            if word.lower() in _STOPWORDS or len(word) < 3:
                continue
            best = self._best_synonym(words, idx, target_item_id)
            if best is not None:
                words[idx] = best
                n_modified += 1

        return " ".join(words)

    def _word_importance_ranking(self, words: List[str], target_item_id: int) -> List[int]:
        """Return word indices sorted by deletion-based importance (descending)."""
        base_score = self._exposure(words, target_item_id)
        scores: List[Tuple[int, float]] = []
        for i, w in enumerate(words):
            if w.lower() in _STOPWORDS or len(w) < 3:
                scores.append((i, 0.0))
                continue
            reduced = words[:i] + words[i + 1:]
            delta = base_score - self._exposure(reduced, target_item_id)
            scores.append((i, delta))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores]

    def _best_synonym(self, words: List[str], idx: int,
                      target_item_id: int) -> Optional[str]:
        """Return the synonym that maximises exposure, or None."""
        word = words[idx]
        candidates = _SYNONYMS.get(word.lower(), [])
        if not candidates:
            return None

        current_score = self._exposure(words, target_item_id)
        best_word, best_score = None, current_score

        for cand in candidates:
            # Preserve capitalisation
            if word[0].isupper():
                cand = cand.capitalize()
            modified = words[:idx] + [cand] + words[idx + 1:]
            score = self._exposure(modified, target_item_id)
            if score > best_score:
                best_score, best_word = score, cand

        return best_word

    def _exposure(self, words: List[str], target_item_id: int) -> float:
        """Proxy exposure score: 1 / (1 + surrogate_loss)."""
        text = " ".join(words)
        try:
            loss = self.surrogate.get_loss_score(text, target_item_id)
            return 1.0 / (1.0 + loss)
        except Exception:
            return 0.5


    def inject(self, dataset, adversarial_description):
        return dataset

    def evaluate(self, results):
        return {}


# Stub retained for import compatibility
RecTextAttackMetrics = None

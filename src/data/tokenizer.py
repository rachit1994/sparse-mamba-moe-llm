"""Deterministic word-level tokenizer for the synthetic biography corpus.

LEAD-OWNED. Written by the lead rather than delegated because of one specific
trap (implementation/03_edge_cases_and_scares.md, dataset table):

    "Tokeniser splits names inconsistently -> answer inferable from token stats"

If a name or an attribute value tokenises differently depending on surrounding
context, the answer becomes partially inferable from token statistics alone and
every bits/param number is corrupted -- silently, with no exception and no red
test. The defence here is structural rather than statistical:

  * The vocabulary is WORD-LEVEL over a CLOSED corpus. The generator draws every
    attribute value from a fixed, finite vocabulary known at import time, so the
    complete set of tokens is enumerable up front. There is no subword merging,
    no BPE, no context-dependent segmentation -- a word maps to exactly one id,
    always, everywhere.
  * Encoding is a pure function of the token sequence: `text.split()` on
    whitespace after separating punctuation. Position cannot change a token's id.
  * `encode` raises on any out-of-vocabulary word rather than emitting <unk>.
    An <unk> would silently merge distinct attribute values into one token and
    understate the dataset's true entropy.

Why word-level is legitimate here and not a shortcut: the experiment measures
knowledge capacity against a known-entropy synthetic corpus, not open-domain
language modelling. A closed vocabulary is the correct instrument -- it makes the
mapping from dataset entropy to model targets exact. It would be the wrong
instrument for a real corpus.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "OutOfVocabularyError",
    "Tokenizer",
    "build_tokenizer",
]

# Punctuation that must be split off so that "Crelldale." and "Crelldale" map to
# the same value token. Without this, a value at the end of a sentence would get
# a different id than the same value mid-sentence -- exactly the inconsistency
# this module exists to prevent.
_PUNCT: Final[str] = ".,?!:;'\"()-"
_SPLIT_RE: Final[re.Pattern[str]] = re.compile(rf"([{re.escape(_PUNCT)}])")

# Digit runs are split into individual digits.
#
# Without this, every 4-digit name suffix ("-4417") and every year would be its
# own token, and the name space alone would need |LastA|x|LastB|x10000 ~= 3.3e6
# entries. At d_model=128 that is ~425M embedding parameters -- larger than the
# entire 100M preset, and it would make the embedding table dominate the
# parameter count that the capacity metric divides by (edge case S3).
#
# Splitting digits keeps the digit vocabulary at exactly 10 tokens and is
# perfectly consistent: "4417" -> "4","4","1","7" everywhere, always.
_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"(\d)")

PAD: Final[str] = "<pad>"
BOS: Final[str] = "<bos>"
EOS: Final[str] = "<eos>"
_SPECIALS: Final[tuple[str, ...]] = (PAD, BOS, EOS)


class OutOfVocabularyError(KeyError):
    """Raised when text contains a word absent from the closed vocabulary.

    Deliberately fatal. Emitting an <unk> token instead would merge distinct
    attribute values into a single id, understating dataset entropy and
    inflating measured bits-per-parameter.
    """


def _split_words(text: str) -> list[str]:
    """Split text into words, separating punctuation into its own tokens.

    Pure function of the input string: the same word yields the same output
    regardless of position or surrounding context.
    """
    spaced = _SPLIT_RE.sub(r" \1 ", text)
    spaced = _DIGIT_RE.sub(r" \1 ", spaced)
    return spaced.split()


class Tokenizer:
    """Closed-vocabulary word-level tokenizer.

    Attributes:
        vocab_size: number of distinct ids, suitable for ``ModelConfig.vocab_size``.
    """

    __slots__ = ("_itos", "_stoi")

    def __init__(self, words: list[str]) -> None:
        """Build from an explicit, already-deterministic word list.

        Args:
            words: every word the corpus can contain. Order defines ids, so the
                caller is responsible for sorting -- see :func:`build_tokenizer`.

        Raises:
            ValueError: if `words` is empty or contains duplicates.
        """
        if not words:
            raise ValueError("vocabulary must be non-empty")
        if len(set(words)) != len(words):
            dupes = sorted({w for w in words if words.count(w) > 1})[:5]
            raise ValueError(f"vocabulary contains duplicates, e.g. {dupes}")
        self._itos: Final[tuple[str, ...]] = tuple(words)
        self._stoi: Final[dict[str, int]] = {w: i for i, w in enumerate(words)}

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids.

        Args:
            text: any string drawn from the closed corpus.

        Returns:
            Token ids, one per word/punctuation mark.

        Raises:
            OutOfVocabularyError: if any word is not in the vocabulary. Fatal by
                design -- see the module docstring.
        """
        ids: list[int] = []
        for word in _split_words(text):
            try:
                ids.append(self._stoi[word])
            except KeyError:
                raise OutOfVocabularyError(
                    f"word {word!r} is not in the closed vocabulary "
                    f"(size {self.vocab_size}); refusing to emit <unk> because it "
                    f"would merge distinct attribute values and understate entropy"
                ) from None
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text.

        Round-trips ``encode`` up to whitespace normalisation around punctuation.

        Raises:
            IndexError: if any id is outside the vocabulary.
        """
        for i in ids:
            if not 0 <= i < self.vocab_size:
                raise IndexError(f"token id {i} outside vocabulary [0,{self.vocab_size})")
        return " ".join(self._itos[i] for i in ids)

    def encode_qa(self, question: str, answer: str) -> tuple[list[int], list[bool]]:
        """Encode a probe Q/A pair with a mask selecting only the ANSWER tokens.

        The mask is what ``src.metrics.bits.answer_cross_entropy_bits`` consumes.
        Scoring prompt tokens would inflate or deflate the measurement depending
        on prompt length (edge case S1 / 02_testing_philosophy section 5).

        Args:
            question: the probe question text.
            answer: the ground-truth answer text.

        Returns:
            (ids, answer_mask) of equal length. ``answer_mask[i]`` is True exactly
            where ``ids[i]`` belongs to the answer.

        Raises:
            OutOfVocabularyError: propagated from :meth:`encode`.
            ValueError: if the answer encodes to zero tokens, which would make the
                measurement silently vacuous.
        """
        q_ids = self.encode(question)
        a_ids = self.encode(answer)
        if not a_ids:
            raise ValueError(f"answer {answer!r} encoded to zero tokens")
        ids = q_ids + a_ids
        mask = [False] * len(q_ids) + [True] * len(a_ids)
        return ids, mask


def build_tokenizer() -> Tokenizer:
    """Build the corpus tokenizer deterministically from the generator's vocabularies.

    Reaches into ``src.data.generate``'s module-level vocabularies rather than
    sampling text, so the vocabulary covers every value the generator *can*
    produce, not merely every value some sample happened to contain. A sampled
    vocabulary would make ``encode`` raise on rare values later.

    Returns:
        A Tokenizer whose id assignment is fixed by sorted order, hence stable
        across processes and machines.
    """
    from src.data import generate as g

    words: set[str] = set()

    # Attribute values with a materialised vocabulary.
    for attribute in ("birth_city", "university", "major", "employer"):
        for value in g._attribute_value_vocab(attribute):  # noqa: SLF001
            words.update(_split_words(value))

    # birth_date is closed-form rather than materialised, so its components are
    # enumerated directly: month names plus digits (day and year are digit runs).
    words.update(g._MONTH_NAMES)  # noqa: SLF001
    words.update(str(d) for d in range(10))

    # Every name the generator can emit, decomposed into its parts. The full
    # name space is ~3.3e9 and cannot be enumerated, but it is built from a small
    # closed set of syllable parts plus a 4-digit suffix, all of which can be.
    for part in (g._NAME_FIRST_A, g._NAME_FIRST_B, g._NAME_LAST_A, g._NAME_LAST_B):  # noqa: SLF001
        words.update(part)
    # Names render as "<FirstA><FirstB> <LastA><LastB>-NNNN"; the fused forms and
    # the hyphen+digits tail are what actually appear in text.
    for a in g._NAME_FIRST_A:  # noqa: SLF001
        for b in g._NAME_FIRST_B:  # noqa: SLF001
            words.add(a + b)
    for a in g._NAME_LAST_A:  # noqa: SLF001
        for b in g._NAME_LAST_B:  # noqa: SLF001
            words.add(a + b)  # the "-NNNN" tail splits into "-" and digits

    # Template words, punctuation, and specials.
    for templates in (g._TRAIN_TEMPLATES, g._PROBE_TEMPLATES):  # noqa: SLF001
        for per_attr in templates.values():
            for tpl in per_attr:
                words.update(_split_words(tpl.replace("{name}", "").replace("{value}", "")))
    words.update(_PUNCT)

    ordered = list(_SPECIALS) + sorted(w for w in words if w and w not in _SPECIALS)
    return Tokenizer(ordered)

"""
Sentiment analysis service for financial articles.

Uses the multilingual ``jplu/tf-xlm-roberta-large`` transformer model
(XLM-RoBERTa) which natively supports **French**, **Arabic**, and 100+
other languages — ideal for analysing BVMT-related news articles that may
be written in French or Arabic.

This module is a **pure service class** with no HTTP endpoints.  It is
designed to be imported and called programmatically by any other module
in the FixTrade modular monolith (e.g. the ``trading`` infrastructure
adapter).

Label mapping
-------------
The model outputs one of the following labels:

* ``positive``  → ``1``
* ``negative``  → ``-1``
* ``neutral``   → ``0``

If the model returns an unknown label, an ``UnknownLabelError`` is raised
so the problem is surfaced immediately rather than swallowed silently.

If the model's confidence is below a configurable threshold, a
``LowConfidenceError`` is raised so the caller can decide what to do.

Example usage
-------------
::

    from app.nlp import SentimentAnalyzer

    analyzer = SentimentAnalyzer()

    # --- Single article ---------------------------------------------------
    score = analyzer.analyze(
        "La société XYZ a annoncé une augmentation de 20 %% de son "
        "chiffre d'affaires au troisième trimestre."
    )
    print(score)   # 1  (positive)

    # --- Batch of articles ------------------------------------------------
    articles = [
        "Les bénéfices ont fortement augmenté.",
        "أعلنت الشركة عن انخفاض حاد في أرباحها الفصلية.",
        "Le marché reste stable aujourd'hui.",
    ]
    scores = analyzer.analyze_batch(articles)
    print(scores)  # [1, -1, 0]

    # --- Custom confidence threshold --------------------------------------
    strict = SentimentAnalyzer(min_confidence=0.85)
    strict.analyze("Résultats mitigés pour la société.")
    # raises LowConfidenceError if model confidence < 0.85
"""

from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from transformers import pipeline  # type: ignore[import-untyped]

from app.nlp.unknownlabelserror import UnknownLabelError
from app.nlp.lowconfidenceerror import LowConfidenceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Using nlptown/bert-base-multilingual-uncased-sentiment
# Small multilingual model (~180MB) - supports 6 languages including French
# Outputs 1-5 stars, we'll map: 1-2 stars = negative, 3 = neutral, 4-5 = positive
_MODEL_NAME: str = "nlptown/bert-base-multilingual-uncased-sentiment"

_LABEL_MAP: dict[str, int] = {
    "1 star": -1,
    "2 stars": -1,
    "3 stars": 0,
    "4 stars": 1,
    "5 stars": 1,
    "POSITIVE": 1,
    "NEGATIVE": -1,
    "NEUTRAL": 0,
    "positive": 1,
    "negative": -1,
    "neutral": 0,
    "Positive": 1,
    "Negative": -1,
    "Neutral": 0,
}

_DEFAULT_MIN_CONFIDENCE: float = 0.0  # disabled by default
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_MAX_WORKERS: int = 4

# Regex: emoji & miscellaneous symbol Unicode blocks
_EMOJI_RE: re.Pattern[str] = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Arabic diacritics (tashkeel) — safe to strip for NLP
_ARABIC_DIACRITICS_RE: re.Pattern[str] = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
    "\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """Multilingual sentiment analyser for financial articles (French/Arabic).

    Wraps the HuggingFace ``transformers`` sentiment-analysis pipeline
    using ``nlptown/bert-base-multilingual-uncased-sentiment`` - a small
    multilingual BERT model (~180MB) supporting 6 languages including French.

    Parameters
    ----------
    model_name : str, optional
        Override the default model identifier (useful for testing with a
        smaller/faster model).
    min_confidence : float, optional
        Minimum model confidence required to accept a prediction.
        Set to ``0.0`` (default) to accept all predictions.
        Raises ``LowConfidenceError`` when the threshold is not met.
    batch_size : int, optional
        Number of texts to feed to the pipeline in a single forward pass
        (default ``32``).
    max_workers : int, optional
        Maximum threads for ``analyze_batch`` parallelisation
        (default ``4``).

    Attributes
    ----------
    _pipeline : transformers.Pipeline
        The underlying HuggingFace inference pipeline.

    Example
    -------
    >>> from app.nlp import SentimentAnalyzer
    >>> analyzer = SentimentAnalyzer()
    >>> analyzer.analyze("Les bénéfices ont fortement augmenté.")
    1
    >>> analyzer.analyze("الشركة تعاني من خسائر كبيرة.")
    -1
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1.")

        self._min_confidence = min_confidence
        self._batch_size = batch_size
        self._max_workers = max_workers

        logger.info("Loading sentiment model: %s …", model_name)
        self._pipeline: Any = pipeline(
            task="sentiment-analysis",
            model=model_name,
        )
        logger.info("Sentiment model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Text preprocessing
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess(text: str) -> str:
        """Normalise and clean a mixed-script article for NLP inference.

        Steps applied:
        1. Unicode NFC normalisation (canonical decomposition + composition).
        2. Strip Arabic diacritics (tashkeel) — they add noise for
           sentiment models and are inconsistently present in news.
        3. Remove emojis and miscellaneous symbols.
        4. Collapse multiple whitespace characters into a single space.

        Parameters
        ----------
        text : str
            Raw article text (may mix French, Arabic, numbers, symbols).

        Returns
        -------
        str
            Cleaned text ready for the sentiment pipeline.
        """
        text = unicodedata.normalize("NFC", text)
        text = _ARABIC_DIACRITICS_RE.sub("", text)
        text = _EMOJI_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------ #
    # Single-article analysis
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> int:
        """Analyse the sentiment of a single financial article.

        The text is preprocessed automatically before inference.

        Parameters
        ----------
        text : str
            The full text (or headline) of a financial article.

        Returns
        -------
        int
            ``1`` for positive, ``-1`` for negative, ``0`` for neutral.

        Raises
        ------
        ValueError
            If *text* is empty or blank.
        UnknownLabelError
            If the model returns a label not in the expected mapping.
        LowConfidenceError
            If the model confidence is below ``min_confidence``.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty or blank.")

        cleaned = self.preprocess(text)
        result: dict[str, Any] = self._pipeline(cleaned)[0]

        return self._map_result(result)

    # ------------------------------------------------------------------ #
    # Batch analysis
    # ------------------------------------------------------------------ #

    def analyze_batch(
        self,
        texts: list[str],
        max_workers: int | None = None,
    ) -> list[int]:
        """Analyse sentiment for a list of articles in parallel.

        Texts are preprocessed, split into chunks of ``batch_size``, and
        each chunk is sent to the pipeline.  Chunks are processed across
        multiple threads so that CPU-bound preprocessing and I/O-bound
        model loading do not block each other.

        Parameters
        ----------
        texts : list[str]
            Articles to analyse.
        max_workers : int, optional
            Override the instance-level ``max_workers`` for this call.

        Returns
        -------
        list[int]
            Sentiment scores in the same order as the input texts.

        Raises
        ------
        ValueError
            If *texts* is empty or contains empty / blank strings.
        UnknownLabelError
            If any prediction returns an unknown label.
        LowConfidenceError
            If any prediction falls below ``min_confidence``.
        """
        if not texts:
            raise ValueError("texts list must not be empty.")

        # Validate & preprocess up-front so errors surface early
        cleaned: list[str] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(
                    f"Text at index {i} is empty or blank."
                )
            cleaned.append(self.preprocess(t))

        workers = max_workers if max_workers is not None else self._max_workers

        # Split into chunks for pipeline batch inference
        chunks: list[list[str]] = [
            cleaned[i : i + self._batch_size]
            for i in range(0, len(cleaned), self._batch_size)
        ]

        # Ordered results placeholder
        all_scores: list[int] = [0] * len(cleaned)
        chunk_offsets = list(range(0, len(cleaned), self._batch_size))

        def _process_chunk(
            chunk_idx: int, chunk: list[str]
        ) -> tuple[int, list[int]]:
            results: list[dict[str, Any]] = self._pipeline(chunk)
            scores = [self._map_result(r) for r in results]
            return chunk_idx, scores

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_chunk, idx, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                chunk_idx, scores = future.result()
                offset = chunk_offsets[chunk_idx]
                for j, score in enumerate(scores):
                    all_scores[offset + j] = score

        logger.info(
            "Batch analysis complete: %d articles processed.", len(texts)
        )
        return all_scores

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _map_result(self, result: dict[str, Any]) -> int:
        """Map a single pipeline result dict to an integer score.

        Parameters
        ----------
        result : dict
            A single element from the HuggingFace pipeline output,
            containing ``"label"`` and ``"score"`` keys.

        Returns
        -------
        int
            Mapped sentiment score.

        Raises
        ------
        UnknownLabelError
            If the label is not in ``_LABEL_MAP``.
        LowConfidenceError
            If confidence is below ``min_confidence``.
        """
        raw_label: str = result["label"].strip().lower()
        confidence: float = result["score"]

        score = _LABEL_MAP.get(raw_label)

        if score is None:
            raise UnknownLabelError(label=raw_label, confidence=confidence)

        if self._min_confidence > 0.0 and confidence < self._min_confidence:
            raise LowConfidenceError(
                label=raw_label,
                confidence=confidence,
                threshold=self._min_confidence,
            )

        logger.debug(
            "Sentiment: label=%s  score=%d  confidence=%.4f",
            raw_label,
            score,
            confidence,
        )
        return score

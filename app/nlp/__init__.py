"""
NLP module for sentiment analysis.

Exports the SentimentAnalyzer and custom error types.
"""

from app.nlp.lowconfidenceerror import LowConfidenceError
from app.nlp.sentiment import SentimentAnalyzer
from app.nlp.unknownlabelserror import UnknownLabelError

__all__ = [
    "SentimentAnalyzer",
    "LowConfidenceError",
    "UnknownLabelError",
]

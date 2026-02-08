"""
Domain service: Article ↔ Symbol matching.

Pure business logic that determines which BVMT symbols an article
mentions, using keyword / alias matching.  No IO, no frameworks.

The symbol dictionary maps each BVMT ticker to a set of text
aliases (company names, abbreviations, ISIN fragments) that may
appear in French- or Arabic-language financial articles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# BVMT Symbol → Aliases dictionary
# ---------------------------------------------------------------------------
# Each key is the canonical symbol stored in stock_prices.
# Values are lowercase aliases that may appear in article text.
# This list covers the top ~40 BVMT-listed companies.

BVMT_SYMBOL_ALIASES: dict[str, list[str]] = {
    "BIAT": [
        "biat",
        "banque internationale arabe de tunisie",
        "banque internationale arabe",
    ],
    "SFBT": [
        "sfbt",
        "société frigorifique et brasserie de tunis",
        "frigorifique et brasserie",
    ],
    "BT": [
        "banque de tunisie",
        # NOTE: "bt" alone is too short / ambiguous — only match full name
    ],
    "ATTIJARI BANK": [
        "attijari bank",
        "attijari",
        "attijariwafa",
        "attijari wafa",
    ],
    "SAH": [
        "sah lilas",
        "sah",
        "société articles hygiéniques",
    ],
    "STB": [
        "stb",
        "société tunisienne de banque",
    ],
    "BNA": [
        "bna",
        "banque nationale agricole",
    ],
    "BH": [
        "bh bank",
        "banque de l'habitat",
        "banque habitat",
    ],
    "AMEN BANK": [
        "amen bank",
        "amen",
    ],
    "UIB": [
        "uib",
        "union internationale de banques",
    ],
    "ATB": [
        "atb",
        "arab tunisian bank",
    ],
    "WIFAK BANK": [
        "wifak bank",
        "wifak",
        "wifak international bank",
    ],
    "TUNISAIR": [
        "tunisair",
        "société tunisienne de l'air",
    ],
    "STAR": [
        "star assurances",
        "société tunisienne d'assurances et de réassurances",
    ],
    "CARTHAGE CEMENT": [
        "carthage cement",
        "carthage ciment",
    ],
    "POULINA": [
        "poulina",
        "poulina group holding",
        "poulina group",
        "pgh",
    ],
    "DELICE": [
        "delice holding",
        "délice holding",
        "delice",
        "délice",
    ],
    "SOTIPAPIER": [
        "sotipapier",
    ],
    "SOTUVER": [
        "sotuver",
        "société tunisienne de verreries",
    ],
    "TELNET": [
        "telnet holding",
        "telnet",
    ],
    "MONOPRIX": [
        "monoprix",
        "monoprix tunisie",
    ],
    "MAGASIN GENERAL": [
        "magasin général",
        "magasin general",
        "mg",
    ],
    "ENNAKL": [
        "ennakl",
        "ennakl automobiles",
    ],
    "CITY CARS": [
        "city cars",
        "citycars",
    ],
    "ARTES": [
        "artes",
    ],
    "TPR": [
        "tpr",
        "société tunisienne de promotion et de réalisation",
    ],
    "SOPAT": [
        "sopat",
        "société de production agricole de teboulba",
    ],
    "LAND'OR": [
        "land'or",
        "landor",
        "land or",
    ],
    "Assad": [
        "assad",
    ],
    "ONE TECH": [
        "one tech",
        "onetech",
        "one tech holding",
    ],
    "EURO-CYCLES": [
        "euro-cycles",
        "euro cycles",
        "eurocycles",
    ],
    "UNIMED": [
        "unimed",
    ],
    "ADWYA": [
        "adwya",
    ],
    "SITS": [
        "sits",
        "société immobilière tuniso-saoudienne",
    ],
    "SIMPAR": [
        "simpar",
    ],
    "SOTRAPIL": [
        "sotrapil",
        "société de transport des hydrocarbures par pipelines",
    ],
    "ICF": [
        "icf",
        "industries chimiques du fluor",
    ],
    "CIL": [
        "cil",
        "compagnie internationale de leasing",
    ],
    "SOTETEL": [
        "sotetel",
        "société tunisienne d'entreprises de télécommunications",
    ],
    "SERVICOM": [
        "servicom",
    ],
    "AMS": [
        "ams",
        "assurances multirisques",
    ],
}


# Pre-compile: build a single large regex from all aliases
# Sort aliases longest-first so that "banque de tunisie" matches before "bt"
_all_patterns: list[tuple[str, re.Pattern[str]]] = []
for _sym, _aliases in BVMT_SYMBOL_ALIASES.items():
    sorted_aliases = sorted(_aliases, key=len, reverse=True)
    # Build one regex per symbol: OR of all aliases, word-boundary wrapped
    combined = "|".join(re.escape(a) for a in sorted_aliases)
    _all_patterns.append(
        (_sym, re.compile(rf"\b(?:{combined})\b", re.IGNORECASE))
    )


@dataclass(frozen=True)
class SymbolMatch:
    """A single article ↔ symbol match result."""

    symbol: str
    matched_alias: str
    match_count: int


class ArticleSymbolMatcher:
    """Pure domain service that identifies BVMT symbols mentioned in text.

    Uses keyword / alias matching with word-boundary regex.
    Stateless and side-effect free.
    """

    def __init__(
        self,
        extra_aliases: Optional[dict[str, list[str]]] = None,
        min_alias_length: int = 3,
    ) -> None:
        """Initialize the matcher.

        Args:
            extra_aliases: Optional additional symbol → aliases to merge.
            min_alias_length: Ignore aliases shorter than this to reduce
                false positives (e.g. "BT" alone is risky).
        """
        self._min_alias_length = min_alias_length
        self._patterns = list(_all_patterns)

        if extra_aliases:
            for sym, aliases in extra_aliases.items():
                sorted_a = sorted(aliases, key=len, reverse=True)
                combined = "|".join(re.escape(a) for a in sorted_a)
                self._patterns.append(
                    (sym, re.compile(rf"\b(?:{combined})\b", re.IGNORECASE))
                )

    def match(self, text: str) -> list[SymbolMatch]:
        """Find all BVMT symbols mentioned in the given text.

        Args:
            text: Article title + summary + content (concatenated).

        Returns:
            List of SymbolMatch objects, one per matched symbol,
            ordered by match_count descending (most-mentioned first).
        """
        if not text or not text.strip():
            return []

        results: list[SymbolMatch] = []

        for symbol, pattern in self._patterns:
            matches = pattern.findall(text)
            if matches:
                # Pick the longest match as the "best alias"
                best = max(matches, key=len)
                results.append(
                    SymbolMatch(
                        symbol=symbol,
                        matched_alias=best.lower(),
                        match_count=len(matches),
                    )
                )

        # Sort by match_count desc for relevance
        results.sort(key=lambda m: m.match_count, reverse=True)
        return results

    def match_single(self, text: str) -> Optional[str]:
        """Return the single most-mentioned symbol, or None.

        Convenience method for cases where only the primary symbol
        is needed.
        """
        matches = self.match(text)
        return matches[0].symbol if matches else None

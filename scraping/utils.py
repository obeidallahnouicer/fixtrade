import re
from typing import Optional

try:
    import dateparser as _dateparser
except Exception:
    _dateparser = None

from dateutil import parser as _dateutil_parser

MONTH_WORDS_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
DATE_LIKE_RE = re.compile(rf"({MONTH_WORDS_RE}[^\n\r]+\d{{4}}(?:[^\n\r]*)?)", re.IGNORECASE)


def parse_date_text(text: Optional[str]) -> Optional[str]:
    """Try parsing a date-like string and return an ISO 8601 string on success.

    Tries `dateparser` (supports languages) first, then `dateutil` as a fallback.
    Returns None if parsing fails.
    """
    if not text:
        return None
    text = text.strip()

    # extract a date-like fragment if the entire text contains extra content
    m = DATE_LIKE_RE.search(text)
    date_fragment = m.group(1) if m else text

    # try dateparser (handles many locales)
    if _dateparser:
        try:
            dt = _dateparser.parse(date_fragment, settings={"RETURN_AS_TIMEZONE_AWARE": False})
            if dt:
                return dt.isoformat()
        except Exception:
            pass

    # fallback to dateutil parser with fuzzy parsing
    try:
        dt = _dateutil_parser.parse(date_fragment, fuzzy=True)
        return dt.isoformat()
    except Exception:
        return None

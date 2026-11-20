"""Clean extracted document text without changing valid Unicode or formatting."""

import re

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def sanitize_text(text: str) -> str:
    """Remove PostgreSQL-incompatible NUL and replace non-UTF-8 surrogate code points.

    Keep valid Unicode, combining marks, whitespace, Markdown and LaTeX intact.
    Replacing invalid surrogates with U+FFFD preserves an indication of missing
    characters rather than silently merging the surrounding text.
    """
    return _SURROGATES.sub("\ufffd", text.replace("\x00", ""))

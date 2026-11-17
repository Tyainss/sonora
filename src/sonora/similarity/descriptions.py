import re
from html.parser import HTMLParser


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")


_ATTRIBUTION = re.compile(
    r"Read more on Last\.fm\.?|"
    r"User-contributed text is available under the Creative Commons "
    r"By-SA License; additional terms may apply\.?",
    flags=re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def clean_artist_description(value: str | None) -> str | None:
    """Remove Last.fm markup and attribution from an artist description."""
    if not value:
        return None

    parser = _TextParser()
    parser.feed(value)
    parser.close()
    text = " ".join(parser.parts)
    text = _ATTRIBUTION.sub(" ", text)
    text = _URL.sub(" ", text)
    text = re.sub(r"\s+\.(?=\s|$)", " ", text)
    text = " ".join(text.split()).strip().rstrip(":;").strip()
    return text if any(char.isalpha() for char in text) else None

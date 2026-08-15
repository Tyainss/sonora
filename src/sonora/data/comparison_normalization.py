import polars as pl

_APOSTROPHE_PATTERN = r"[’‘ʼ]"
_QUOTE_PATTERN = r"[“”„‟]"
_COMBINING_MARK_PATTERN = r"\p{M}+"
_PUNCTUATION_PATTERN = r"\p{P}+"
_WHITESPACE_PATTERN = r"\s+"


def normalize_for_comparison(expr: pl.Expr) -> pl.Expr:
    """Normalize text for identity-matching comparisons without changing stored names."""
    return (
        expr.cast(pl.String)
        .str.normalize("NFKD")
        .str.to_lowercase()
        .str.replace_all(_COMBINING_MARK_PATTERN, "")
        .str.replace_all(_APOSTROPHE_PATTERN, "'")
        .str.replace_all(_QUOTE_PATTERN, '"')
        .str.replace_all("&", " and ", literal=True)
        .str.replace_all(_PUNCTUATION_PATTERN, " ")
        .str.replace_all(_WHITESPACE_PATTERN, " ")
        .str.strip_chars()
        .replace("", None)
    )

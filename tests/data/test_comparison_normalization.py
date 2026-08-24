import polars as pl

from sonora.data.comparison_normalization import normalize_for_comparison


def _normalize(values: list[str | None]) -> list[str | None]:
    frame = pl.DataFrame({"value": values})
    return frame.select(normalize_for_comparison(pl.col("value"))).to_series().to_list()


def test_normalize_for_comparison_handles_unicode_case_and_whitespace():
    assert _normalize(["  Samuel   Úria  ", "Ｂjörk"]) == ["samuel uria", "bjork"]


def test_normalize_for_comparison_normalizes_ampersands_and_punctuation():
    values = [
        "King Gizzard & The Lizard Wizard",
        "King Gizzard And The Lizard Wizard",
        "AC/DC",
        "AC - DC",
    ]

    assert _normalize(values) == [
        "king gizzard and the lizard wizard",
        "king gizzard and the lizard wizard",
        "ac dc",
        "ac dc",
    ]


def test_normalize_for_comparison_normalizes_apostrophe_variants_without_overmerging():
    normalized = _normalize(["Chang’e", "Chang'e", "Change"])

    assert normalized == ["chang e", "chang e", "change"]
    assert normalized[0] != normalized[2]


def test_normalize_for_comparison_preserves_semantic_suffix_content():
    assert _normalize(["Song (Live)", "Song - Remastered 2024"]) == [
        "song live",
        "song remastered 2024",
    ]


def test_normalize_for_comparison_preserves_nulls_and_maps_blank_values_to_null():
    assert _normalize([None, "", "   "]) == [None, None, None]


def test_normalize_for_comparison_handles_all_null_columns():
    frame = pl.DataFrame({"value": pl.Series([None, None], dtype=pl.Null)})

    result = frame.select(normalize_for_comparison(pl.col("value")))

    assert result.to_series().to_list() == [None, None]
    assert result.schema["value"] == pl.String


def test_normalization_is_derived_without_overwriting_observed_values():
    frame = pl.DataFrame({"artist_name": ["Samuel Úria"]})

    result = frame.with_columns(
        normalize_for_comparison(pl.col("artist_name")).alias("artist_name_comparison")
    )

    assert result.row(0, named=True) == {
        "artist_name": "Samuel Úria",
        "artist_name_comparison": "samuel uria",
    }

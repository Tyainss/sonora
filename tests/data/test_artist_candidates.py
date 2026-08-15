import polars as pl
import pytest

from sonora.data.artist_candidates import (
    ArtistCandidateConfig,
    collect_artist_aliases,
    generate_artist_candidate_pairs,
)


def _aliases(*names: str) -> pl.DataFrame:
    events = pl.DataFrame(
        {
            "user_id": ["user-1"] * len(names),
            "artist_name": list(names),
        }
    ).lazy()
    return collect_artist_aliases(events)


def _candidate_names(candidates: pl.DataFrame) -> set[frozenset[str]]:
    return {
        frozenset((left, right))
        for left, right in candidates.select(
            "observed_artist_name_left", "observed_artist_name_right"
        ).iter_rows()
    }


def _candidate_row(
    candidates: pl.DataFrame,
    left: str,
    right: str,
) -> dict[str, object]:
    return candidates.filter(
        pl.col("observed_artist_name_left").is_in([left, right])
        & pl.col("observed_artist_name_right").is_in([left, right])
    ).row(0, named=True)


def test_collect_artist_aliases_is_global_and_preserves_observed_names():
    events = pl.DataFrame(
        {
            "user_id": ["user-1", "user-2", "user-2"],
            "artist_name": ["Samuel Úria", "Samuel Úria", "Samuel Uria"],
        }
    ).lazy()

    result = collect_artist_aliases(events)

    assert result.to_dicts() == [
        {
            "observed_artist_name": "Samuel Uria",
            "artist_name_comparison": "samuel uria",
        },
        {
            "observed_artist_name": "Samuel Úria",
            "artist_name_comparison": "samuel uria",
        },
    ]


def test_generate_candidates_includes_exact_normalized_variants():
    aliases = _aliases(
        "Samuel Uria",
        "Samuel Úria",
        "King Gizzard & The Lizard Wizard",
        "King Gizzard And The Lizard Wizard",
    )

    candidates = generate_artist_candidate_pairs(aliases)
    candidate_names = _candidate_names(candidates)

    assert frozenset(("Samuel Uria", "Samuel Úria")) in candidate_names
    assert (
        frozenset(
            (
                "King Gizzard & The Lizard Wizard",
                "King Gizzard And The Lizard Wizard",
            )
        )
        in candidate_names
    )
    samuel = _candidate_row(candidates, "Samuel Uria", "Samuel Úria")
    assert samuel["blocking_exact_name_match"] is True


def test_generate_candidates_includes_token_containment_variants():
    aliases = _aliases("Corona", "Conjunto Corona", "Nirvana")

    candidates = generate_artist_candidate_pairs(aliases)

    assert _candidate_names(candidates) == {frozenset(("Corona", "Conjunto Corona"))}
    corona = _candidate_row(candidates, "Corona", "Conjunto Corona")
    assert corona["blocking_shared_token_count"] == 1


def test_generate_candidates_includes_small_spelling_variants_via_ngrams():
    aliases = _aliases("Weeknd", "Weekend", "Nirvana")

    candidates = generate_artist_candidate_pairs(aliases)
    candidate_names = _candidate_names(candidates)

    assert frozenset(("Weeknd", "Weekend")) in candidate_names
    assert all("Nirvana" not in pair for pair in candidate_names)
    weeknd = _candidate_row(candidates, "Weeknd", "Weekend")
    assert weeknd["blocking_shared_ngram_count"] >= 2


def test_generate_candidates_does_not_fall_back_to_all_vs_all():
    aliases = _aliases("Björk", "Aphex Twin", "Nirvana", "Outkast")

    candidates = generate_artist_candidate_pairs(aliases)

    assert candidates.is_empty()


def test_large_common_blocks_are_skipped():
    aliases = _aliases(
        "Collective Alpha",
        "Collective Bravo",
        "Collective Charlie",
    )
    config = ArtistCandidateConfig(max_block_size=2)

    candidates = generate_artist_candidate_pairs(aliases, config=config)

    assert candidates.is_empty()


def test_generate_candidates_rejects_duplicate_alias_rows():
    aliases = pl.DataFrame(
        {
            "observed_artist_name": ["Artist", "Artist"],
            "artist_name_comparison": ["artist", "artist"],
        }
    )

    with pytest.raises(ValueError, match="unique observed artist names"):
        generate_artist_candidate_pairs(aliases)


def test_candidate_config_rejects_invalid_blocking_settings():
    with pytest.raises(ValueError, match="ngram_size"):
        ArtistCandidateConfig(ngram_size=1)

import polars as pl
import pytest

from sonora.data.artist_candidates import (
    ArtistCandidateConfig,
    collect_artist_aliases,
    generate_artist_candidate_pairs,
)


def _events(*rows: tuple[str, str]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "artist_name": [artist for artist, _ in rows],
            "track_name": [track for _, track in rows],
        }
    ).lazy()


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
    events = _events(
        ("Samuel Uria", "Song A"),
        ("Samuel Úria", "Song B"),
        ("King Gizzard & The Lizard Wizard", "Song C"),
        ("King Gizzard And The Lizard Wizard", "Song D"),
    )

    candidates = generate_artist_candidate_pairs(events)
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
    assert samuel["candidate_from_exact_name"] is True


def test_generate_candidates_includes_token_containment_variants():
    events = _events(
        ("Corona", "Song A"),
        ("Conjunto Corona", "Song B"),
        ("Nirvana", "Song C"),
    )

    candidates = generate_artist_candidate_pairs(events)

    assert _candidate_names(candidates) == {frozenset(("Corona", "Conjunto Corona"))}
    corona = _candidate_row(candidates, "Corona", "Conjunto Corona")
    assert corona["candidate_from_shared_token"] is True
    assert corona["blocking_shared_token_count"] == 1


def test_generate_candidates_includes_small_spelling_variants_via_ngrams():
    events = _events(
        ("Weeknd", "Song A"),
        ("Weekend", "Song B"),
        ("Nirvana", "Song C"),
    )

    candidates = generate_artist_candidate_pairs(events)
    candidate_names = _candidate_names(candidates)

    assert frozenset(("Weeknd", "Weekend")) in candidate_names
    assert all("Nirvana" not in pair for pair in candidate_names)
    weeknd = _candidate_row(candidates, "Weeknd", "Weekend")
    assert weeknd["candidate_from_shared_ngrams"] is True
    assert weeknd["blocking_shared_ngram_count"] >= 2


def test_generate_candidates_includes_shared_track_variants_with_unrelated_names():
    events = _events(
        ("Miguel Luz", "Same Song"),
        ("Mike Lyte", "Same Song"),
        ("Nirvana", "Different Song"),
    )

    candidates = generate_artist_candidate_pairs(events)

    assert _candidate_names(candidates) == {frozenset(("Miguel Luz", "Mike Lyte"))}
    renamed = _candidate_row(candidates, "Miguel Luz", "Mike Lyte")
    assert renamed["candidate_from_shared_tracks"] is True
    assert renamed["candidate_from_exact_name"] is False
    assert renamed["candidate_from_shared_token"] is False
    assert renamed["candidate_from_shared_ngrams"] is False
    assert renamed["blocking_shared_track_count"] == 1


def test_common_track_blocks_do_not_add_track_candidates():
    events = _events(
        ("Artist A", "Intro"),
        ("Artist B", "Intro"),
        ("Artist C", "Intro"),
    )
    config = ArtistCandidateConfig(max_track_block_size=2)

    candidates = generate_artist_candidate_pairs(events, config=config)

    assert not candidates.get_column("candidate_from_shared_tracks").any()
    assert candidates.get_column("blocking_shared_track_count").sum() == 0


def test_generate_candidates_does_not_fall_back_to_all_vs_all():
    events = _events(
        ("Björk", "Song A"),
        ("Aphex Twin", "Song B"),
        ("Nirvana", "Song C"),
        ("Outkast", "Song D"),
    )

    candidates = generate_artist_candidate_pairs(events)

    assert candidates.is_empty()


def test_large_common_name_blocks_are_skipped():
    events = _events(
        ("Collective Alpha", "Song A"),
        ("Collective Bravo", "Song B"),
        ("Collective Charlie", "Song C"),
    )
    config = ArtistCandidateConfig(max_name_block_size=2)

    candidates = generate_artist_candidate_pairs(events, config=config)

    assert candidates.is_empty()


def test_candidate_config_rejects_invalid_blocking_settings():
    with pytest.raises(ValueError, match="ngram_size"):
        ArtistCandidateConfig(ngram_size=1)

    with pytest.raises(ValueError, match="max_track_block_size"):
        ArtistCandidateConfig(max_track_block_size=1)

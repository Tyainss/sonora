from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from sonora.data.artist_candidates import (
    DEFAULT_ARTIST_CANDIDATE_CONFIG,
    ArtistCandidateConfig,
    collect_artist_aliases,
    generate_artist_candidate_pairs,
)
from sonora.data.comparison_normalization import normalize_for_comparison

_REQUIRED_EVENT_COLUMNS = {
    "artist_name",
    "track_name",
    "album_name",
    "artist_mbid",
    "track_mbid",
}
_REQUIRED_CANDIDATE_COLUMNS = {
    "observed_artist_name_left",
    "artist_name_comparison_left",
    "observed_artist_name_right",
    "artist_name_comparison_right",
}

ArtistMbidRelation = Literal["missing", "shared", "conflict"]

_EVIDENCE_SCHEMA = {
    "name_levenshtein_similarity": pl.Float64,
    "name_wratio": pl.Float64,
    "name_token_set_ratio": pl.Float64,
    "left_scrobble_count": pl.UInt64,
    "right_scrobble_count": pl.UInt64,
    "left_track_count": pl.UInt64,
    "right_track_count": pl.UInt64,
    "shared_track_count": pl.UInt64,
    "track_jaccard": pl.Float64,
    "track_containment": pl.Float64,
    "shared_track_scrobble_share_left": pl.Float64,
    "shared_track_scrobble_share_right": pl.Float64,
    "left_track_mbid_count": pl.UInt64,
    "right_track_mbid_count": pl.UInt64,
    "shared_track_mbid_count": pl.UInt64,
    "track_mbid_jaccard": pl.Float64,
    "track_mbid_containment": pl.Float64,
    "left_album_count": pl.UInt64,
    "right_album_count": pl.UInt64,
    "shared_album_count": pl.UInt64,
    "album_jaccard": pl.Float64,
    "album_containment": pl.Float64,
    "left_artist_mbid_count": pl.UInt64,
    "right_artist_mbid_count": pl.UInt64,
    "shared_artist_mbid_count": pl.UInt64,
    "artist_mbid_relation": pl.String,
}


@dataclass(slots=True)
class _ArtistProfile:
    scrobble_count: int
    tracks: frozenset[str]
    albums: frozenset[str]
    track_mbids: frozenset[str]
    artist_mbids: frozenset[str]
    track_play_counts: dict[str, int]


def build_artist_candidate_evidence(
    events: pl.LazyFrame,
    *,
    candidate_config: ArtistCandidateConfig = DEFAULT_ARTIST_CANDIDATE_CONFIG,
) -> pl.DataFrame:
    """Generate artist candidates and attach interpretable identity evidence."""
    aliases = collect_artist_aliases(events)
    candidates = generate_artist_candidate_pairs(aliases, config=candidate_config)
    return score_artist_candidate_evidence(events, candidates)


def score_artist_candidate_evidence(
    events: pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Attach lexical, catalogue, and identifier evidence to candidate pairs."""
    _validate_event_schema(events)
    _validate_candidate_schema(candidates)

    if candidates.is_empty():
        return _empty_evidence_frame(candidates)

    profiles = _build_artist_profiles(events)
    _validate_candidate_artists(candidates, profiles)

    evidence_rows = [
        _score_candidate(row, profiles) for row in candidates.iter_rows(named=True)
    ]
    evidence = pl.DataFrame(evidence_rows, schema=_EVIDENCE_SCHEMA)
    return candidates.hstack(evidence)


def _build_artist_profiles(events: pl.LazyFrame) -> dict[str, _ArtistProfile]:
    normalized = events.select(
        pl.col("artist_name"),
        normalize_for_comparison(pl.col("track_name")).alias("track_comparison"),
        normalize_for_comparison(pl.col("album_name")).alias("album_comparison"),
        pl.col("artist_mbid"),
        pl.col("track_mbid"),
    ).collect()

    catalogues = normalized.group_by("artist_name").agg(
        pl.len().alias("scrobble_count"),
        pl.col("track_comparison").drop_nulls().unique().alias("tracks"),
        pl.col("album_comparison").drop_nulls().unique().alias("albums"),
        pl.col("track_mbid").drop_nulls().unique().alias("track_mbids"),
        pl.col("artist_mbid").drop_nulls().unique().alias("artist_mbids"),
    )
    track_counts = (
        normalized.drop_nulls("track_comparison")
        .group_by("artist_name", "track_comparison")
        .len(name="play_count")
    )

    plays_by_artist: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for artist_name, track_name, play_count in track_counts.iter_rows():
        plays_by_artist[artist_name][track_name] = play_count

    profiles: dict[str, _ArtistProfile] = {}
    for row in catalogues.iter_rows(named=True):
        artist_name = row["artist_name"]
        profiles[artist_name] = _ArtistProfile(
            scrobble_count=row["scrobble_count"],
            tracks=frozenset(row["tracks"]),
            albums=frozenset(row["albums"]),
            track_mbids=frozenset(row["track_mbids"]),
            artist_mbids=frozenset(row["artist_mbids"]),
            track_play_counts=plays_by_artist[artist_name],
        )
    return profiles


def _score_candidate(
    candidate: dict[str, object],
    profiles: dict[str, _ArtistProfile],
) -> dict[str, object]:
    left_name = candidate["observed_artist_name_left"]
    right_name = candidate["observed_artist_name_right"]
    left_comparison = candidate["artist_name_comparison_left"]
    right_comparison = candidate["artist_name_comparison_right"]
    left = profiles[left_name]
    right = profiles[right_name]

    track_overlap = _set_overlap(left.tracks, right.tracks)
    track_mbid_overlap = _set_overlap(left.track_mbids, right.track_mbids)
    album_overlap = _set_overlap(left.albums, right.albums)
    artist_mbid_overlap = left.artist_mbids & right.artist_mbids

    shared_tracks = left.tracks & right.tracks

    return {
        "name_levenshtein_similarity": Levenshtein.normalized_similarity(
            left_comparison,
            right_comparison,
        ),
        "name_wratio": fuzz.WRatio(left_comparison, right_comparison) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(
            left_comparison,
            right_comparison,
        )
        / 100.0,
        "left_scrobble_count": left.scrobble_count,
        "right_scrobble_count": right.scrobble_count,
        "left_track_count": len(left.tracks),
        "right_track_count": len(right.tracks),
        "shared_track_count": track_overlap.shared_count,
        "track_jaccard": track_overlap.jaccard,
        "track_containment": track_overlap.containment,
        "shared_track_scrobble_share_left": _shared_scrobble_share(
            shared_tracks,
            left,
        ),
        "shared_track_scrobble_share_right": _shared_scrobble_share(
            shared_tracks,
            right,
        ),
        "left_track_mbid_count": len(left.track_mbids),
        "right_track_mbid_count": len(right.track_mbids),
        "shared_track_mbid_count": track_mbid_overlap.shared_count,
        "track_mbid_jaccard": track_mbid_overlap.jaccard,
        "track_mbid_containment": track_mbid_overlap.containment,
        "left_album_count": len(left.albums),
        "right_album_count": len(right.albums),
        "shared_album_count": album_overlap.shared_count,
        "album_jaccard": album_overlap.jaccard,
        "album_containment": album_overlap.containment,
        "left_artist_mbid_count": len(left.artist_mbids),
        "right_artist_mbid_count": len(right.artist_mbids),
        "shared_artist_mbid_count": len(artist_mbid_overlap),
        "artist_mbid_relation": _artist_mbid_relation(
            left.artist_mbids,
            right.artist_mbids,
        ),
    }


@dataclass(frozen=True, slots=True)
class _SetOverlap:
    shared_count: int
    jaccard: float | None
    containment: float | None


def _set_overlap(left: frozenset[str], right: frozenset[str]) -> _SetOverlap:
    shared_count = len(left & right)
    union_count = len(left | right)
    smaller_count = min(len(left), len(right))

    return _SetOverlap(
        shared_count=shared_count,
        jaccard=shared_count / union_count if union_count else None,
        containment=shared_count / smaller_count if smaller_count else None,
    )


def _shared_scrobble_share(
    shared_tracks: frozenset[str],
    profile: _ArtistProfile,
) -> float | None:
    if not profile.scrobble_count:
        return None
    shared_scrobbles = sum(
        profile.track_play_counts[track_name] for track_name in shared_tracks
    )
    return shared_scrobbles / profile.scrobble_count


def _artist_mbid_relation(
    left: frozenset[str],
    right: frozenset[str],
) -> ArtistMbidRelation:
    if not left or not right:
        return "missing"
    if left & right:
        return "shared"
    return "conflict"


def _validate_event_schema(events: pl.LazyFrame) -> None:
    columns = set(events.collect_schema().names())
    missing = sorted(_REQUIRED_EVENT_COLUMNS - columns)
    if missing:
        raise ValueError(f"Listening events are missing required columns: {missing}")


def _validate_candidate_schema(candidates: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Artist candidates are missing required columns: {missing}")


def _validate_candidate_artists(
    candidates: pl.DataFrame,
    profiles: dict[str, _ArtistProfile],
) -> None:
    candidate_artists = set(
        candidates.get_column("observed_artist_name_left").to_list()
    ) | set(candidates.get_column("observed_artist_name_right").to_list())
    missing = sorted(candidate_artists - set(profiles))
    if missing:
        raise ValueError(
            f"Artist candidates reference names absent from listening events: {missing}"
        )


def _empty_evidence_frame(candidates: pl.DataFrame) -> pl.DataFrame:
    return candidates.hstack(pl.DataFrame(schema=_EVIDENCE_SCHEMA))

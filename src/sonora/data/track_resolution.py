from dataclasses import dataclass

import polars as pl

from sonora.data.comparison_normalization import normalize_for_comparison

_REQUIRED_EVENT_COLUMNS = {"artist_id", "track_name", "track_mbid"}
_REQUIRED_EXISTING_ALIAS_COLUMNS = {"artist_id", "observed_track_name", "track_id"}
_TRACK_CLUSTER_SCHEMA = {
    "artist_id": pl.String,
    "observed_track_name": pl.String,
    "cluster_key": pl.String,
    "canonical_candidate_name": pl.String,
    "collapse_rule": pl.String,
    "existing_track_id": pl.String,
}


@dataclass(frozen=True, slots=True)
class TrackResolutionResult:
    clusters: pl.DataFrame


def build_track_song_keys(track_catalogue: pl.DataFrame) -> pl.DataFrame:
    """Build conservative song-level comparison keys inside canonical artists."""
    required = {"artist_id", "track_name"}
    missing = sorted(required - set(track_catalogue.columns))
    if missing:
        raise ValueError(f"Track catalogue is missing required columns: {missing}")

    duplicate_aliases = track_catalogue.select(
        pl.struct("artist_id", "track_name").is_duplicated().sum()
    ).item()
    if duplicate_aliases:
        raise ValueError("Track catalogue must contain one row per artist-track label")

    qualified = (
        track_catalogue.with_columns(
            pl.col("track_name")
            .str.extract(r"\(([^()]*)\)\s*$", 1)
            .alias("parenthetical_qualifier"),
            pl.col("track_name")
            .str.extract(r"\[([^\[\]]*)\]\s*$", 1)
            .alias("bracket_qualifier"),
            pl.col("track_name")
            .str.extract(r"\s[-–—]\s(.+)$", 1)
            .alias("dash_qualifier"),
        )
        .with_columns(
            pl.coalesce(
                "parenthetical_qualifier",
                "bracket_qualifier",
                "dash_qualifier",
            ).alias("track_qualifier"),
            pl.when(pl.col("parenthetical_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s*\([^()]*\)\s*$", ""))
            .when(pl.col("bracket_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s*\[[^\[\]]*\]\s*$", ""))
            .when(pl.col("dash_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s[-–—]\s.+$", ""))
            .otherwise(pl.col("track_name"))
            .alias("base_track_name"),
        )
        .with_columns(
            normalize_for_comparison(pl.col("track_qualifier")).alias(
                "comparison_qualifier"
            )
        )
        .with_columns(_collapse_rule_expr().alias("collapse_rule"))
        .with_columns(
            pl.when(pl.col("collapse_rule").is_not_null())
            .then(pl.col("base_track_name"))
            .otherwise(pl.col("track_name"))
            .alias("song_name")
        )
        .with_columns(
            normalize_for_comparison(pl.col("song_name")).alias("comparison_song_key")
        )
        .with_columns(
            pl.when(pl.col("comparison_song_key").is_null())
            .then(pl.concat_str([pl.lit("raw:"), pl.col("track_name")]))
            .otherwise(pl.col("comparison_song_key"))
            .alias("song_key")
        )
        .drop(
            "parenthetical_qualifier",
            "bracket_qualifier",
            "dash_qualifier",
            "comparison_qualifier",
            "comparison_song_key",
        )
    )
    return qualified


def resolve_track_identities(
    events: pl.LazyFrame,
    *,
    existing_aliases: pl.DataFrame | None = None,
) -> TrackResolutionResult:
    """Resolve track labels into song-level clusters within each canonical artist."""
    _validate_events(events)
    alias_catalogue = (
        events.group_by("artist_id", "track_name")
        .agg(
            pl.len().alias("scrobble_count"),
            pl.col("track_mbid").drop_nulls().unique().alias("track_mbids"),
        )
        .collect()
        .sort("artist_id", "track_name")
    )
    song_keys = build_track_song_keys(alias_catalogue)
    existing_map = _existing_alias_map(existing_aliases, song_keys=song_keys)

    nodes = [
        (row["artist_id"], row["track_name"]) for row in song_keys.iter_rows(named=True)
    ]
    parents = {node: node for node in nodes}

    song_key_anchor: dict[tuple[str, str], tuple[str, str]] = {}
    mbid_anchor: dict[tuple[str, str], tuple[str, str]] = {}
    for row in song_keys.iter_rows(named=True):
        node = (row["artist_id"], row["track_name"])
        song_group = (row["artist_id"], row["song_key"])
        anchor = song_key_anchor.setdefault(song_group, node)
        _union(parents, node, anchor)

        for track_mbid in row["track_mbids"]:
            mbid_group = (row["artist_id"], track_mbid)
            anchor = mbid_anchor.setdefault(mbid_group, node)
            _union(parents, node, anchor)

    components: dict[tuple[str, str], list[dict]] = {}
    for row in song_keys.iter_rows(named=True):
        node = (row["artist_id"], row["track_name"])
        root = _find(parents, node)
        components.setdefault(root, []).append(row)

    output_rows: list[dict] = []
    for members in components.values():
        existing_ids = {
            existing_map[(member["artist_id"], member["track_name"])]
            for member in members
            if (member["artist_id"], member["track_name"]) in existing_map
        }
        if len(existing_ids) > 1:
            names = sorted(member["track_name"] for member in members)
            raise ValueError(
                "Track resolution would merge multiple existing track IDs for "
                f"artist {members[0]['artist_id']}: {names}"
            )

        cluster_key = _cluster_key(members)
        existing_track_id = next(iter(existing_ids), None)
        for member in members:
            output_rows.append(
                {
                    "artist_id": member["artist_id"],
                    "observed_track_name": member["track_name"],
                    "cluster_key": cluster_key,
                    "canonical_candidate_name": member["song_name"],
                    "collapse_rule": member["collapse_rule"],
                    "existing_track_id": existing_track_id,
                }
            )

    clusters = pl.DataFrame(output_rows, schema=_TRACK_CLUSTER_SCHEMA).sort(
        "artist_id", "observed_track_name"
    )
    return TrackResolutionResult(clusters=clusters)


def _collapse_rule_expr() -> pl.Expr:
    qualifier = pl.col("comparison_qualifier")
    return (
        pl.when(
            qualifier.str.contains(r"\b(?:feat(?:uring)?|ft)\b|^(?:duet with|with)\b")
        )
        .then(pl.lit("featured_credit"))
        .when(qualifier.str.contains(r"\bremaster(?:ed)?\b"))
        .then(pl.lit("remaster"))
        .when(qualifier.str.contains(r"\b(?:live|ao vivo|en vivo)\b"))
        .then(pl.lit("live"))
        .when(qualifier.str.contains(r"\b(?:acoustic|acustic[oa]?)\b"))
        .then(pl.lit("acoustic"))
        .when(qualifier.str.contains(r"\bdemo\b"))
        .then(pl.lit("demo"))
        .when(qualifier.str.contains(r"\binstrumental\b"))
        .then(pl.lit("instrumental"))
        .when(qualifier.str.contains(r"\b(?:explicit|clean)(?: version)?\b"))
        .then(pl.lit("content_label"))
        .when(qualifier.str.contains(r"\brecorded at\b"))
        .then(pl.lit("recorded_at"))
        .when(qualifier.str.contains(r"\b(?:alternate|alternative|alt) take\b"))
        .then(pl.lit("alternate_take"))
        .when(qualifier.str.contains(r"\bradio\b.*\bedit\b|\bedit\b"))
        .then(pl.lit("edit"))
        .when(qualifier.str.contains(r"\b(?:remix|mix)\b"))
        .then(pl.lit("mix_or_remix"))
        .when(qualifier.str.contains(r"\b(?:mono|stereo)\b"))
        .then(pl.lit("mono_or_stereo"))
        .when(qualifier.str.contains(r"\bsession\b"))
        .then(pl.lit("session"))
        .when(qualifier.str.contains(r"\b(?:anniversary|deluxe|reissue)\b"))
        .then(pl.lit("release_edition"))
        .when(qualifier.str.contains(r"\bversion\b"))
        .then(pl.lit("version"))
        .otherwise(None)
    )


def _validate_events(events: pl.LazyFrame) -> None:
    missing = sorted(_REQUIRED_EVENT_COLUMNS - set(events.collect_schema().names()))
    if missing:
        raise ValueError(
            f"Resolved track events are missing required columns: {missing}"
        )


def _existing_alias_map(
    existing_aliases: pl.DataFrame | None,
    *,
    song_keys: pl.DataFrame,
) -> dict[tuple[str, str], str]:
    if existing_aliases is None:
        return {}

    missing = sorted(_REQUIRED_EXISTING_ALIAS_COLUMNS - set(existing_aliases.columns))
    if missing:
        raise ValueError(f"Existing track aliases are missing columns: {missing}")
    if any(
        existing_aliases.get_column(column).null_count()
        for column in _REQUIRED_EXISTING_ALIAS_COLUMNS
    ):
        raise ValueError("Existing track aliases cannot contain nulls")

    duplicate_aliases = existing_aliases.select(
        pl.struct("artist_id", "observed_track_name").is_duplicated().sum()
    ).item()
    if duplicate_aliases:
        raise ValueError("Existing track aliases must map each artist-track label once")

    observed = {
        (row["artist_id"], row["track_name"])
        for row in song_keys.select("artist_id", "track_name").iter_rows(named=True)
    }
    existing = {
        (row["artist_id"], row["observed_track_name"])
        for row in existing_aliases.select(
            "artist_id", "observed_track_name"
        ).iter_rows(named=True)
    }
    missing_from_events = sorted(existing - observed)
    if missing_from_events:
        raise ValueError(
            "Existing track aliases are absent from the current listening history: "
            f"{missing_from_events}"
        )

    return {
        (row["artist_id"], row["observed_track_name"]): row["track_id"]
        for row in existing_aliases.iter_rows(named=True)
    }


def _cluster_key(members: list[dict]) -> str:
    artist_id = members[0]["artist_id"]
    first_song_key = min(member["song_key"] for member in members)
    return f"{artist_id}\x1f{first_song_key}"


def _find(
    parents: dict[tuple[str, str], tuple[str, str]],
    node: tuple[str, str],
) -> tuple[str, str]:
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def _union(
    parents: dict[tuple[str, str], tuple[str, str]],
    left: tuple[str, str],
    right: tuple[str, str],
) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root == right_root:
        return
    if left_root < right_root:
        parents[right_root] = left_root
    else:
        parents[left_root] = right_root

from dataclasses import dataclass
from typing import Literal

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from sonora.data.artist_candidates import (
    DEFAULT_ARTIST_CANDIDATE_CONFIG,
    ArtistCandidateConfig,
)
from sonora.data.artist_evidence import build_artist_candidate_evidence
from sonora.data.comparison_normalization import normalize_for_comparison

ArtistIdentityDecision = Literal["merge", "possible_match", "reject"]

_DECISION_COLUMNS = {
    "identity_decision": pl.String,
    "decision_rule": pl.String,
}
_BLOCKED_MERGE_SCHEMA = {
    "observed_artist_name_left": pl.String,
    "observed_artist_name_right": pl.String,
    "reason": pl.String,
}
_CLUSTER_SCHEMA = {
    "observed_artist_name": pl.String,
    "cluster_key": pl.String,
    "existing_artist_id": pl.String,
}


@dataclass(frozen=True, slots=True)
class ArtistResolutionConfig:
    """Provisional rules for turning artist evidence into identity decisions."""

    near_exact_levenshtein: float = 0.95
    near_exact_wratio: float = 0.95
    name_catalogue_wratio: float = 0.88
    name_catalogue_min_shared_tracks: int = 2
    name_catalogue_min_track_containment: float = 0.50
    catalogue_merge_min_shared_tracks: int = 5
    catalogue_merge_min_track_containment: float = 0.50
    catalogue_merge_min_scrobble_share: float = 0.60
    conflict_override_wratio: float = 0.85
    conflict_override_min_shared_tracks: int = 5
    conflict_override_min_track_containment: float = 0.75
    possible_name_levenshtein: float = 0.85
    possible_name_wratio: float = 0.90
    possible_catalogue_min_shared_tracks: int = 3
    possible_catalogue_min_track_containment: float = 0.50

    def __post_init__(self) -> None:
        for field_name in (
            "near_exact_levenshtein",
            "near_exact_wratio",
            "name_catalogue_wratio",
            "name_catalogue_min_track_containment",
            "catalogue_merge_min_track_containment",
            "catalogue_merge_min_scrobble_share",
            "conflict_override_wratio",
            "conflict_override_min_track_containment",
            "possible_name_levenshtein",
            "possible_name_wratio",
            "possible_catalogue_min_track_containment",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        for field_name in (
            "name_catalogue_min_shared_tracks",
            "catalogue_merge_min_shared_tracks",
            "conflict_override_min_shared_tracks",
            "possible_catalogue_min_shared_tracks",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")


DEFAULT_ARTIST_RESOLUTION_CONFIG = ArtistResolutionConfig()


@dataclass(frozen=True, slots=True)
class ArtistResolutionResult:
    evidence: pl.DataFrame
    decisions: pl.DataFrame
    clusters: pl.DataFrame
    blocked_merges: pl.DataFrame


def resolve_artist_identities(
    events: pl.LazyFrame,
    *,
    candidate_config: ArtistCandidateConfig = DEFAULT_ARTIST_CANDIDATE_CONFIG,
    resolution_config: ArtistResolutionConfig = DEFAULT_ARTIST_RESOLUTION_CONFIG,
    existing_aliases: pl.DataFrame | None = None,
) -> ArtistResolutionResult:
    """Resolve observed artist names into conservative identity clusters."""
    observed_names = (
        events.select(pl.col("artist_name").alias("observed_artist_name"))
        .unique()
        .sort("observed_artist_name")
        .collect()
    )
    observed_name_set = set(observed_names.get_column("observed_artist_name").to_list())
    existing_map = _existing_alias_map(
        existing_aliases,
        observed_names=observed_name_set,
    )
    focus_artist_names = (
        observed_name_set - set(existing_map) if existing_aliases is not None else None
    )
    evidence = build_artist_candidate_evidence(
        events,
        candidate_config=candidate_config,
        focus_artist_names=focus_artist_names,
    )
    decisions = decide_artist_identity(evidence, config=resolution_config)
    clusters, blocked_merges = cluster_artist_aliases(
        events,
        observed_names,
        decisions,
        config=resolution_config,
        existing_aliases=existing_aliases,
    )
    return ArtistResolutionResult(
        evidence=evidence,
        decisions=decisions,
        clusters=clusters,
        blocked_merges=blocked_merges,
    )


def decide_artist_identity(
    evidence: pl.DataFrame,
    *,
    config: ArtistResolutionConfig = DEFAULT_ARTIST_RESOLUTION_CONFIG,
) -> pl.DataFrame:
    """Classify candidate pairs as merge, possible match, or reject."""
    if evidence.is_empty():
        return evidence.hstack(pl.DataFrame(schema=_DECISION_COLUMNS))

    rows = [_decide_pair(row, config=config) for row in evidence.iter_rows(named=True)]
    decisions = pl.DataFrame(rows, schema=_DECISION_COLUMNS)
    return evidence.hstack(decisions)


def cluster_artist_aliases(
    events: pl.LazyFrame,
    observed_names: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    config: ArtistResolutionConfig = DEFAULT_ARTIST_RESOLUTION_CONFIG,
    existing_aliases: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build clusters using combined catalogue, identifier, and name compatibility."""
    _validate_observed_names(observed_names)
    _validate_decisions(decisions)

    names = observed_names.get_column("observed_artist_name").to_list()
    profiles = _build_alias_profiles(events)
    missing_profiles = sorted(set(names) - set(profiles))
    if missing_profiles:
        raise ValueError(
            f"Observed artist names are missing profiles: {missing_profiles}"
        )
    existing_map = _existing_alias_map(existing_aliases, observed_names=set(names))

    clusters: dict[int, set[str]] = {}
    persisted_ids: dict[int, str | None] = {}
    alias_to_cluster: dict[str, int] = {}
    next_cluster_id = 0

    existing_groups: dict[str, list[str]] = {}
    for name, artist_id in existing_map.items():
        existing_groups.setdefault(artist_id, []).append(name)

    for artist_id in sorted(existing_groups):
        members = sorted(existing_groups[artist_id])
        clusters[next_cluster_id] = set(members)
        persisted_ids[next_cluster_id] = artist_id
        for name in members:
            alias_to_cluster[name] = next_cluster_id
        next_cluster_id += 1

    for name in names:
        if name in alias_to_cluster:
            continue
        clusters[next_cluster_id] = {name}
        persisted_ids[next_cluster_id] = None
        alias_to_cluster[name] = next_cluster_id
        next_cluster_id += 1

    blocked_rows: list[dict[str, str]] = []
    merge_rows = [
        row
        for row in decisions.iter_rows(named=True)
        if row["identity_decision"] == "merge"
    ]
    merge_rows.sort(key=_merge_sort_key)

    for row in merge_rows:
        left = row["observed_artist_name_left"]
        right = row["observed_artist_name_right"]
        left_cluster = alias_to_cluster[left]
        right_cluster = alias_to_cluster[right]

        if left_cluster == right_cluster:
            continue

        left_persisted = persisted_ids[left_cluster]
        right_persisted = persisted_ids[right_cluster]
        if (
            left_persisted is not None
            and right_persisted is not None
            and left_persisted != right_persisted
        ):
            blocked_rows.append(
                {
                    "observed_artist_name_left": left,
                    "observed_artist_name_right": right,
                    "reason": "canonical_merge_required",
                }
            )
            continue

        if not _clusters_are_compatible(
            clusters[left_cluster],
            clusters[right_cluster],
            profiles=profiles,
            config=config,
        ):
            blocked_rows.append(
                {
                    "observed_artist_name_left": left,
                    "observed_artist_name_right": right,
                    "reason": "cluster_incompatible",
                }
            )
            continue

        keep_cluster, drop_cluster = _choose_cluster_to_keep(
            left_cluster,
            right_cluster,
            persisted_ids=persisted_ids,
            clusters=clusters,
        )
        clusters[keep_cluster].update(clusters[drop_cluster])
        persisted_ids[keep_cluster] = (
            persisted_ids[keep_cluster] or persisted_ids[drop_cluster]
        )
        for name in clusters[drop_cluster]:
            alias_to_cluster[name] = keep_cluster
        del clusters[drop_cluster]
        del persisted_ids[drop_cluster]

    cluster_rows: list[dict[str, str | None]] = []
    for cluster_id, members in sorted(
        clusters.items(),
        key=lambda item: min(item[1]),
    ):
        cluster_key = min(members)
        for name in sorted(members):
            cluster_rows.append(
                {
                    "observed_artist_name": name,
                    "cluster_key": cluster_key,
                    "existing_artist_id": persisted_ids[cluster_id],
                }
            )

    return (
        pl.DataFrame(cluster_rows, schema=_CLUSTER_SCHEMA),
        pl.DataFrame(blocked_rows, schema=_BLOCKED_MERGE_SCHEMA),
    )


def _decide_pair(
    row: dict[str, object],
    *,
    config: ArtistResolutionConfig,
) -> dict[str, str]:
    mbid_relation = row["artist_mbid_relation"]
    exact_name = bool(row.get("candidate_from_exact_name", False))
    levenshtein = float(row["name_levenshtein_similarity"])
    wratio = float(row["name_wratio"])
    shared_tracks = int(row["shared_track_count"])
    track_containment = float(row["track_containment"] or 0.0)
    shared_scrobble_share_left = float(row["shared_track_scrobble_share_left"] or 0.0)
    shared_scrobble_share_right = float(row["shared_track_scrobble_share_right"] or 0.0)

    near_exact_name = (
        levenshtein >= config.near_exact_levenshtein
        and wratio >= config.near_exact_wratio
    )
    name_and_catalogue = (
        wratio >= config.name_catalogue_wratio
        and shared_tracks >= config.name_catalogue_min_shared_tracks
        and track_containment >= config.name_catalogue_min_track_containment
    )
    bidirectional_catalogue = (
        shared_tracks >= config.catalogue_merge_min_shared_tracks
        and track_containment >= config.catalogue_merge_min_track_containment
        and shared_scrobble_share_left >= config.catalogue_merge_min_scrobble_share
        and shared_scrobble_share_right >= config.catalogue_merge_min_scrobble_share
    )
    conflict_override = (
        wratio >= config.conflict_override_wratio
        and shared_tracks >= config.conflict_override_min_shared_tracks
        and track_containment >= config.conflict_override_min_track_containment
    )
    strong_name = (
        levenshtein >= config.possible_name_levenshtein
        or wratio >= config.possible_name_wratio
    )
    strong_catalogue = (
        shared_tracks >= config.possible_catalogue_min_shared_tracks
        and track_containment >= config.possible_catalogue_min_track_containment
    )

    if mbid_relation == "shared":
        return _decision("merge", "shared_artist_mbid")
    if mbid_relation == "conflict":
        return _decide_mbid_conflict(
            exact_name=exact_name,
            near_exact_name=near_exact_name,
            name_and_catalogue=name_and_catalogue,
            strong_catalogue=strong_catalogue,
            strong_name=strong_name,
            conflict_override=conflict_override,
        )
    return _decide_without_mbid_conflict(
        exact_name=exact_name,
        near_exact_name=near_exact_name,
        name_and_catalogue=name_and_catalogue,
        bidirectional_catalogue=bidirectional_catalogue,
        strong_catalogue=strong_catalogue,
        strong_name=strong_name,
    )


def _decide_mbid_conflict(
    *,
    exact_name: bool,
    near_exact_name: bool,
    name_and_catalogue: bool,
    strong_catalogue: bool,
    strong_name: bool,
    conflict_override: bool,
) -> dict[str, str]:
    if conflict_override:
        return _decision("merge", "mbid_conflict_overridden_by_catalogue")
    if (
        exact_name
        or near_exact_name
        or name_and_catalogue
        or strong_catalogue
        or strong_name
    ):
        return _decision("possible_match", "mbid_conflict_with_support")
    return _decision("reject", "mbid_conflict")


def _decide_without_mbid_conflict(
    *,
    exact_name: bool,
    near_exact_name: bool,
    name_and_catalogue: bool,
    bidirectional_catalogue: bool,
    strong_catalogue: bool,
    strong_name: bool,
) -> dict[str, str]:
    if exact_name:
        return _decision("merge", "same_comparison_name")
    if near_exact_name:
        return _decision("merge", "near_exact_name")
    if name_and_catalogue:
        return _decision("merge", "name_and_catalogue")
    if bidirectional_catalogue:
        return _decision("merge", "bidirectional_catalogue")
    if strong_catalogue:
        return _decision("possible_match", "strong_catalogue")
    if strong_name:
        return _decision("possible_match", "strong_name")
    return _decision("reject", "insufficient_evidence")


def _decision(decision: ArtistIdentityDecision, rule: str) -> dict[str, str]:
    return {
        "identity_decision": decision,
        "decision_rule": rule,
    }


def _validate_observed_names(observed_names: pl.DataFrame) -> None:
    if observed_names.columns != ["observed_artist_name"]:
        raise ValueError("Observed artist names must contain only observed_artist_name")
    if observed_names.get_column("observed_artist_name").null_count():
        raise ValueError("Observed artist names cannot contain nulls")
    if (
        observed_names.get_column("observed_artist_name").n_unique()
        != observed_names.height
    ):
        raise ValueError("Observed artist names must be unique")


def _validate_decisions(decisions: pl.DataFrame) -> None:
    required = {
        "observed_artist_name_left",
        "observed_artist_name_right",
        "identity_decision",
        "decision_rule",
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Artist decisions are missing required columns: {missing}")


def _existing_alias_map(
    existing_aliases: pl.DataFrame | None,
    *,
    observed_names: set[str],
) -> dict[str, str]:
    if existing_aliases is None:
        return {}

    required = {"observed_artist_name", "artist_id"}
    missing = sorted(required - set(existing_aliases.columns))
    if missing:
        raise ValueError(f"Existing artist aliases are missing columns: {missing}")

    if (
        existing_aliases.get_column("observed_artist_name").null_count()
        or existing_aliases.get_column("artist_id").null_count()
    ):
        raise ValueError("Existing artist aliases cannot contain nulls")

    duplicate_names = existing_aliases.select(
        pl.col("observed_artist_name").is_duplicated().sum()
    ).item()
    if duplicate_names:
        raise ValueError("Existing artist aliases must map each name once")

    existing_names = set(existing_aliases.get_column("observed_artist_name").to_list())
    missing_from_events = sorted(existing_names - observed_names)
    if missing_from_events:
        raise ValueError(
            "Existing artist aliases are absent from the current listening history: "
            f"{missing_from_events}"
        )

    return dict(
        existing_aliases.select("observed_artist_name", "artist_id").iter_rows()
    )


@dataclass(frozen=True, slots=True)
class _AliasProfile:
    comparison_name: str
    scrobble_count: int
    tracks: frozenset[str]
    artist_mbids: frozenset[str]


def _build_alias_profiles(events: pl.LazyFrame) -> dict[str, _AliasProfile]:
    rows = (
        events.select(
            pl.col("artist_name"),
            normalize_for_comparison(pl.col("artist_name")).alias("comparison_name"),
            normalize_for_comparison(pl.col("track_name")).alias("track_comparison"),
            pl.col("artist_mbid").cast(pl.String),
        )
        .group_by("artist_name")
        .agg(
            pl.col("comparison_name").first(),
            pl.len().alias("scrobble_count"),
            pl.col("track_comparison").drop_nulls().unique().alias("tracks"),
            pl.col("artist_mbid").drop_nulls().unique().alias("artist_mbids"),
        )
        .collect()
    )
    return {
        row["artist_name"]: _AliasProfile(
            comparison_name=row["comparison_name"],
            scrobble_count=row["scrobble_count"],
            tracks=frozenset(row["tracks"]),
            artist_mbids=frozenset(row["artist_mbids"]),
        )
        for row in rows.iter_rows(named=True)
    }


def _clusters_are_compatible(
    left_members: set[str],
    right_members: set[str],
    *,
    profiles: dict[str, _AliasProfile],
    config: ArtistResolutionConfig,
) -> bool:
    left_mbids = _combined_values(left_members, profiles, attribute="artist_mbids")
    right_mbids = _combined_values(right_members, profiles, attribute="artist_mbids")
    left_tracks = _combined_values(left_members, profiles, attribute="tracks")
    right_tracks = _combined_values(right_members, profiles, attribute="tracks")

    shared_tracks = len(left_tracks & right_tracks)
    smaller_catalogue = min(len(left_tracks), len(right_tracks))
    track_containment = shared_tracks / smaller_catalogue if smaller_catalogue else 0.0

    left_representative = _cluster_representative(left_members, profiles)
    right_representative = _cluster_representative(right_members, profiles)
    left_name = profiles[left_representative].comparison_name
    right_name = profiles[right_representative].comparison_name
    levenshtein = Levenshtein.normalized_similarity(left_name, right_name)
    wratio = fuzz.WRatio(left_name, right_name) / 100.0

    mbid_conflict = bool(
        left_mbids and right_mbids and left_mbids.isdisjoint(right_mbids)
    )
    if mbid_conflict:
        return (
            wratio >= config.conflict_override_wratio
            and shared_tracks >= config.conflict_override_min_shared_tracks
            and track_containment >= config.conflict_override_min_track_containment
        )

    if len(left_members) == 1 and len(right_members) == 1:
        return True

    catalogue_compatible = (
        shared_tracks >= config.name_catalogue_min_shared_tracks
        and track_containment >= config.name_catalogue_min_track_containment
    )
    representative_compatible = (
        levenshtein >= config.near_exact_levenshtein
        and wratio >= config.near_exact_wratio
    )
    return catalogue_compatible or representative_compatible


def _combined_values(
    members: set[str],
    profiles: dict[str, _AliasProfile],
    *,
    attribute: Literal["tracks", "artist_mbids"],
) -> frozenset[str]:
    values: set[str] = set()
    for member in members:
        values.update(getattr(profiles[member], attribute))
    return frozenset(values)


def _cluster_representative(
    members: set[str],
    profiles: dict[str, _AliasProfile],
) -> str:
    return min(
        members,
        key=lambda name: (-profiles[name].scrobble_count, name.casefold(), name),
    )


def _choose_cluster_to_keep(
    left_cluster: int,
    right_cluster: int,
    *,
    persisted_ids: dict[int, str | None],
    clusters: dict[int, set[str]],
) -> tuple[int, int]:
    left_persisted = persisted_ids[left_cluster]
    right_persisted = persisted_ids[right_cluster]

    if left_persisted is not None:
        return left_cluster, right_cluster
    if right_persisted is not None:
        return right_cluster, left_cluster

    left_key = min(clusters[left_cluster])
    right_key = min(clusters[right_cluster])
    if left_key <= right_key:
        return left_cluster, right_cluster
    return right_cluster, left_cluster


def _merge_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    rule_priority = {
        "shared_artist_mbid": 0,
        "same_comparison_name": 1,
        "near_exact_name": 2,
        "name_and_catalogue": 3,
        "bidirectional_catalogue": 4,
        "mbid_conflict_overridden_by_catalogue": 5,
    }
    return (
        rule_priority.get(str(row["decision_rule"]), 99),
        -float(row.get("name_wratio", 0.0) or 0.0),
        -float(row.get("name_levenshtein_similarity", 0.0) or 0.0),
        -float(row.get("track_containment", 0.0) or 0.0),
        -int(row.get("shared_track_count", 0) or 0),
        str(row["observed_artist_name_left"]),
        str(row["observed_artist_name_right"]),
    )

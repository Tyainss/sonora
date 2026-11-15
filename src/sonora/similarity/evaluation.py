import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import polars as pl

from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_TOP_N = 500


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _reference_names(path: Path) -> dict[str, list[str]]:
    references = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(references, dict) or not references:
        raise ValueError("Artist similarity references must be a nonempty mapping")
    for artist, matches in references.items():
        if (
            not isinstance(artist, str)
            or not artist.strip()
            or not isinstance(matches, list)
            or len(matches) != 5
            or any(not isinstance(name, str) or not name.strip() for name in matches)
            or len({name.casefold() for name in matches}) != 5
            or artist.casefold() in {name.casefold() for name in matches}
        ):
            raise ValueError(f"Invalid five-artist reference list for {artist!r}")
    return references


def build_artist_similarity_evaluation(
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    user_id: str | None = None,
    update_references: bool = False,
) -> Path:
    """Freeze one user's top artists and resolve ranked reference names to IDs."""
    output = paths.artist_similarity_evaluation
    source = paths.artist_similarity_references
    references = _reference_names(source)
    reference_hash = _sha256(source)

    artists = pl.read_parquet(paths.curated_artists).select(
        "artist_id", "canonical_name"
    )
    if artists["artist_id"].n_unique() != artists.height:
        raise ValueError("Duplicate canonical artist IDs")
    artist_names = dict(artists.iter_rows())

    if output.exists():
        snapshot = json.loads(output.read_text(encoding="utf-8"))
        if user_id is not None and user_id != snapshot["user_id"]:
            raise ValueError("Cannot change the user of a frozen evaluation")
        if not update_references:
            if snapshot["references_sha256"] != reference_hash:
                raise ValueError("References changed; use --update-references")
            return output
        candidate_ids = snapshot["candidate_artist_ids"]
        selected_user = snapshot["user_id"]
        source_hashes = snapshot["source_sha256"]
    else:
        events = pl.read_parquet(paths.curated_listening_events).select(
            "user_id", "artist_id"
        )
        users = events["user_id"].unique().to_list()
        if user_id is None:
            if len(users) != 1:
                raise ValueError("Select a user_id for this evaluation")
            user_id = users[0]
        if user_id not in users:
            raise ValueError(f"Unknown user_id: {user_id}")
        selected_user = user_id
        counts = (
            events.filter(pl.col("user_id") == selected_user)
            .group_by("artist_id")
            .len(name="listens")
            .join(artists, on="artist_id", how="left", validate="m:1")
            .with_columns(
                pl.col("canonical_name").str.to_lowercase().alias("_sort_name")
            )
            .sort(
                ["listens", "_sort_name", "artist_id"],
                descending=[True, False, False],
            )
            .head(_TOP_N)
        )
        if counts["canonical_name"].null_count():
            raise ValueError("Listening events contain unknown artist IDs")
        if counts.height != _TOP_N:
            raise ValueError(f"Expected at least {_TOP_N} listened-to artists")
        candidate_ids = counts["artist_id"].to_list()
        source_hashes = {
            "listening_events": _sha256(paths.curated_listening_events),
            "artists": _sha256(paths.curated_artists),
        }

    if not set(candidate_ids).issubset(artist_names):
        raise ValueError("Frozen candidate pool contains unknown artist IDs")
    names_to_ids: dict[str, str] = {}
    for artist_id in candidate_ids:
        name = artist_names[artist_id]
        key = name.strip().casefold()
        if key in names_to_ids:
            raise ValueError(f"Ambiguous artist name in candidate pool: {name}")
        names_to_ids[key] = artist_id

    def resolve(name: str) -> dict[str, str]:
        artist_id = names_to_ids.get(name.strip().casefold())
        if artist_id is None:
            raise ValueError(f"Artist not found in frozen top {_TOP_N}: {name}")
        return {"artist_id": artist_id, "canonical_name": artist_names[artist_id]}

    rows = []
    for artist, matches in references.items():
        reference = resolve(artist)
        selected = [resolve(name) for name in matches]
        if reference["artist_id"] in {match["artist_id"] for match in selected}:
            raise ValueError(f"An artist cannot reference itself: {artist}")
        if len({match["artist_id"] for match in selected}) != 5:
            raise ValueError(f"Duplicate reference artists for {artist}")
        rows.append({**reference, "matches": selected})
    if len({row["artist_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate reference artists")

    result = {
        "schema_version": 1,
        "user_id": selected_user,
        "top_n": _TOP_N,
        "source_sha256": source_hashes,
        "references_sha256": reference_hash,
        "candidate_artist_ids": candidate_ids,
        "references": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return output


def load_artist_similarity_evaluation(
    *, paths: DataPaths = DEFAULT_DATA_PATHS
) -> dict:
    snapshot = json.loads(
        paths.artist_similarity_evaluation.read_text(encoding="utf-8")
    )
    if snapshot["schema_version"] != 1 or snapshot["top_n"] != _TOP_N:
        raise ValueError("Unsupported similarity evaluation snapshot")
    if snapshot["references_sha256"] != _sha256(paths.artist_similarity_references):
        raise ValueError("References changed; rebuild with --update-references")
    if len(snapshot["candidate_artist_ids"]) != _TOP_N or len(
        set(snapshot["candidate_artist_ids"])
    ) != _TOP_N:
        raise ValueError("Invalid frozen candidate pool")
    pool = set(snapshot["candidate_artist_ids"])
    if any(
        row["artist_id"] not in pool
        or len(row["matches"]) != 5
        or any(match["artist_id"] not in pool for match in row["matches"])
        for row in snapshot["references"]
    ):
        raise ValueError("Reference artists must belong to the frozen candidate pool")
    return snapshot


def score_artist_similarity(
    snapshot: dict, rankings: dict[str, dict[str, list[str]]]
) -> pl.DataFrame:
    """Score reference recall and rank order within each approach's top ten."""
    pool = set(snapshot["candidate_artist_ids"])
    rows = []
    for approach, artist_rankings in rankings.items():
        for reference in snapshot["references"]:
            artist_id = reference["artist_id"]
            ranked = artist_rankings.get(artist_id, [])
            if (
                len(set(ranked)) != len(ranked)
                or any(
                    candidate not in pool or candidate == artist_id
                    for candidate in ranked
                )
            ):
                raise ValueError(f"Invalid {approach} ranking for {artist_id}")
            relevance = {
                match["artist_id"]: len(reference["matches"]) - index
                for index, match in enumerate(reference["matches"])
            }
            ideal_dcg = sum(
                gain / math.log2(index + 2)
                for index, gain in enumerate(range(len(relevance), 0, -1))
            )
            dcg = sum(
                relevance.get(candidate, 0) / math.log2(index + 2)
                for index, candidate in enumerate(ranked[:10])
            )
            rows.append(
                {
                    "Approach": approach,
                    "Artist": reference["canonical_name"],
                    "Available": artist_id in artist_rankings,
                    "Top 5": len(set(relevance).intersection(ranked[:5])),
                    "Top 10": len(set(relevance).intersection(ranked[:10])),
                    "NDCG@10": dcg / ideal_dcg if artist_id in artist_rankings else None,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze artist similarity references")
    parser.add_argument("--user-id")
    parser.add_argument("--update-references", action="store_true")
    args = parser.parse_args()
    print(
        build_artist_similarity_evaluation(
            user_id=args.user_id, update_references=args.update_references
        )
    )


if __name__ == "__main__":
    main()

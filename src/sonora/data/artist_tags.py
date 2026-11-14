import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl
from dotenv import load_dotenv
from lastfm_export.clients.lastfm import LastFMClient
from lastfm_export.io.artist_metadata import load_manifest, read_collection
from lastfm_export.models import ArtistInput
from lastfm_export.pipelines.artist_metadata import collect_artist_metadata

from sonora.data.artist_identity import (
    ambiguous_artist_ids as _ambiguous_artist_ids,
    artist_catalogue as _artist_catalogue,
    name_key as _name_key,
)
from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_TAG_SCHEMA = pl.List(pl.Struct({"name": pl.String, "weight": pl.Int64}))
_OUTPUT_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "queried_name": pl.String,
    "reported_name": pl.String,
    "status": pl.String,
    "collected_at": pl.String,
    "collection_id": pl.String,
    "tags": _TAG_SCHEMA,
}
_COMPLETED = {
    "success", "empty", "not_found", "identity_mismatch", "identity_unverified"
}
_STATUS_PRIORITY = {"success": 2, "empty": 1}


@dataclass(frozen=True, slots=True)
class _Record:
    artist_id: str
    queried_name: str
    reported_name: str | None
    status: str
    collected_at: str
    collection_id: str
    tags: list[dict]


def _read_mapping(path: Path) -> tuple[list[dict], bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid artist mapping: {path}")
    mapping = payload.get("artists")
    refresh = payload.get("refresh")
    if not isinstance(mapping, list) or not mapping or not isinstance(refresh, bool):
        raise ValueError(f"Invalid artist mapping: {path}")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("artist_id"), str)
        or not row["artist_id"].strip()
        or not isinstance(row.get("artist_name"), str)
        or not row["artist_name"].strip()
        for row in mapping
    ):
        raise ValueError(f"Invalid artist mapping: {path}")
    ids = [row["artist_id"] for row in mapping]
    names = [_name_key(row["artist_name"]) for row in mapping]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError(f"Duplicate artists in {path}")
    return mapping, refresh


def _extract_tags(raw: dict) -> tuple[str | None, list[dict]]:
    block = raw.get("toptags")
    if not isinstance(block, dict):
        raise ValueError("Invalid Last.fm tags response")
    attributes = block.get("@attr")
    reported = attributes.get("artist") if isinstance(attributes, dict) else None
    if not isinstance(reported, str) or not reported.strip():
        reported = None
    items = block.get("tag")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise ValueError("Last.fm tags must be a list")
    tags: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid Last.fm tag")
        name = item.get("name")
        weight = item.get("count")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Last.fm tag has no name")
        try:
            parsed_weight = int(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for tag {name!r}") from exc
        if parsed_weight < 0 or isinstance(weight, bool):
            raise ValueError(f"Invalid weight for tag {name!r}")
        key = _name_key(name)
        if key not in tags or parsed_weight > tags[key]["weight"]:
            tags[key] = {"name": name.strip(), "weight": parsed_weight}
    return reported, sorted(
        tags.values(), key=lambda tag: (-tag["weight"], tag["name"])
    )


def _load_collections(paths: DataPaths) -> tuple[list[_Record], list[Path]]:
    base = paths.raw_lastfm_artist_tags_dir
    records: list[_Record] = []
    unfinished: list[Path] = []
    if not base.exists():
        return records, unfinished

    for mapping_path in sorted(base.glob("*.artists.json")):
        directory = base / mapping_path.name.removesuffix(".artists.json")
        mapping, _ = _read_mapping(mapping_path)
        if not directory.exists():
            unfinished.append(directory)
            continue
        manifest = load_manifest(directory)
        if manifest["plan"]["settings"]["operations"] != ["lastfm_tags"]:
            raise ValueError(f"Unexpected operations in {directory}")
        plan_names = [row["artist_name"] for row in manifest["plan"]["inputs"]]
        if [row["artist_name"] for row in mapping] != plan_names:
            raise ValueError(f"Artist mapping disagrees with {directory}")
        if manifest["state"] != "completed":
            unfinished.append(directory)
            continue
        _, results = read_collection(directory)
        by_name = {
            result.input_artist["artist_name"]: result for result in results.values()
        }
        if len(by_name) != len(mapping):
            raise ValueError(f"Incomplete artist results: {directory}")
        for row in mapping:
            name = row["artist_name"]
            result = by_name[name]
            status = result.status
            tags: list[dict] = []
            reported = None
            if status in {"success", "empty"}:
                reported, tags = _extract_tags(result.raw_response)
                if reported is None:
                    status = "identity_unverified"
                    tags = []
                elif _name_key(reported) != _name_key(name):
                    status = "identity_mismatch"
                    tags = []
                elif (status == "empty" and tags) or (status == "success" and not tags):
                    raise ValueError(f"Tag status differs from payload: {name}")
            records.append(
                _Record(
                    artist_id=row["artist_id"],
                    queried_name=name,
                    reported_name=reported,
                    status=status,
                    collected_at=result.collected_at,
                    collection_id=result.collection_id,
                    tags=tags,
                )
            )
    collection_names = {p.name for p in base.iterdir() if p.is_dir()}
    mapping_names = {
        p.name.removesuffix(".artists.json") for p in base.glob("*.artists.json")
    }
    if collection_names - mapping_names:
        raise ValueError("Artist tag collection is missing its artist-ID mapping")
    return records, unfinished


def _latest_by_artist(
    records: list[_Record], artists: dict[str, str]
) -> dict[str, _Record]:
    latest: dict[str, _Record] = {}
    for record in sorted(records, key=lambda r: (r.collected_at, r.collection_id)):
        canonical_name = artists.get(record.artist_id)
        if canonical_name is None or _name_key(canonical_name) != _name_key(record.queried_name):
            continue
        prior = latest.get(record.artist_id)
        if prior is None or (
            _STATUS_PRIORITY.get(record.status, 0)
            >= _STATUS_PRIORITY.get(prior.status, 0)
            and (record.status in _COMPLETED or prior.status not in _COMPLETED)
        ):
            latest[record.artist_id] = record
    return latest


def _write_mapping(path: Path, mapping: list[dict], *, refresh: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {"refresh": refresh, "artists": mapping},
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")


def _report_progress(summary) -> None:
    if summary.attempted % 100 == 0 or summary.remaining == 0:
        print(
            f"Artist tags: {summary.total - summary.remaining}/{summary.total}, "
            f"failed={summary.failed}, empty={summary.empty}"
        )


def _lastfm_client() -> LastFMClient:
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LASTFM_API_KEY")
    return LastFMClient(api_key=api_key, user_agent="sonora")


def collect_artist_tags(
    *, paths: DataPaths = DEFAULT_DATA_PATHS, refresh: bool = False
) -> int:
    """Collect new or failed artists, or explicitly refresh the entire catalogue."""
    load_dotenv()
    artists = _artist_catalogue(paths)
    records, unfinished = _load_collections(paths)
    if len(unfinished) > 1:
        raise RuntimeError(
            "Multiple unfinished artist tag collections; resolve them first"
        )

    lastfm = None
    if unfinished:
        directory = unfinished[0]
        mapping_path = directory.with_name(f"{directory.name}.artists.json")
        mapping, resumed_refresh = _read_mapping(mapping_path)
        args = {
            "out": directory,
            "mode": "resume" if directory.exists() else "new",
            "retry_failed": directory.exists(),
            "recover_tail": directory.exists(),
        }
        if not directory.exists():
            args.update(
                artists=[ArtistInput(row["artist_name"]) for row in mapping],
                operations=["lastfm_tags"],
            )
        lastfm = _lastfm_client()
        args["lastfm"] = lastfm
        args["on_result"] = _report_progress
        summary = collect_artist_metadata(**args)
        if summary.failed or not summary.completed:
            raise RuntimeError(f"Unfinished Last.fm collection: {directory}")
        records, _ = _load_collections(paths)

    latest = _latest_by_artist(records, artists)
    ambiguous = _ambiguous_artist_ids(artists)
    pending = [
        {"artist_id": artist_id, "artist_name": name}
        for artist_id, name in sorted(
            artists.items(), key=lambda item: (item[1], item[0])
        )
        if artist_id not in ambiguous
        and (
            (refresh and not (unfinished and resumed_refresh))
            or artist_id not in latest
            or latest[artist_id].status not in _COMPLETED
        )
    ]
    if not pending:
        return 0
    if lastfm is None:
        lastfm = _lastfm_client()

    batch_name = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    )
    directory = paths.raw_lastfm_artist_tags_dir / batch_name
    _write_mapping(
        directory.with_name(f"{batch_name}.artists.json"), pending, refresh=refresh
    )
    summary = collect_artist_metadata(
        out=directory,
        artists=[ArtistInput(row["artist_name"]) for row in pending],
        operations=["lastfm_tags"],
        lastfm=lastfm,
        on_result=_report_progress,
    )
    if summary.failed or not summary.completed:
        raise RuntimeError(f"Last.fm collection has failures: {directory}")
    return len(pending)


def build_curated_artist_tags(*, paths: DataPaths = DEFAULT_DATA_PATHS) -> Path:
    """Rebuild one row per canonical artist from validated raw collections."""
    artists = _artist_catalogue(paths)
    records, unfinished = _load_collections(paths)
    if unfinished:
        raise RuntimeError("Finish or resume artist tag collections before curating")
    latest = _latest_by_artist(records, artists)
    ambiguous = _ambiguous_artist_ids(artists)
    rows = []
    for artist_id, canonical_name in sorted(artists.items()):
        record = None if artist_id in ambiguous else latest.get(artist_id)
        rows.append(
            {
                "artist_id": artist_id,
                "canonical_name": canonical_name,
                "queried_name": record.queried_name if record else None,
                "reported_name": record.reported_name if record else None,
                "status": (
                    "ambiguous_name"
                    if artist_id in ambiguous
                    else record.status if record else "missing"
                ),
                "collected_at": record.collected_at if record else None,
                "collection_id": record.collection_id if record else None,
                "tags": record.tags if record else [],
            }
        )
    frame = pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
    output = paths.curated_artist_tags
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial artist tag build exists: {partial}")
    try:
        frame.write_parquet(partial)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-demand Last.fm artist tag enrichment"
    )
    parser.add_argument("action", choices=["collect", "build"])
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh all artist tags"
    )
    args = parser.parse_args()
    if args.refresh and args.action != "collect":
        parser.error("--refresh is only available with collect")
    if args.action == "collect":
        count = collect_artist_tags(refresh=args.refresh)
        print(f"Collected artist tags for {count} artists")
    else:
        path = build_curated_artist_tags()
        print(f"Wrote curated artist tags to {path}")


if __name__ == "__main__":
    main()

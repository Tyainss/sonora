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
    ambiguous_artist_ids,
    artist_catalogue,
    name_key,
)
from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_OUTPUT_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "queried_name": pl.String,
    "reported_name": pl.String,
    "status": pl.String,
    "collected_at": pl.String,
    "collection_id": pl.String,
    "summary": pl.String,
    "content": pl.String,
    "source_url": pl.String,
}
_COMPLETED = {
    "success", "empty", "not_found", "identity_mismatch", "identity_unverified"
}
_STATUS_PRIORITY = {"success": 3, "empty": 2, "not_found": 1}


@dataclass(frozen=True, slots=True)
class _Record:
    artist_id: str
    queried_name: str
    reported_name: str | None
    status: str
    collected_at: str
    collection_id: str
    summary: str | None
    content: str | None
    source_url: str | None


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
    names = [name_key(row["artist_name"]) for row in mapping]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError(f"Duplicate artists in {path}")
    return mapping, refresh


def _extract_description(raw: dict) -> tuple[str | None, str | None, str | None, str | None]:
    if not isinstance(raw, dict) or not isinstance(raw.get("artist"), dict):
        raise ValueError("Invalid Last.fm artist info response")
    artist = raw["artist"]
    reported = artist.get("name")
    if not isinstance(reported, str) or not reported.strip():
        reported = None
    bio = artist.get("bio") or {}
    if not isinstance(bio, dict):
        raise ValueError("Invalid Last.fm artist biography")
    summary = bio.get("summary")
    content = bio.get("content")
    if (summary is not None and not isinstance(summary, str)) or (
        content is not None and not isinstance(content, str)
    ):
        raise ValueError("Invalid Last.fm artist biography text")
    url = artist.get("url")
    if url is not None and not isinstance(url, str):
        raise ValueError("Invalid Last.fm artist URL")
    return (
        reported,
        summary if summary and summary.strip() else None,
        content if content and content.strip() else None,
        url or None,
    )


def _load_collections(paths: DataPaths) -> tuple[list[_Record], list[Path]]:
    base = paths.raw_lastfm_artist_descriptions_dir
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
        if manifest["plan"]["settings"]["operations"] != ["lastfm_info"]:
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
            reported = summary = content = url = None
            if status in {"success", "empty"}:
                reported, summary, content, url = _extract_description(result.raw_response)
                if reported is None:
                    status = "identity_unverified"
                elif name_key(reported) != name_key(name):
                    status = "identity_mismatch"
                elif (status == "success") != bool(summary or content):
                    raise ValueError(f"Biography status differs from payload: {name}")
                if status not in {"success", "empty"}:
                    summary = content = url = None
            records.append(
                _Record(
                    artist_id=row["artist_id"],
                    queried_name=name,
                    reported_name=reported,
                    status=status,
                    collected_at=result.collected_at,
                    collection_id=result.collection_id,
                    summary=summary,
                    content=content,
                    source_url=url,
                )
            )
    collection_names = {path.name for path in base.iterdir() if path.is_dir()}
    mapping_names = {
        path.name.removesuffix(".artists.json") for path in base.glob("*.artists.json")
    }
    if collection_names - mapping_names:
        raise ValueError("Artist description collection is missing its artist-ID mapping")
    return records, unfinished


def _latest_by_artist(
    records: list[_Record], artists: dict[str, str]
) -> dict[str, _Record]:
    latest: dict[str, _Record] = {}
    for record in sorted(records, key=lambda r: (r.collected_at, r.collection_id)):
        canonical = artists.get(record.artist_id)
        if canonical is None or name_key(canonical) != name_key(record.queried_name):
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
            f"Artist descriptions: {summary.total - summary.remaining}/{summary.total}, "
            f"failed={summary.failed}, empty={summary.empty}"
        )


def _lastfm_client() -> LastFMClient:
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LASTFM_API_KEY")
    return LastFMClient(api_key=api_key, user_agent="sonora")


def collect_artist_descriptions(
    *, paths: DataPaths = DEFAULT_DATA_PATHS, refresh: bool = False
) -> int:
    load_dotenv()
    artists = artist_catalogue(paths)
    records, unfinished = _load_collections(paths)
    if len(unfinished) > 1:
        raise RuntimeError("Multiple unfinished artist description collections")

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
                operations=["lastfm_info"],
            )
        lastfm = _lastfm_client()
        args["lastfm"] = lastfm
        args["on_result"] = _report_progress
        summary = collect_artist_metadata(**args)
        if summary.failed or not summary.completed:
            raise RuntimeError(f"Unfinished Last.fm collection: {directory}")
        records, _ = _load_collections(paths)

    latest = _latest_by_artist(records, artists)
    ambiguous = ambiguous_artist_ids(artists)
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
    directory = paths.raw_lastfm_artist_descriptions_dir / batch_name
    _write_mapping(
        directory.with_name(f"{batch_name}.artists.json"), pending, refresh=refresh
    )
    summary = collect_artist_metadata(
        out=directory,
        artists=[ArtistInput(row["artist_name"]) for row in pending],
        operations=["lastfm_info"],
        lastfm=lastfm,
        on_result=_report_progress,
    )
    if summary.failed or not summary.completed:
        raise RuntimeError(f"Last.fm collection has failures: {directory}")
    return len(pending)


def build_curated_artist_descriptions(
    *, paths: DataPaths = DEFAULT_DATA_PATHS
) -> Path:
    artists = artist_catalogue(paths)
    records, unfinished = _load_collections(paths)
    if unfinished:
        raise RuntimeError("Finish or resume artist description collections before curating")
    latest = _latest_by_artist(records, artists)
    ambiguous = ambiguous_artist_ids(artists)
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
                    "ambiguous_name" if artist_id in ambiguous
                    else record.status if record else "missing"
                ),
                "collected_at": record.collected_at if record else None,
                "collection_id": record.collection_id if record else None,
                "summary": record.summary if record else None,
                "content": record.content if record else None,
                "source_url": record.source_url if record else None,
            }
        )
    frame = pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
    output = paths.curated_artist_descriptions
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial artist description build exists: {partial}")
    try:
        frame.write_parquet(partial)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-demand Last.fm artist biography enrichment"
    )
    parser.add_argument("action", choices=["collect", "build"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh and args.action != "collect":
        parser.error("--refresh is only available with collect")
    if args.action == "collect":
        count = collect_artist_descriptions(refresh=args.refresh)
        print(f"Collected descriptions for {count} artists")
    else:
        path = build_curated_artist_descriptions()
        print(f"Wrote curated artist descriptions to {path}")


if __name__ == "__main__":
    main()

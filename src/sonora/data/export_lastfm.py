import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv
from lastfm_export.clients.lastfm import LastFMClient
from lastfm_export.pipelines.lastfm_export import collect_verified_scrobbles

RAW_DIR = Path("data/raw/lastfm")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def export_snapshot(*, cutoff: datetime | None = None) -> Path:
    """Replace current canonical raw snapshot with a strictly verified export."""
    load_dotenv()

    cutoff = cutoff or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")

    cutoff = cutoff.astimezone(UTC).replace(microsecond=0)
    cutoff_unix = int(cutoff.timestamp())
    output_path = RAW_DIR / "scrobbles.ndjson"
    metadata_path = output_path.with_suffix(".integrity.json")
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    metadata_partial_path = metadata_path.with_suffix(".json.partial")

    if partial_path.exists() or metadata_partial_path.exists():
        raise FileExistsError(
            "A partial Last.fm export already exists; inspect or remove it before retrying."
        )

    client = LastFMClient(
        api_key=_required_env("LASTFM_API_KEY"),
        username=_required_env("LASTFM_USERNAME"),
        user_agent="sonora",
    )
    from_unix = client.get_user_registration_unix()
    if from_unix is None:
        raise RuntimeError(
            "Could not determine the Last.fm account registration timestamp for "
            "a verified full-history export."
        )

    scrobbles, reports = collect_verified_scrobbles(
        lastfm=client,
        from_unix=from_unix,
        to_unix=cutoff_unix,
        stop_on_violation=True,
    )
    metadata = _metadata(
        status="ok" if all(report.ok for report in reports) else "failed",
        from_unix=from_unix,
        to_unix=cutoff_unix,
        record_count=len(scrobbles),
        reports=reports,
    )

    if metadata["status"] != "ok":
        failure_path = RAW_DIR / (
            f"scrobbles.integrity.failed_{cutoff:%Y%m%dT%H%M%SZ}.json"
        )
        _write_json(failure_path, metadata, overwrite=False)
        raise RuntimeError(
            "Verified Last.fm export failed integrity checks; the canonical snapshot "
            f"was not modified. Report: {failure_path}"
        )

    _write_snapshot(
        output_path=output_path,
        metadata_path=metadata_path,
        partial_path=partial_path,
        metadata_partial_path=metadata_partial_path,
        scrobbles=scrobbles,
        metadata=metadata,
    )

    print(f"Exported {len(scrobbles):,} verified scrobbles to {output_path}")
    return output_path


def _metadata(*, status: str, from_unix: int, to_unix: int, record_count: int, reports):
    return {
        "status": status,
        "acquisition_mode": "verified",
        "integrity_policy": "strict",
        "lastfm_export_version": _lastfm_export_version(),
        "from_unix": from_unix,
        "to_unix": to_unix,
        "record_count": record_count,
        "windows": [report.to_record() for report in reports],
    }


def _lastfm_export_version() -> str:
    try:
        return version("lastfm-export")
    except PackageNotFoundError:
        return "unknown"


def _write_snapshot(
    *,
    output_path: Path,
    metadata_path: Path,
    partial_path: Path,
    metadata_partial_path: Path,
    scrobbles,
    metadata,
) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with partial_path.open("x", encoding="utf-8", newline="\n") as file:
            for scrobble in scrobbles:
                file.write(
                    json.dumps(scrobble.to_record(include_raw=True), ensure_ascii=False)
                )
                file.write("\n")

        with metadata_partial_path.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")

        os.replace(partial_path, output_path)
        os.replace(metadata_partial_path, metadata_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        metadata_partial_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    export_snapshot()

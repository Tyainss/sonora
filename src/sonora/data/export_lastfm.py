import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from lastfm_export.clients.lastfm import LastFMClient
from lastfm_export.pipelines.lastfm_export import (
    VerifiedProgress,
    collect_verified_scrobbles,
)

RAW_DIR = Path("data/raw/lastfm")


@dataclass
class _ProgressReporter:
    """Compact interactive progress modelled on lastfm-export's CLI reporter."""

    interval_seconds: float = 15
    stream: TextIO | None = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._stream = self.stream or sys.stderr
        self._enabled = self._stream.isatty()
        self._last_update: float | None = None
        self._live_line = False

    def start(self, message: str) -> None:
        self._write_line(message)

    def update(self, message: str, *, force: bool = False) -> None:
        if not self._enabled:
            return
        now = self.clock()
        if (
            not force
            and self._last_update is not None
            and now - self._last_update < self.interval_seconds
        ):
            return
        self._last_update = now
        self._stream.write(f"\r{message}")
        self._stream.flush()
        self._live_line = True

    def milestone(self, message: str) -> None:
        self._write_line(message)

    def finish(self, message: str) -> None:
        self._write_line(message)

    def close(self) -> None:
        if self._enabled and self._live_line:
            self._stream.write("\n")
            self._stream.flush()
            self._live_line = False

    def _write_line(self, message: str) -> None:
        if not self._enabled:
            return
        self.close()
        self._stream.write(f"{message}\n")
        self._stream.flush()


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

    progress = _ProgressReporter()
    progress.start(
        "Starting verified export: "
        f"{_utc_date(from_unix)} to {_utc_date(cutoff_unix)} UTC"
    )

    def on_window_start(day: date, days_checked: int, tracks_collected: int) -> None:
        progress.update(
            f"Working: {days_checked:,} days checked, {tracks_collected:,} scrobbles "
            f"collected; fetching {day.isoformat()}"
        )

    def on_window_complete(event: VerifiedProgress) -> None:
        if _is_completed_full_quarter(
            day=event.day, from_unix=from_unix, to_unix=cutoff_unix
        ):
            progress.milestone(
                f"Completed {_quarter_label(event.day)}: {event.days_checked:,} days "
                f"checked, {event.tracks_collected:,} scrobbles collected"
            )

    try:
        scrobbles, reports = collect_verified_scrobbles(
            lastfm=client,
            from_unix=from_unix,
            to_unix=cutoff_unix,
            stop_on_violation=True,
            on_window_start=on_window_start,
            on_window_complete=on_window_complete,
        )
    finally:
        progress.close()
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
        progress.finish(
            f"Stopped verified export after {len(scrobbles):,} scrobbles from "
            f"{len(reports):,} days; integrity: failed"
        )
        raise RuntimeError(
            "Verified Last.fm export failed integrity checks; the canonical snapshot "
            f"was not modified. Report: {failure_path}"
        )

    progress.update(f"Writing scrobbles to {output_path}", force=True)
    _write_snapshot(
        output_path=output_path,
        metadata_path=metadata_path,
        partial_path=partial_path,
        metadata_partial_path=metadata_partial_path,
        scrobbles=scrobbles,
        metadata=metadata,
    )

    progress.finish(
        f"Completed verified export: {len(scrobbles):,} scrobbles from "
        f"{len(reports):,} days; integrity: ok"
    )
    return output_path


def _utc_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).date().isoformat()


def _quarter_label(day: date) -> str:
    return f"{day.year} Q{((day.month - 1) // 3) + 1}"


def _is_completed_full_quarter(*, day: date, from_unix: int, to_unix: int) -> bool:
    quarter_month = ((day.month - 1) // 3) * 3 + 1
    if day != date(day.year, quarter_month, 1):
        return False
    quarter_start = int(datetime(day.year, quarter_month, 1, tzinfo=UTC).timestamp())
    next_quarter = (
        datetime(day.year + 1, 1, 1, tzinfo=UTC)
        if quarter_month == 10
        else datetime(day.year, quarter_month + 3, 1, tzinfo=UTC)
    )
    quarter_end = int(next_quarter.timestamp()) - 1
    return from_unix <= quarter_start and to_unix >= quarter_end


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

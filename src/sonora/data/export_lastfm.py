import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from lastfm_export.clients.lastfm import LastFMClient
from lastfm_export.pipelines.lastfm_export import export_scrobbles

RAW_DIR = Path("data/raw/lastfm")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def export_snapshot(*, cutoff: datetime | None = None) -> Path:
    """Export a complete raw Last.fm listening-history snapshot."""
    load_dotenv()

    cutoff = cutoff or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")

    cutoff = cutoff.astimezone(UTC).replace(microsecond=0)
    cutoff_unix = int(cutoff.timestamp())

    output_path = RAW_DIR / "scrobbles.ndjson"
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")

    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"Export already exists: {output_path}")

    client = LastFMClient(
        api_key=_required_env("LASTFM_API_KEY"),
        username=_required_env("LASTFM_USERNAME"),
        user_agent="sonora",
    )

    scrobbles = export_scrobbles(
        lastfm=client,
        to_unix=cutoff_unix,
    )

    count = 0

    try:
        partial_path.parent.mkdir(parents=True, exist_ok=True)

        with partial_path.open("x", encoding="utf-8", newline="\n") as file:
            for scrobble in scrobbles:
                record = scrobble.to_record(include_raw=True)
                file.write(json.dumps(record, ensure_ascii=False))
                file.write("\n")

                count += 1
                if count % 10_000 == 0:
                    print(f"Exported {count:,} scrobbles...")

        partial_path.replace(output_path)

    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    print(f"Exported {count:,} scrobbles to {output_path}")
    return output_path


if __name__ == "__main__":
    export_snapshot()

import json
from datetime import UTC, datetime
from io import StringIO

import pytest
from lastfm_export.integrity import WindowReport
from lastfm_export.models import Scrobble

from sonora.data import export_lastfm
from sonora.data.paths import DataPaths


class _TTYStream(StringIO):
    def isatty(self) -> bool:
        return True


def _scrobble() -> Scrobble:
    return Scrobble(
        artist_name="Artist",
        track_name="Track",
        album_name="Album",
        timestamp_unix=1_584_617_400,
        mbid="track-mbid",
        raw={"name": "Track", "artist": {"#text": "Artist"}},
    )


def _report(*violations: str) -> WindowReport:
    return WindowReport(
        from_unix=1,
        to_unix=2,
        api_total=1,
        materialized_count=1,
        page_count=1,
        violations=list(violations),
    )


def _configure_export(monkeypatch, tmp_path, *, registration_unix=1):
    monkeypatch.setenv("LASTFM_API_KEY", "test-key")
    monkeypatch.setenv("LASTFM_USERNAME", "test-user")
    paths = DataPaths(data_dir=tmp_path / "data")

    class FakeLastFMClient:
        def __init__(self, **kwargs):
            pass

        def get_user_registration_unix(self):
            return registration_unix

    monkeypatch.setattr(export_lastfm, "LastFMClient", FakeLastFMClient)
    return paths


def test_progress_reporter_throttles_live_updates_and_keeps_milestones():
    stream = _TTYStream()
    timestamps = iter([0.0, 1.0, 16.0])
    reporter = export_lastfm._ProgressReporter(
        stream=stream,
        clock=lambda: next(timestamps),
    )

    reporter.start("Starting verified export")
    reporter.update("Working: first")
    reporter.update("Working: skipped")
    reporter.update("Working: next")
    reporter.milestone("Completed 2020 Q1")
    reporter.finish("Completed verified export")

    assert stream.getvalue() == (
        "Starting verified export\n"
        "\rWorking: first\rWorking: next\n"
        "Completed 2020 Q1\n"
        "Completed verified export\n"
    )


def test_export_snapshot_writes_verified_raw_scrobbles_and_metadata(
    tmp_path, monkeypatch
):
    cutoff = datetime(2020, 3, 19, 12, 30, tzinfo=UTC)
    paths = _configure_export(monkeypatch, tmp_path, registration_unix=123)
    captured = {}

    def fake_collect_verified_scrobbles(**kwargs):
        captured.update(kwargs)
        return [_scrobble()], [_report()]

    monkeypatch.setattr(
        export_lastfm, "collect_verified_scrobbles", fake_collect_verified_scrobbles
    )
    monkeypatch.setattr(export_lastfm, "_lastfm_export_version", lambda: "0.3.0")

    output = export_lastfm.export_snapshot(cutoff=cutoff, paths=paths)

    assert output == paths.raw_lastfm_scrobbles
    assert captured["from_unix"] == 123
    assert captured["to_unix"] == int(cutoff.timestamp())
    assert captured["stop_on_violation"] is True

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["artist_name"] == "Artist"
    assert record["raw"] == {"name": "Track", "artist": {"#text": "Artist"}}

    metadata = json.loads(paths.raw_lastfm_integrity.read_text())
    assert metadata["status"] == "ok"
    assert metadata["acquisition_mode"] == "verified"
    assert metadata["integrity_policy"] == "strict"
    assert metadata["lastfm_export_version"] == "0.3.0"
    assert metadata["record_count"] == 1
    assert metadata["windows"] == [_report().to_record()]


def test_integrity_failure_preserves_canonical_snapshot_and_writes_report(
    tmp_path, monkeypatch
):
    cutoff = datetime(2020, 3, 19, 12, 30, tzinfo=UTC)
    paths = _configure_export(monkeypatch, tmp_path)
    output = paths.raw_lastfm_scrobbles
    metadata = paths.raw_lastfm_integrity
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"track_name":"old"}\n', encoding="utf-8")
    metadata.write_text('{"status":"ok"}\n', encoding="utf-8")
    monkeypatch.setattr(
        export_lastfm,
        "collect_verified_scrobbles",
        lambda **kwargs: ([_scrobble()], [_report("exact overlap")]),
    )

    with pytest.raises(RuntimeError, match="failed integrity checks"):
        export_lastfm.export_snapshot(cutoff=cutoff, paths=paths)

    assert output.read_text(encoding="utf-8") == '{"track_name":"old"}\n'
    assert metadata.read_text(encoding="utf-8") == '{"status":"ok"}\n'
    failure = paths.raw_lastfm_dir / "scrobbles.integrity.failed_20200319T123000Z.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "failed"


def test_export_snapshot_rejects_naive_cutoff(tmp_path, monkeypatch):
    paths = _configure_export(monkeypatch, tmp_path)
    naive_cutoff = datetime(2020, 3, 19, 12, 30, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(ValueError, match="timezone-aware"):
        export_lastfm.export_snapshot(cutoff=naive_cutoff, paths=paths)


def test_export_snapshot_requires_registration_timestamp(tmp_path, monkeypatch):
    paths = _configure_export(monkeypatch, tmp_path, registration_unix=None)

    with pytest.raises(RuntimeError, match="registration timestamp"):
        export_lastfm.export_snapshot(
            cutoff=datetime(2020, 3, 19, tzinfo=UTC), paths=paths
        )


@pytest.mark.parametrize(
    "partial_name",
    ["scrobbles.ndjson.partial", "scrobbles.integrity.json.partial"],
)
def test_export_snapshot_rejects_existing_partial_file(
    tmp_path, monkeypatch, partial_name
):
    paths = _configure_export(monkeypatch, tmp_path)
    paths.raw_lastfm_dir.mkdir(parents=True, exist_ok=True)
    (paths.raw_lastfm_dir / partial_name).write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="partial Last.fm export"):
        export_lastfm.export_snapshot(
            cutoff=datetime(2020, 3, 19, tzinfo=UTC), paths=paths
        )

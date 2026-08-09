import json
from datetime import UTC, datetime

from lastfm_export.models import Scrobble

from sonora.data import export_lastfm


def test_export_snapshot_writes_raw_scrobbles(tmp_path, monkeypatch):
    cutoff = datetime(2020, 3, 19, 12, 30, tzinfo=UTC)

    monkeypatch.setenv("LASTFM_API_KEY", "test-key")
    monkeypatch.setenv("LASTFM_USERNAME", "test-user")
    monkeypatch.setattr(export_lastfm, "RAW_DIR", tmp_path)

    class FakeLastFMClient:
        def __init__(self, **kwargs):
            pass

    def fake_export_scrobbles(*, lastfm, to_unix):
        assert to_unix == int(cutoff.timestamp())

        yield Scrobble(
            artist_name="Artist",
            track_name="Track",
            album_name="Album",
            timestamp_unix=1_584_617_400,
            mbid="track-mbid",
            raw={"name": "Track", "artist": {"#text": "Artist"}},
        )

    monkeypatch.setattr(export_lastfm, "LastFMClient", FakeLastFMClient)
    monkeypatch.setattr(
        export_lastfm,
        "export_scrobbles",
        fake_export_scrobbles,
    )

    output = export_lastfm.export_snapshot(cutoff=cutoff)

    assert output.name == "scrobbles.ndjson"

    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["artist_name"] == "Artist"
    assert record["raw"] == {
        "name": "Track",
        "artist": {"#text": "Artist"},
    }

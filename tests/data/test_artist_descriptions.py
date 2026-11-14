import json

import polars as pl
import pytest
from lastfm_export.errors import HttpRequestError
from lastfm_export.models import ArtistMetadataResponse

from sonora.data import artist_descriptions
from sonora.data.paths import DataPaths


def _catalogue(paths, rows):
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows, schema={"artist_id": pl.String, "canonical_name": pl.String}
    ).write_parquet(paths.curated_artists)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = DataPaths(data_dir=tmp_path / "data")
    _catalogue(paths, [("id-a", "Alpha"), ("id-b", "Beta"), ("id-c", "Gamma")])
    monkeypatch.setenv("LASTFM_API_KEY", "test-key")
    calls = []
    responses = {}
    interrupted = {"name": None}

    class FakeClient:
        base_url = "https://ws.audioscrobbler.com/2.0/"

        def __init__(self, *, api_key, user_agent):
            self.api_key = api_key

        def get_artist_info(self, *, artist_name, mbid=None, autocorrect=False, lang="en"):
            assert lang == "en" and not autocorrect
            calls.append(artist_name)
            if interrupted["name"] == artist_name:
                interrupted["name"] = None
                raise KeyboardInterrupt
            outcome = responses.get(artist_name)
            if outcome == "failed":
                raise HttpRequestError("GET", self.base_url, 503, "Service unavailable")
            if outcome == "not_found":
                raise HttpRequestError(
                    "GET", self.base_url, 200, "Artist not found",
                    payload={"error": 6, "message": "Artist not found"},
                    provider_code=6,
                )
            text = "" if artist_name == "Beta" or outcome == "empty" else "Folk <a>music</a>"
            reported = "Different artist" if outcome == "mismatch" else artist_name
            block = {
                "artist": {
                    "name": reported,
                    "url": f"https://www.last.fm/music/{artist_name}",
                    "bio": {"summary": text, "content": text + " details" if text else ""},
                }
            }
            if outcome == "unverified":
                block["artist"].pop("name")
            return ArtistMetadataResponse(
                status="success" if text else "empty",
                raw_response=block,
                http_status=200,
                wire_attempts=1,
            )

    monkeypatch.setattr(artist_descriptions, "LastFMClient", FakeClient)
    return paths, calls, responses, interrupted


def test_incremental_collection_and_curated_biographies(setup):
    paths, calls, _, _ = setup
    assert artist_descriptions.collect_artist_descriptions(paths=paths) == 3
    assert calls == ["Alpha", "Beta", "Gamma"]
    assert artist_descriptions.collect_artist_descriptions(paths=paths) == 0
    frame = pl.read_parquet(
        artist_descriptions.build_curated_artist_descriptions(paths=paths)
    ).sort("artist_id")
    assert frame.get_column("status").to_list() == ["success", "empty", "success"]
    assert frame.item(0, "summary") == "Folk <a>music</a>"
    assert frame.item(0, "content") == "Folk <a>music</a> details"
    assert frame.item(1, "summary") is None
    assert frame.item(1, "content") is None
    assert frame.item(0, "queried_name") == "Alpha"


def test_invalid_refresh_preserves_earlier_biography(setup):
    paths, _, responses, _ = setup
    artist_descriptions.collect_artist_descriptions(paths=paths)
    before = pl.read_parquet(artist_descriptions.build_curated_artist_descriptions(paths=paths))
    original = before.filter(pl.col("artist_id") == "id-a").row(0, named=True)
    for outcome in ("mismatch", "unverified", "empty", "not_found", "failed"):
        responses["Alpha"] = outcome
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="failures"):
                artist_descriptions.collect_artist_descriptions(paths=paths, refresh=True)
        else:
            artist_descriptions.collect_artist_descriptions(paths=paths, refresh=True)
        after = pl.read_parquet(
            artist_descriptions.build_curated_artist_descriptions(paths=paths)
        )
        assert after.filter(pl.col("artist_id") == "id-a").row(0, named=True) == original


def test_interrupted_collection_resumes_without_duplicates(setup):
    paths, calls, _, interrupted = setup
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_descriptions.collect_artist_descriptions(paths=paths)
    calls.clear()
    assert artist_descriptions.collect_artist_descriptions(paths=paths) == 0
    assert calls == ["Beta", "Gamma"]
    assert pl.read_parquet(
        artist_descriptions.build_curated_artist_descriptions(paths=paths)
    ).height == 3


def test_interrupted_refresh_does_not_start_second_full_refresh(setup):
    paths, calls, _, interrupted = setup
    artist_descriptions.collect_artist_descriptions(paths=paths)
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_descriptions.collect_artist_descriptions(paths=paths, refresh=True)
    calls.clear()
    assert artist_descriptions.collect_artist_descriptions(paths=paths, refresh=True) == 0
    assert calls == ["Beta", "Gamma"]


def test_renamed_artist_is_refetched_and_ambiguous_name_is_flagged(setup):
    paths, calls, _, _ = setup
    artist_descriptions.collect_artist_descriptions(paths=paths)
    _catalogue(
        paths,
        [("id-a", "Alpha"), ("id-b", "Beta Renamed"),
         ("id-c", "Gamma"), ("id-d", "alpha")],
    )
    calls.clear()
    assert artist_descriptions.collect_artist_descriptions(paths=paths) == 1
    assert calls == ["Beta Renamed"]
    frame = pl.read_parquet(artist_descriptions.build_curated_artist_descriptions(paths=paths))
    statuses = dict(frame.select("artist_id", "status").iter_rows())
    assert statuses == {
        "id-a": "ambiguous_name", "id-b": "success", "id-c": "success",
        "id-d": "ambiguous_name",
    }


def test_unfinished_result_tail_can_be_recovered(setup):
    paths, calls, _, interrupted = setup
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_descriptions.collect_artist_descriptions(paths=paths)
    directory = next(p for p in paths.raw_lastfm_artist_descriptions_dir.iterdir() if p.is_dir())
    with (directory / "results.ndjson").open("ab") as stream:
        stream.write(b'{"incomplete":')
    calls.clear()
    assert artist_descriptions.collect_artist_descriptions(paths=paths) == 0
    assert calls == ["Beta", "Gamma"]
    assert list(directory.glob("results.tail-*.bin"))


def test_mismatched_mapping_rejected(setup):
    paths, _, _, _ = setup
    artist_descriptions.collect_artist_descriptions(paths=paths)
    mapping_path = next(paths.raw_lastfm_artist_descriptions_dir.glob("*.artists.json"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["artists"][0]["artist_name"] = "Wrong Artist"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping disagrees"):
        artist_descriptions.build_curated_artist_descriptions(paths=paths)


def test_without_collections_every_artist_is_missing(setup):
    paths, _, _, _ = setup
    frame = pl.read_parquet(artist_descriptions.build_curated_artist_descriptions(paths=paths))
    assert frame.height == 3
    assert frame.get_column("status").to_list() == ["missing"] * 3

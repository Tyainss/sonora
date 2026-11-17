import json

import polars as pl
import pytest
from lastfm_export.errors import HttpRequestError
from lastfm_export.models import ArtistMetadataResponse

from sonora.data import artist_tags
from sonora.data.paths import DataPaths


def _catalogue(paths, rows):
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={"artist_id": pl.String, "canonical_name": pl.String},
        orient="row",
    ).write_parquet(paths.curated_artists)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = DataPaths(data_dir=tmp_path / "data")
    _catalogue(
        paths,
        [
            ("id-a", "Alpha"),
            ("id-b", "Beta"),
            ("id-c", "Gamma"),
        ],
    )
    monkeypatch.setenv("LASTFM_API_KEY", "fake-key")
    calls = []
    interrupted = {"name": None}
    responses = {}

    class FakeClient:
        base_url = "https://ws.audioscrobbler.com/2.0/"

        def __init__(self, *, api_key, user_agent):
            self.api_key = api_key

        def get_artist_top_tags(self, *, artist_name, mbid=None, autocorrect=False):
            calls.append(artist_name)
            if interrupted["name"] == artist_name:
                interrupted["name"] = None
                raise KeyboardInterrupt
            outcome = responses.get(artist_name)
            if outcome == "failed":
                raise HttpRequestError(
                    "GET", self.base_url, 503, "Service unavailable"
                )
            if artist_name == "Beta" or outcome == "empty":
                tags = []
            else:
                tags = [
                    {"name": "Indie Rock", "count": "100"},
                    {"name": "indie rock", "count": "10"},
                    {"name": "Folk", "count": "25"},
                ]
            reported = (
                "Someone Else"
                if artist_name == "Gamma" or outcome == "mismatch"
                else artist_name
            )
            attributes = {} if outcome == "unverified" else {"artist": reported}
            return ArtistMetadataResponse(
                status="success" if tags else "empty",
                raw_response={
                    "toptags": {"@attr": attributes, "tag": tags}
                },
                http_status=200,
                wire_attempts=1,
            )

    monkeypatch.setattr(artist_tags, "LastFMClient", FakeClient)
    return paths, calls, interrupted, responses


def test_collect_and_build_are_incremental_and_preserve_missing_status(setup, monkeypatch):
    paths, calls, _, _ = setup
    assert artist_tags.collect_artist_tags(paths=paths) == 3
    assert calls == ["Alpha", "Beta", "Gamma"]

    output = artist_tags.build_curated_artist_tags(paths=paths)
    frame = pl.read_parquet(output).sort("artist_id")
    assert frame.get_column("artist_id").to_list() == ["id-a", "id-b", "id-c"]
    assert frame.get_column("status").to_list() == [
        "success", "empty", "identity_mismatch"
    ]
    assert frame.row(0, named=True)["tags"] == [
        {"name": "Indie Rock", "weight": 100},
        {"name": "Folk", "weight": 25},
    ]
    assert frame.row(1, named=True)["tags"] == []
    assert frame.row(2, named=True)["tags"] == []

    # Identity mismatches are held for review, rather than repeatedly fetched.
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == []

    # A rerun with a healthy catalogue never repeats successful queries.
    _catalogue(paths, [("id-a", "Alpha"), ("id-b", "Beta")])
    calls.clear()
    monkeypatch.delenv("LASTFM_API_KEY")
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == []


def test_new_and_renamed_artists_are_collected_without_repeating_existing(setup):
    paths, calls, _, _ = setup
    artist_tags.collect_artist_tags(paths=paths)
    _catalogue(
        paths,
        [
            ("id-a", "Alpha"),
            ("id-b", "Beta Renamed"),
            ("id-c", "Gamma"),
            ("id-d", "Delta"),
        ],
    )
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 2
    assert calls == ["Beta Renamed", "Delta"]
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    assert frame.filter(pl.col("artist_id") == "id-b").row(0, named=True)[
        "queried_name"
    ] == "Beta Renamed"


def test_case_only_rename_does_not_repeat_api_requests(setup):
    paths, calls, _, _ = setup
    artist_tags.collect_artist_tags(paths=paths)
    _catalogue(
        paths,
        [("id-a", "alpha"), ("id-b", "Beta"), ("id-c", "Gamma")],
    )
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == []
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    alpha = frame.filter(pl.col("artist_id") == "id-a").row(0, named=True)
    assert alpha["status"] == "success"
    assert alpha["canonical_name"] == "alpha"
    assert alpha["queried_name"] == "Alpha"


def test_refresh_explicitly_requests_all_artists(setup):
    paths, calls, _, _ = setup
    artist_tags.collect_artist_tags(paths=paths)
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths, refresh=True) == 3
    assert calls == ["Alpha", "Beta", "Gamma"]


def test_interrupted_collection_is_resumed_without_repeating_completed_requests(setup):
    paths, calls, interrupted, _ = setup
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_tags.collect_artist_tags(paths=paths)

    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == ["Beta", "Gamma"]
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    assert frame.height == 3


def test_invalid_artist_mapping_is_rejected_before_curation(setup):
    paths, _, _, _ = setup
    artist_tags.collect_artist_tags(paths=paths)
    mapping_path = next(paths.raw_lastfm_artist_tags_dir.glob("*.artists.json"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["artists"][0]["artist_name"] = "Wrong Artist"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping disagrees"):
        artist_tags.build_curated_artist_tags(paths=paths)


def test_without_collections_build_has_one_missing_row_per_artist(setup):
    paths, _, _, _ = setup
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    assert frame.height == 3
    assert frame.get_column("status").to_list() == ["missing"] * 3


def test_ambiguous_names_are_excluded_without_blocking_other_artists(setup):
    paths, calls, _, _ = setup
    _catalogue(
        paths,
        [("id-a", "Alpha"), ("id-b", "alpha"), ("id-c", "Gamma")],
    )
    assert artist_tags.collect_artist_tags(paths=paths) == 1
    assert calls == ["Gamma"]
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    statuses = dict(frame.select("artist_id", "status").iter_rows())
    assert statuses == {
        "id-a": "ambiguous_name",
        "id-b": "ambiguous_name",
        "id-c": "identity_mismatch",
    }


def test_invalid_or_empty_refresh_preserves_valid_metadata(setup):
    paths, _, _, responses = setup
    artist_tags.collect_artist_tags(paths=paths)
    initial = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    initial_alpha = initial.filter(pl.col("artist_id") == "id-a").row(0, named=True)

    for outcome in ("mismatch", "unverified", "empty", "failed"):
        responses["Alpha"] = outcome
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="failures"):
                artist_tags.collect_artist_tags(paths=paths, refresh=True)
        else:
            artist_tags.collect_artist_tags(paths=paths, refresh=True)
        updated = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
        alpha = updated.filter(pl.col("artist_id") == "id-a").row(0, named=True)
        assert alpha == initial_alpha


def test_interrupted_refresh_does_not_start_a_second_full_refresh(setup):
    paths, calls, interrupted, _ = setup
    artist_tags.collect_artist_tags(paths=paths)
    calls.clear()
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_tags.collect_artist_tags(paths=paths, refresh=True)
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths, refresh=True) == 0
    assert calls == ["Beta", "Gamma"]


def test_resume_recovers_an_incomplete_result_tail(setup):
    paths, calls, interrupted, _ = setup
    interrupted["name"] = "Beta"
    with pytest.raises(KeyboardInterrupt):
        artist_tags.collect_artist_tags(paths=paths)
    collection_dir = next(
        path for path in paths.raw_lastfm_artist_tags_dir.iterdir() if path.is_dir()
    )
    with (collection_dir / "results.ndjson").open("ab") as stream:
        stream.write(b'{"incomplete":')
    calls.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == ["Beta", "Gamma"]
    assert list(collection_dir.glob("results.tail-*.bin"))


def test_failed_request_can_be_collected_on_next_run(setup):
    paths, calls, _, responses = setup
    responses["Alpha"] = "failed"
    with pytest.raises(RuntimeError, match="failures"):
        artist_tags.collect_artist_tags(paths=paths)
    calls.clear()
    responses.clear()
    assert artist_tags.collect_artist_tags(paths=paths) == 1
    assert calls == ["Alpha"]
    frame = pl.read_parquet(artist_tags.build_curated_artist_tags(paths=paths))
    assert frame.filter(pl.col("artist_id") == "id-a").item(0, "status") == "success"


def test_mapping_is_written_before_network_collection_and_recoverable(setup, monkeypatch):
    paths, calls, _, _ = setup
    real_collect = artist_tags.collect_artist_metadata

    def interrupted_collect(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(artist_tags, "collect_artist_metadata", interrupted_collect)
    with pytest.raises(KeyboardInterrupt):
        artist_tags.collect_artist_tags(paths=paths)
    assert list(paths.raw_lastfm_artist_tags_dir.glob("*.artists.json"))

    monkeypatch.setattr(artist_tags, "collect_artist_metadata", real_collect)
    assert artist_tags.collect_artist_tags(paths=paths) == 0
    assert calls == ["Alpha", "Beta", "Gamma"]

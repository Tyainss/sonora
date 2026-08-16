from sonora.data.paths import DataPaths


def test_data_paths_are_derived_from_one_data_root(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "sonora-data")

    assert paths.raw_lastfm_dir == tmp_path / "sonora-data/raw/lastfm"
    assert paths.raw_lastfm_scrobbles == (
        tmp_path / "sonora-data/raw/lastfm/scrobbles.ndjson"
    )
    assert paths.raw_lastfm_integrity == (
        tmp_path / "sonora-data/raw/lastfm/scrobbles.integrity.json"
    )
    assert paths.interim_dir == tmp_path / "sonora-data/interim"
    assert paths.listening_events_clean == (
        tmp_path / "sonora-data/interim/listening_events_clean.parquet"
    )
    assert paths.curated_dir == tmp_path / "sonora-data/curated"

    assert paths.curated_artists == (tmp_path / "sonora-data/curated/artists.parquet")
    assert paths.curated_artist_aliases == (
        tmp_path / "sonora-data/curated/artist_aliases.parquet"
    )

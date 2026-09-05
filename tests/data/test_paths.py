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
    assert paths.curated_tracks == (tmp_path / "sonora-data/curated/tracks.parquet")
    assert paths.curated_track_aliases == (
        tmp_path / "sonora-data/curated/track_aliases.parquet"
    )
    assert paths.curated_listening_events == (
        tmp_path / "sonora-data/curated/listening_events.parquet"
    )
    assert paths.breakout_dir == tmp_path / "sonora-data/processed/breakout"
    assert paths.breakout_events == paths.breakout_dir / "first_breakout_events.parquet"
    assert paths.daily_breakout_targets == paths.breakout_dir / "daily_targets.parquet"
    assert paths.breakout_build_metadata == paths.breakout_dir / "build_metadata.json"

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem contract for Sonora datasets."""

    data_dir: Path = Path("data")

    @property
    def raw_lastfm_dir(self) -> Path:
        return self.data_dir / "raw" / "lastfm"

    @property
    def raw_lastfm_scrobbles(self) -> Path:
        return self.raw_lastfm_dir / "scrobbles.ndjson"

    @property
    def raw_lastfm_integrity(self) -> Path:
        return self.raw_lastfm_dir / "scrobbles.integrity.json"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def listening_events_clean(self) -> Path:
        return self.interim_dir / "listening_events_clean.parquet"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def curated_artists(self) -> Path:
        return self.curated_dir / "artists.parquet"

    @property
    def curated_artist_aliases(self) -> Path:
        return self.curated_dir / "artist_aliases.parquet"

    @property
    def curated_tracks(self) -> Path:
        return self.curated_dir / "tracks.parquet"

    @property
    def curated_track_aliases(self) -> Path:
        return self.curated_dir / "track_aliases.parquet"


DEFAULT_DATA_PATHS = DataPaths()

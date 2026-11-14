import polars as pl

from sonora.data.paths import DataPaths


def artist_catalogue(paths: DataPaths) -> dict[str, str]:
    artists = pl.read_parquet(paths.curated_artists)
    if dict(artists.schema) != {"artist_id": pl.String, "canonical_name": pl.String}:
        raise ValueError("Unexpected canonical artist schema")
    if artists.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError("Canonical artist IDs and names must not be null")
    if artists.get_column("artist_id").n_unique() != artists.height:
        raise ValueError("Duplicate canonical artist IDs")
    names = artists.get_column("canonical_name").to_list()
    if any(not name.strip() for name in names):
        raise ValueError("Empty canonical artist name")
    return dict(artists.select("artist_id", "canonical_name").iter_rows())


def name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def ambiguous_artist_ids(artists: dict[str, str]) -> set[str]:
    by_name: dict[str, str] = {}
    ambiguous = set()
    for artist_id, name in artists.items():
        key = name_key(name)
        if key in by_name:
            ambiguous.update((artist_id, by_name[key]))
        else:
            by_name[key] = artist_id
    return ambiguous

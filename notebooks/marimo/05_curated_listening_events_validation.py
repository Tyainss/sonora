import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curated listening events

    Builds the final event table and checks that canonical IDs were added without losing or multiplying listens.
    """)
    return


@app.cell
def _():
    import polars as pl

    from sonora.data.curate_listening_events import build_curated_listening_events
    from sonora.data.paths import DEFAULT_DATA_PATHS

    EXPECTED_COLUMNS = [
        "user_id",
        "listened_at",
        "artist_id",
        "track_id",
        "album_name",
        "source",
    ]
    paths = DEFAULT_DATA_PATHS
    return EXPECTED_COLUMNS, build_curated_listening_events, paths, pl


@app.cell
def _(mo, paths):
    _required_paths = [
        paths.listening_events_clean,
        paths.curated_artists,
        paths.curated_artist_aliases,
        paths.curated_tracks,
        paths.curated_track_aliases,
    ]
    _missing_paths = [path for path in _required_paths if not path.is_file()]
    mo.stop(
        bool(_missing_paths),
        mo.callout(
            "Missing required datasets: "
            + ", ".join(f"`{path}`" for path in _missing_paths),
            kind="danger",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build
    """)
    return


@app.cell
def _(mo):
    build_button = mo.ui.run_button(
        label="Build curated listening events",
        kind="success",
    )
    build_button
    return (build_button,)


@app.cell
def _(build_button, build_curated_listening_events, mo, paths):
    mo.stop(not build_button.value)

    output_path = build_curated_listening_events(paths=paths)
    mo.md(f"Built `{output_path}`.")
    return (output_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checks
    """)
    return


@app.cell
def _(output_path, paths, pl):
    clean_events = pl.scan_parquet(paths.listening_events_clean)
    curated_events = pl.scan_parquet(output_path)
    artists = pl.scan_parquet(paths.curated_artists)
    tracks = pl.scan_parquet(paths.curated_tracks)
    return artists, clean_events, curated_events, tracks


@app.cell
def _(clean_events, curated_events, mo, pl):
    clean_count = clean_events.select(pl.len()).collect().item()
    curated_count = curated_events.select(pl.len()).collect().item()
    artists_used = curated_events.select(pl.col("artist_id").n_unique()).collect().item()
    tracks_used = curated_events.select(pl.col("track_id").n_unique()).collect().item()

    mo.md(f"""
    - **Clean events:** {clean_count:,}
    - **Curated events:** {curated_count:,}
    - **Artists referenced:** {artists_used:,}
    - **Tracks referenced:** {tracks_used:,}

    The event count should stay exactly the same. Canonicalization changes identity fields, not the listening history itself.
    """)
    return clean_count, curated_count


@app.cell
def _(curated_events, pl):
    _schema = curated_events.collect_schema()
    schema_summary = pl.DataFrame(
        {
            "column": _schema.names(),
            "dtype": [str(dtype) for dtype in _schema.dtypes()],
        }
    )
    schema_summary
    return


@app.cell
def _(curated_events, pl):
    null_counts = curated_events.select(pl.all().null_count()).collect().row(
        0, named=True
    )
    null_summary = pl.DataFrame(
        {
            "column": list(null_counts),
            "null_count": list(null_counts.values()),
        }
    )
    null_summary
    return (null_counts,)


@app.cell
def _(clean_events, curated_events, pl):
    clean_groups = (
        clean_events.group_by("user_id", "source")
        .agg(pl.len().alias("clean_events"))
        .collect()
    )
    curated_groups = (
        curated_events.group_by("user_id", "source")
        .agg(pl.len().alias("curated_events"))
        .collect()
    )
    user_source_counts = (
        clean_groups.join(
            curated_groups,
            on=["user_id", "source"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            (pl.col("clean_events") == pl.col("curated_events")).alias("matches")
        )
        .sort("user_id", "source")
    )
    user_source_counts
    return (user_source_counts,)


@app.cell
def _(artists, curated_events, pl, tracks):
    unknown_artist_count = (
        curated_events.select("artist_id")
        .unique()
        .join(artists.select("artist_id"), on="artist_id", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    unknown_track_count = (
        curated_events.select("track_id")
        .unique()
        .join(tracks.select("track_id"), on="track_id", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    wrong_track_artist_count = (
        curated_events.select("artist_id", "track_id")
        .unique()
        .join(
            tracks.select(
                "track_id",
                pl.col("artist_id").alias("track_artist_id"),
            ),
            on="track_id",
            how="left",
            validate="m:1",
        )
        .filter(pl.col("artist_id") != pl.col("track_artist_id"))
        .select(pl.len())
        .collect()
        .item()
    )

    reference_summary = pl.DataFrame(
        {
            "check": [
                "Unknown artist IDs",
                "Unknown track IDs",
                "Tracks linked to the wrong artist",
            ],
            "count": [
                unknown_artist_count,
                unknown_track_count,
                wrong_track_artist_count,
            ],
        }
    )
    reference_summary
    return unknown_artist_count, unknown_track_count, wrong_track_artist_count


@app.cell
def _(
    EXPECTED_COLUMNS,
    clean_count,
    curated_count,
    unknown_artist_count,
    unknown_track_count,
    wrong_track_artist_count,
    curated_events,
    null_counts,
    user_source_counts,
):
    _schema = curated_events.collect_schema()
    validation_results = {
        "Expected columns and order": _schema.names() == EXPECTED_COLUMNS,
        "listened_at is UTC": "UTC" in str(_schema["listened_at"]),
        "Event count is preserved": clean_count == curated_count,
        "Required values are present": all(
            null_counts[column] == 0
            for column in ("user_id", "listened_at", "artist_id", "track_id", "source")
        ),
        "User/source counts are preserved": user_source_counts.get_column(
            "matches"
        ).all(),
        "Every artist ID exists": unknown_artist_count == 0,
        "Every track ID exists": unknown_track_count == 0,
        "Every track belongs to the event artist": wrong_track_artist_count == 0,
    }
    return (validation_results,)


@app.cell
def _(pl, validation_results):
    pl.DataFrame(
        {
            "check": list(validation_results),
            "passed": list(validation_results.values()),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo, validation_results):
    if all(validation_results.values()):
        mo.md(r"""
        **Result:** all checks passed. The curated listening-event table is ready to version with DVC.

        Two source rows can become identical after artist/track aliases collapse to the same IDs. That is fine; preserving the event count is the important check here.
        """)
    else:
        mo.md("**Result:** some checks failed. Review them before versioning the dataset.")
    return


if __name__ == "__main__":
    app.run()

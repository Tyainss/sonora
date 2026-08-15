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
    # Clean listening events validation

    Build and validate Sonora's first interim listening-events dataset from the current raw Last.fm snapshot.
    """)
    return


@app.cell
def _():
    import polars as pl

    from sonora.data.clean_listening_events import build_listening_events_clean
    from sonora.data.paths import DEFAULT_DATA_PATHS

    USER_ID = "Tyains"
    EXPECTED_COLUMNS = [
        "user_id",
        "listened_at",
        "artist_name",
        "track_name",
        "album_name",
        "artist_mbid",
        "track_mbid",
        "album_mbid",
        "source",
    ]
    paths = DEFAULT_DATA_PATHS
    return EXPECTED_COLUMNS, USER_ID, build_listening_events_clean, paths, pl


@app.cell(hide_code=True)
def _(USER_ID, mo, paths):
    mo.md(f"""
    - **User:** `{USER_ID}`
    - **Raw input:** `{paths.raw_lastfm_scrobbles}`
    - **Interim output:** `{paths.listening_events_clean}`

    The build is explicit because it replaces the current interim Parquet with a fresh build from the raw snapshot.
    """)
    return


@app.cell
def _(mo):
    build_button = mo.ui.run_button(
        label="Build interim dataset",
        kind="success",
    )
    build_button
    return (build_button,)


@app.cell
def _(USER_ID, build_button, build_listening_events_clean, mo, paths):
    mo.stop(not build_button.value)

    output_path = build_listening_events_clean(
        user_id=USER_ID,
        paths=paths,
    )

    mo.md(f"Built `{output_path}` successfully.")
    return (output_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contract validation
    """)
    return


@app.cell
def _(output_path, paths, pl):
    raw_scrobbles = pl.scan_ndjson(paths.raw_lastfm_scrobbles)
    clean_events = pl.scan_parquet(output_path)
    return clean_events, raw_scrobbles


@app.cell
def _(clean_events, pl, raw_scrobbles):
    raw_event_count = raw_scrobbles.select(pl.len()).collect().item()
    clean_event_count = clean_events.select(pl.len()).collect().item()
    duplicate_rows_removed = raw_event_count - clean_event_count

    build_summary = pl.DataFrame(
        {
            "metric": [
                "Raw events",
                "Clean events",
                "Rows removed by cleaned-event deduplication",
            ],
            "value": [
                raw_event_count,
                clean_event_count,
                duplicate_rows_removed,
            ],
        }
    )

    build_summary
    return clean_event_count, duplicate_rows_removed, raw_event_count


@app.cell
def _(clean_events, pl):
    _schema = clean_events.collect_schema()
    schema_summary = pl.DataFrame(
        {
            "column": _schema.names(),
            "dtype": [str(dtype) for dtype in _schema.dtypes()],
        }
    )

    schema_summary
    return (schema_summary,)


@app.cell
def _(clean_events, pl, raw_scrobbles):
    raw_bounds = (
        raw_scrobbles.select(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .dt.replace_time_zone("UTC")
            .min()
            .alias("first_listened_at"),
            pl.from_epoch("timestamp_unix", time_unit="s")
            .dt.replace_time_zone("UTC")
            .max()
            .alias("last_listened_at"),
        )
        .collect()
        .row(0, named=True)
    )
    clean_bounds = (
        clean_events.select(
            pl.col("listened_at").min().alias("first_listened_at"),
            pl.col("listened_at").max().alias("last_listened_at"),
        )
        .collect()
        .row(0, named=True)
    )

    timestamp_summary = pl.DataFrame(
        {
            "dataset": ["raw", "clean"],
            "first_listened_at": [
                raw_bounds["first_listened_at"],
                clean_bounds["first_listened_at"],
            ],
            "last_listened_at": [
                raw_bounds["last_listened_at"],
                clean_bounds["last_listened_at"],
            ],
        }
    )

    timestamp_summary
    return clean_bounds, raw_bounds


@app.cell
def _(clean_events, pl):
    null_counts = clean_events.select(pl.all().null_count()).collect().row(
        0,
        named=True,
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
def _(clean_events, pl):
    identifier_coverage = (
        clean_events.select(
            pl.len().alias("events"),
            pl.col("artist_mbid").is_not_null().sum().alias("artist_mbid"),
            pl.col("track_mbid").is_not_null().sum().alias("track_mbid"),
            pl.col("album_mbid").is_not_null().sum().alias("album_mbid"),
        )
        .with_columns(
            (pl.col("artist_mbid") / pl.col("events") * 100)
            .round(1)
            .alias("artist_mbid_pct"),
            (pl.col("track_mbid") / pl.col("events") * 100)
            .round(1)
            .alias("track_mbid_pct"),
            (pl.col("album_mbid") / pl.col("events") * 100)
            .round(1)
            .alias("album_mbid_pct"),
        )
        .collect()
    )

    identifier_coverage
    return


@app.cell
def _(clean_events, pl, raw_scrobbles):
    def _name_difference_counts(column: str) -> tuple[int, int]:
        _raw_names = (
            raw_scrobbles.select(
                pl.col(column).str.strip_chars().replace("", None).alias(column)
            )
            .filter(pl.col(column).is_not_null())
            .unique()
        )
        _clean_names = (
            clean_events.select(pl.col(column))
            .filter(pl.col(column).is_not_null())
            .unique()
        )
        _missing_from_clean = (
            _raw_names.join(_clean_names, on=column, how="anti")
            .select(pl.len())
            .collect()
            .item()
        )
        _new_in_clean = (
            _clean_names.join(_raw_names, on=column, how="anti")
            .select(pl.len())
            .collect()
            .item()
        )
        return _missing_from_clean, _new_in_clean

    _name_rows = []
    for _column in ("artist_name", "track_name", "album_name"):
        _missing, _new = _name_difference_counts(_column)
        _name_rows.append(
            {
                "column": _column,
                "raw_trimmed_names_missing_from_clean": _missing,
                "new_names_in_clean": _new,
            }
        )

    name_preservation = pl.DataFrame(_name_rows)
    name_preservation
    return (name_preservation,)


@app.cell
def _(
    EXPECTED_COLUMNS,
    USER_ID,
    clean_bounds,
    null_counts,
    raw_bounds,
    clean_event_count,
    clean_events,
    name_preservation,
    pl,
    raw_event_count,
):
    _schema = clean_events.collect_schema()
    _user_ids = (
        clean_events.select(pl.col("user_id").unique().sort())
        .collect()
        .get_column("user_id")
        .to_list()
    )
    _sources = (
        clean_events.select(pl.col("source").unique().sort())
        .collect()
        .get_column("source")
        .to_list()
    )
    _deduplicated_count = clean_events.unique().select(pl.len()).collect().item()
    _names_preserved = name_preservation.select(
        (
            (pl.col("raw_trimmed_names_missing_from_clean") == 0)
            & (pl.col("new_names_in_clean") == 0)
        ).all()
    ).item()

    validation_results = pl.DataFrame(
        {
            "check": [
                "Expected columns and order",
                "UTC listened_at",
                "Row count did not increase",
                "No duplicate clean events",
                "Required values are present",
                "User ID is correct",
                "Source is correct",
                "Timestamp range is preserved",
                "Observed names are preserved after trimming",
            ],
            "passed": [
                _schema.names() == EXPECTED_COLUMNS,
                "UTC" in str(_schema["listened_at"]),
                clean_event_count <= raw_event_count,
                _deduplicated_count == clean_event_count,
                all(null_counts[column] == 0 for column in ("user_id", "listened_at", "artist_name", "track_name", "source")),
                _user_ids == [USER_ID],
                _sources == ["lastfm"],
                raw_bounds == clean_bounds,
                _names_preserved,
            ],
        }
    )

    validation_results
    return (validation_results,)


@app.cell(hide_code=True)
def _(mo, validation_results):
    _passed = validation_results.get_column("passed").all()
    if _passed:
        mo.md("**Result:** all interim listening-event contract checks passed.")
    else:
        mo.md("**Result:** at least one contract check failed and should be investigated before accepting the interim dataset.")
    return


if __name__ == "__main__":
    app.run()

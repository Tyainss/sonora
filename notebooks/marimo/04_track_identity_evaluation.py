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
    # Track identity evaluation

    Checks how often Last.fm gives the same song different titles, and which differences we can safely collapse.
    """)
    return


@app.cell
def _():
    import polars as pl

    from sonora.data.comparison_normalization import normalize_for_comparison
    from sonora.data.curate_tracks import build_curated_track_tables
    from sonora.data.paths import DEFAULT_DATA_PATHS
    from sonora.data.track_resolution import (
        build_track_song_keys,
        resolve_track_identities,
    )

    paths = DEFAULT_DATA_PATHS
    return (
        build_curated_track_tables,
        build_track_song_keys,
        normalize_for_comparison,
        paths,
        pl,
        resolve_track_identities,
    )


@app.cell
def _(mo, paths):
    _required_paths = [
        paths.listening_events_clean,
        paths.curated_artists,
        paths.curated_artist_aliases,
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


@app.cell
def _(normalize_for_comparison, paths, pl):
    artist_aliases = pl.read_parquet(paths.curated_artist_aliases)
    artists = pl.read_parquet(paths.curated_artists)
    resolved_events = (
        pl.scan_parquet(paths.listening_events_clean)
        .join(
            artist_aliases.lazy(),
            left_on="artist_name",
            right_on="observed_artist_name",
            how="left",
            validate="m:1",
        )
        .join(artists.lazy(), on="artist_id", how="left", validate="m:1")
        .with_columns(
            normalize_for_comparison(pl.col("track_name")).alias(
                "comparison_track_name"
            )
        )
        .collect()
    )
    return artist_aliases, artists, resolved_events


@app.cell
def _(mo, resolved_events):
    _unmapped_events = resolved_events.get_column("artist_id").null_count()
    mo.stop(
        _unmapped_events > 0,
        mo.callout(
            f"{_unmapped_events:,} listening events do not map to a canonical artist.",
            kind="danger",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Track catalogue
    """)
    return


@app.cell
def _(artists, mo, resolved_events):
    _observed_track_labels = resolved_events.select("artist_id", "track_name").n_unique()
    _comparison_track_labels = resolved_events.select(
        "artist_id", "comparison_track_name"
    ).n_unique()
    _formatting_reduction = _observed_track_labels - _comparison_track_labels

    mo.md(f"""
    - **Listening events:** {resolved_events.height:,}
    - **Canonical artists:** {artists.height:,}
    - **Observed artist-track labels:** {_observed_track_labels:,}
    - **After basic comparison normalization:** {_comparison_track_labels:,}

    Basic formatting only removes **{_formatting_reduction:,}** duplicate labels, so most of the track cleanup is about versions and credits rather than punctuation or casing.
    """)
    return


@app.cell
def _(pl, resolved_events):
    track_catalogue = (
        resolved_events.group_by(
            "artist_id",
            "canonical_name",
            "track_name",
            "comparison_track_name",
        )
        .agg(pl.len().alias("scrobble_count"))
        .sort("scrobble_count", descending=True)
    )
    track_catalogue.head(30)
    return (track_catalogue,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Track matching stays inside each resolved artist. A title shared by two different artists is never compared here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formatting variants

    These are titles that already match after simple casing, punctuation, spacing or accent cleanup.
    """)
    return


@app.cell
def _(pl, track_catalogue):
    exact_normalization_groups = (
        track_catalogue.group_by(
            "artist_id",
            "canonical_name",
            "comparison_track_name",
        )
        .agg(
            pl.col("track_name").sort().alias("observed_track_names"),
            pl.col("track_name").n_unique().alias("observed_name_count"),
            pl.col("scrobble_count").sum().alias("scrobble_count"),
        )
        .filter(pl.col("observed_name_count") > 1)
        .sort(
            ["observed_name_count", "scrobble_count"],
            descending=True,
        )
    )
    exact_normalization_groups.head(60)
    return (exact_normalization_groups,)


@app.cell
def _(exact_normalization_groups, mo):
    mo.md(f"""
    **{exact_normalization_groups.height:,}** groups contain more than one observed spelling. Most are harmless variants such as casing, apostrophes or punctuation.

    The punctuation-only `MAQUINA.` titles are the odd case: their normalized value is empty, so we need to keep their raw titles separate later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Trailing qualifiers

    A lot of version information sits in parentheses, brackets or after a dash. But that shape alone is not enough: `The World (Is Going Up in Flames)` shows why we cannot just strip everything at the end.
    """)
    return


@app.cell
def _(normalize_for_comparison, pl, track_catalogue):
    qualified_tracks = (
        track_catalogue.with_columns(
            pl.col("track_name")
            .str.extract(r"\(([^()]*)\)\s*$", 1)
            .alias("parenthetical_qualifier"),
            pl.col("track_name")
            .str.extract(r"\[([^\[\]]*)\]\s*$", 1)
            .alias("bracket_qualifier"),
            pl.col("track_name")
            .str.extract(r"\s[-–—]\s(.+)$", 1)
            .alias("dash_qualifier"),
        )
        .with_columns(
            pl.coalesce(
                "parenthetical_qualifier",
                "bracket_qualifier",
                "dash_qualifier",
            ).alias("track_qualifier"),
            pl.when(pl.col("parenthetical_qualifier").is_not_null())
            .then(pl.lit("parentheses"))
            .when(pl.col("bracket_qualifier").is_not_null())
            .then(pl.lit("brackets"))
            .when(pl.col("dash_qualifier").is_not_null())
            .then(pl.lit("dash"))
            .otherwise(None)
            .alias("qualifier_style"),
            pl.when(pl.col("parenthetical_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s*\([^()]*\)\s*$", ""))
            .when(pl.col("bracket_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s*\[[^\[\]]*\]\s*$", ""))
            .when(pl.col("dash_qualifier").is_not_null())
            .then(pl.col("track_name").str.replace(r"\s[-–—]\s.+$", ""))
            .otherwise(pl.col("track_name"))
            .alias("base_track_name"),
        )
        .with_columns(
            normalize_for_comparison(pl.col("track_qualifier")).alias(
                "comparison_qualifier"
            ),
            normalize_for_comparison(pl.col("base_track_name")).alias(
                "base_comparison_track_name"
            ),
        )
        .filter(pl.col("track_qualifier").is_not_null())
    )
    return (qualified_tracks,)


@app.cell
def _(pl, qualified_tracks):
    qualifier_summary = (
        qualified_tracks.group_by("comparison_qualifier")
        .agg(
            pl.len().alias("artist_track_labels"),
            pl.col("scrobble_count").sum().alias("scrobbles"),
            pl.col("track_qualifier").n_unique().alias("observed_spellings"),
        )
        .sort(["artist_track_labels", "scrobbles"], descending=True)
    )
    qualifier_summary.head(60)
    return (qualifier_summary,)


@app.cell
def _(pl, qualified_tracks):
    qualifier_examples = qualified_tracks.sort("scrobble_count", descending=True).select(
        "canonical_name",
        "track_name",
        "base_track_name",
        "track_qualifier",
        "qualifier_style",
        "scrobble_count",
    )
    qualifier_examples.head(80)
    return (qualifier_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Common version labels

    Live and remastered tracks are by far the most common recognizable variants, with mixes/remixes a distant third. That is enough volume to make targeted rules worthwhile.
    """)
    return


@app.cell
def _(pl, qualified_tracks):
    version_qualified_tracks = qualified_tracks.with_columns(
        pl.when(pl.col("comparison_qualifier").str.contains(r"\bremaster(?:ed)?\b"))
        .then(pl.lit("remaster"))
        .when(pl.col("comparison_qualifier").str.contains(r"\blive\b"))
        .then(pl.lit("live"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bacoustic\b"))
        .then(pl.lit("acoustic"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bdemo\b"))
        .then(pl.lit("demo"))
        .when(pl.col("comparison_qualifier").str.contains(r"\binstrumental\b"))
        .then(pl.lit("instrumental"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bradio\b"))
        .then(pl.lit("radio"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bedit\b"))
        .then(pl.lit("edit"))
        .when(pl.col("comparison_qualifier").str.contains(r"\b(?:remix|mix)\b"))
        .then(pl.lit("mix_or_remix"))
        .when(pl.col("comparison_qualifier").str.contains(r"\b(?:mono|stereo)\b"))
        .then(pl.lit("mono_or_stereo"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bsession\b"))
        .then(pl.lit("session"))
        .when(
            pl.col("comparison_qualifier").str.contains(
                r"\b(?:anniversary|deluxe|reissue)\b"
            )
        )
        .then(pl.lit("release_edition"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bversion\b"))
        .then(pl.lit("version"))
        .otherwise(pl.lit("other"))
        .alias("qualifier_kind")
    )
    return (version_qualified_tracks,)


@app.cell
def _(pl, version_qualified_tracks):
    version_label_summary = (
        version_qualified_tracks.group_by("qualifier_kind")
        .agg(
            pl.len().alias("artist_track_labels"),
            pl.col("scrobble_count").sum().alias("scrobbles"),
            pl.col("comparison_qualifier").n_unique().alias("distinct_qualifiers"),
        )
        .sort("artist_track_labels", descending=True)
    )
    version_label_summary
    return (version_label_summary,)


@app.cell
def _(version_qualified_tracks):
    version_qualified_tracks.filter(
        version_qualified_tracks.get_column("qualifier_kind") != "other"
    ).sort("scrobble_count", descending=True).select(
        "canonical_name",
        "track_name",
        "base_track_name",
        "track_qualifier",
        "qualifier_kind",
        "scrobble_count",
    ).head(100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Versions with an observed base title

    Here the stripped title also appears on its own for the same artist, which gives us an easy sanity check for the rules.
    """)
    return


@app.cell
def _(pl, qualified_tracks, track_catalogue):
    plain_track_titles = track_catalogue.select(
        "artist_id",
        pl.col("comparison_track_name").alias("base_comparison_track_name"),
        pl.col("track_name").alias("matched_base_track_name"),
        pl.col("scrobble_count").alias("base_scrobble_count"),
    )

    qualifier_base_matches = (
        qualified_tracks.join(
            plain_track_titles,
            on=["artist_id", "base_comparison_track_name"],
            how="inner",
            validate="m:m",
        )
        .filter(pl.col("track_name") != pl.col("matched_base_track_name"))
        .sort(
            ["scrobble_count", "base_scrobble_count"],
            descending=True,
        )
        .select(
            "artist_id",
            "canonical_name",
            "track_name",
            "matched_base_track_name",
            "track_qualifier",
            "qualifier_style",
            "scrobble_count",
            "base_scrobble_count",
        )
        .unique()
    )
    qualifier_base_matches.head(100)
    return (qualifier_base_matches,)


@app.cell
def _(mo, qualifier_base_matches, qualified_tracks):
    mo.md(f"""
    - **Titles with a trailing qualifier:** {qualified_tracks.height:,}
    - **With an observed base-title match:** {qualifier_base_matches.height:,}

    Only a minority have the plain base title in the data, so the production rule cannot depend on seeing both versions first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Song keys

    These are the rules used by the resolver. They only strip suffixes that clearly look like version, recording or featured-artist metadata. Unknown suffixes stay part of the title.
    """)
    return


@app.cell
def _(build_track_song_keys, track_catalogue):
    song_key_catalogue = build_track_song_keys(track_catalogue)
    return (song_key_catalogue,)


@app.cell
def _(pl, song_key_catalogue):
    rule_summary = (
        song_key_catalogue.filter(pl.col("collapse_rule").is_not_null())
        .group_by("collapse_rule")
        .agg(
            pl.len().alias("artist_track_labels"),
            pl.col("scrobble_count").sum().alias("scrobbles"),
        )
        .sort("artist_track_labels", descending=True)
    )
    rule_summary
    return (rule_summary,)


@app.cell
def _(mo, song_key_catalogue):
    _song_key_count = song_key_catalogue.select("artist_id", "song_key").n_unique()
    _raw_fallbacks = song_key_catalogue.get_column("song_key").str.starts_with(
        "raw:"
    ).sum()
    _reduction = song_key_catalogue.height - _song_key_count

    mo.md(f"""
    - **Artist-track labels:** {song_key_catalogue.height:,}
    - **Song keys before MBID matching:** {_song_key_count:,}
    - **Labels using the raw-title fallback:** {_raw_fallbacks:,}

    The title rules consolidate **{_reduction:,}** labels. That is much more useful than basic formatting cleanup, while still leaving unknown suffixes alone.
    """)
    return


@app.cell
def _(pl, song_key_catalogue):
    collapse_groups = (
        song_key_catalogue.group_by(
            "artist_id",
            "canonical_name",
            "song_key",
        )
        .agg(
            pl.col("track_name").sort().alias("observed_track_names"),
            pl.col("track_name").n_unique().alias("observed_name_count"),
            pl.col("collapse_rule").drop_nulls().unique().sort().alias("rules"),
            pl.col("scrobble_count").sum().alias("scrobbles"),
        )
        .filter(pl.col("observed_name_count") > 1)
        .sort(
            ["observed_name_count", "scrobbles"],
            descending=True,
        )
    )
    collapse_groups.head(100)
    return (collapse_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The biggest groups look sensible: repeated live recordings, remasters and other named versions land on the same song. We still leave arrangement-specific or unclear labels alone instead of trying to cover every possible suffix.
    """)
    return


@app.cell
def _(pl, qualifier_base_matches, song_key_catalogue):
    unhandled_base_matches = (
        qualifier_base_matches.join(
            song_key_catalogue.select(
                "artist_id",
                "canonical_name",
                "track_name",
                "collapse_rule",
            ),
            on=["artist_id", "canonical_name", "track_name"],
            how="left",
        )
        .filter(pl.col("collapse_rule").is_null())
        .select(
            "canonical_name",
            "track_name",
            "matched_base_track_name",
            "track_qualifier",
            "scrobble_count",
            "base_scrobble_count",
        )
        .sort(
            ["scrobble_count", "base_scrobble_count"],
            descending=True,
        )
        .unique()
    )
    unhandled_base_matches.head(80)
    return (unhandled_base_matches,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The remaining misses are mixed. A few are obvious metadata, while others such as `Mexico City`, `Interlude` or arrangement names are less clear. Keeping them separate is safer than growing a long dataset-specific rule list.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Track MBIDs

    MBIDs cover enough of the data to help with names that title rules miss. We use a shared track MBID as a merge signal inside one artist, but different MBIDs do not split an otherwise matching song.
    """)
    return


@app.cell
def _(mo, pl, resolved_events):
    _events_with_track_mbid = resolved_events.filter(
        pl.col("track_mbid").is_not_null()
    ).height
    _labels_with_track_mbid = (
        resolved_events.group_by("artist_id", "track_name")
        .agg(pl.col("track_mbid").drop_nulls().len().alias("mbid_count"))
        .filter(pl.col("mbid_count") > 0)
        .height
    )
    _artist_track_labels = resolved_events.select("artist_id", "track_name").n_unique()

    mo.md(f"""
    - **Events with a track MBID:** {_events_with_track_mbid:,} / {resolved_events.height:,} ({_events_with_track_mbid / resolved_events.height * 100:.1f}%)
    - **Artist-track labels with a track MBID:** {_labels_with_track_mbid:,} / {_artist_track_labels:,} ({_labels_with_track_mbid / _artist_track_labels * 100:.1f}%)

    Coverage is high enough for MBIDs to be useful as an extra bridge, but not high enough to make them the main identity key.
    """)
    return


@app.cell
def _(pl, resolved_events):
    shared_track_mbid_titles = (
        resolved_events.filter(pl.col("track_mbid").is_not_null())
        .group_by("artist_id", "canonical_name", "track_mbid")
        .agg(
            pl.col("track_name").unique().sort().alias("observed_track_names"),
            pl.col("track_name").n_unique().alias("observed_name_count"),
            pl.len().alias("scrobbles"),
        )
        .filter(pl.col("observed_name_count") > 1)
        .sort(
            ["observed_name_count", "scrobbles"],
            descending=True,
        )
    )
    shared_track_mbid_titles.head(80)
    return (shared_track_mbid_titles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Kirinaki Shima` / `kirinakijima` is exactly the kind of case the MBID bridge helps with: the names do not match well, but the identifier does.
    """)
    return


@app.cell
def _(pl, resolved_events, song_key_catalogue):
    events_with_song_key = resolved_events.join(
        song_key_catalogue.select(
            "artist_id",
            "track_name",
            "song_key",
        ),
        on=["artist_id", "track_name"],
        how="left",
        validate="m:1",
    )

    shared_mbid_across_song_keys = (
        events_with_song_key.filter(pl.col("track_mbid").is_not_null())
        .group_by("artist_id", "canonical_name", "track_mbid")
        .agg(
            pl.col("song_key").n_unique().alias("song_key_count"),
            pl.col("track_name").unique().sort().alias("observed_track_names"),
            pl.len().alias("scrobbles"),
        )
        .filter(pl.col("song_key_count") > 1)
        .sort(
            ["song_key_count", "scrobbles"],
            descending=True,
        )
    )
    shared_mbid_across_song_keys.head(80)
    return (shared_mbid_across_song_keys,)


@app.cell
def _(pl, resolved_events, song_key_catalogue):
    song_group_mbid_summary = (
        resolved_events.join(
            song_key_catalogue.select(
                "artist_id",
                "track_name",
                "song_key",
            ),
            on=["artist_id", "track_name"],
            how="left",
            validate="m:1",
        )
        .group_by("artist_id", "canonical_name", "song_key")
        .agg(
            pl.col("track_name").n_unique().alias("observed_name_count"),
            pl.col("track_name").unique().sort().alias("observed_track_names"),
            pl.col("track_mbid").drop_nulls().n_unique().alias("track_mbid_count"),
        )
        .filter(
            (pl.col("observed_name_count") > 1)
            & (pl.col("track_mbid_count") > 1)
        )
        .sort(
            ["track_mbid_count", "observed_name_count"],
            descending=True,
        )
    )
    song_group_mbid_summary.head(80)
    return (song_group_mbid_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The opposite also happens: songs such as `Chicago` have several MBIDs across different recordings. That fits our song-level definition, so an MBID conflict is not a reason to split a group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Final track resolution
    """)
    return


@app.cell
def _(resolve_track_identities, resolved_events):
    track_resolution = resolve_track_identities(
        resolved_events.select("artist_id", "track_name", "track_mbid").lazy()
    )
    return (track_resolution,)


@app.cell
def _(mo, song_key_catalogue, track_resolution):
    _song_key_count = song_key_catalogue.select("artist_id", "song_key").n_unique()
    _track_count = track_resolution.clusters.select(
        "artist_id", "cluster_key"
    ).n_unique()
    _mbid_reduction = _song_key_count - _track_count

    mo.md(f"""
    - **Song keys from title rules:** {_song_key_count:,}
    - **Final track clusters after MBID matching:** {_track_count:,}
    - **Extra merges from MBID links:** {_mbid_reduction:,}

    This is the v1 resolver: conservative title rules first, then same-MBID links within the artist. Everything else stays separate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Build track tables
    """)
    return


@app.cell
def _(mo):
    build_track_tables = mo.ui.run_button(label="Build track tables")
    build_track_tables
    return (build_track_tables,)


@app.cell
def _(build_curated_track_tables, build_track_tables, mo, paths):
    mo.stop(not build_track_tables.value)

    _build = build_curated_track_tables(paths=paths)
    mo.md(f"""
    **Tracks:** {_build.tracks.height:,}

    **Aliases:** {_build.track_aliases.height:,}
    """)
    return


if __name__ == "__main__":
    app.run()

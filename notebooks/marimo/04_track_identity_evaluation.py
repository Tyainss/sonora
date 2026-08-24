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

    Looks at how track names vary within each resolved artist before choosing track matching rules.
    """)
    return


@app.cell
def _():
    import polars as pl

    from sonora.data.comparison_normalization import normalize_for_comparison
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return normalize_for_comparison, paths, pl


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

    mo.md(f"""
    - **Listening events:** {resolved_events.height:,}
    - **Canonical artists:** {artists.height:,}
    - **Observed artist-track labels:** {_observed_track_labels:,}
    - **Artist-track labels after comparison normalization:** {_comparison_track_labels:,}
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
    Artist identity is already resolved here, so track comparisons stay inside one canonical artist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formatting variants

    Comparison normalization removes differences in case, diacritics, punctuation and spacing. Groups with more than one observed title show the variants this already catches.
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
    **{exact_normalization_groups.height:,}** normalized artist-track groups contain more than one observed title spelling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Trailing qualifiers

    Track versions are often written as a qualifier at the end of the title. The next tables inventory parenthesized, bracketed and dash-separated qualifiers without using them to merge anything.
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

    These broad categories are only for inspection. They help show how often familiar version labels appear and what kinds of titles they cover.
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
    ## 5. Qualifier variants with a matching base title

    For these rows, removing the trailing qualifier produces another observed title by the same canonical artist. This is the most useful set for judging which version markers can be collapsed safely.
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
    - **Track labels with a trailing qualifier:** {qualified_tracks.height:,}
    - **Qualifier-bearing labels with an observed base-title match:** {qualifier_base_matches.height:,}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Proposed song keys

    This proposal removes only qualifiers that look like recording, release or featured-artist metadata. Other trailing text stays part of the title. The result is an evaluation key only; nothing is persisted or merged here.
    """)
    return


@app.cell
def _(pl, qualified_tracks):
    proposed_qualified_tracks = qualified_tracks.with_columns(
        pl.when(
            pl.col("comparison_qualifier").str.contains(
                r"\b(?:feat(?:uring)?|ft)\b|^duet with\b"
            )
        )
        .then(pl.lit("featured_credit"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bremaster(?:ed)?\b"))
        .then(pl.lit("remaster"))
        .when(
            pl.col("comparison_qualifier").str.contains(
                r"\b(?:live|ao vivo|en vivo)\b"
            )
        )
        .then(pl.lit("live"))
        .when(
            pl.col("comparison_qualifier").str.contains(
                r"\b(?:acoustic|acustic[oa]?)\b"
            )
        )
        .then(pl.lit("acoustic"))
        .when(pl.col("comparison_qualifier").str.contains(r"\bdemo\b"))
        .then(pl.lit("demo"))
        .when(pl.col("comparison_qualifier").str.contains(r"\binstrumental\b"))
        .then(pl.lit("instrumental"))
        .when(
            pl.col("comparison_qualifier").str.contains(
                r"\bradio\b.*\bedit\b|\bedit\b"
            )
        )
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
        .otherwise(None)
        .alias("collapse_rule")
    )
    return (proposed_qualified_tracks,)


@app.cell
def _(normalize_for_comparison, pl, proposed_qualified_tracks, track_catalogue):
    proposed_track_catalogue = (
        track_catalogue.join(
            proposed_qualified_tracks.select(
                "artist_id",
                "track_name",
                "base_track_name",
                "collapse_rule",
            ),
            on=["artist_id", "track_name"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("collapse_rule").is_not_null())
            .then(pl.col("base_track_name"))
            .otherwise(pl.col("track_name"))
            .alias("proposed_song_name")
        )
        .with_columns(
            normalize_for_comparison(pl.col("proposed_song_name")).alias(
                "proposed_song_key"
            )
        )
        .with_columns(
            pl.when(pl.col("proposed_song_key").is_null())
            .then(pl.concat_str([pl.lit("raw:"), pl.col("track_name")]))
            .otherwise(pl.col("proposed_song_key"))
            .alias("safe_proposed_song_key")
        )
    )
    return (proposed_track_catalogue,)


@app.cell
def _(pl, proposed_track_catalogue):
    proposed_rule_summary = (
        proposed_track_catalogue.filter(pl.col("collapse_rule").is_not_null())
        .group_by("collapse_rule")
        .agg(
            pl.len().alias("artist_track_labels"),
            pl.col("scrobble_count").sum().alias("scrobbles"),
        )
        .sort("artist_track_labels", descending=True)
    )
    proposed_rule_summary
    return (proposed_rule_summary,)


@app.cell
def _(mo, proposed_track_catalogue):
    _proposed_track_count = proposed_track_catalogue.select(
        "artist_id", "safe_proposed_song_key"
    ).n_unique()
    _null_song_keys = proposed_track_catalogue.get_column(
        "proposed_song_key"
    ).null_count()

    mo.md(f"""
    - **Artist-track labels before song-key rules:** {proposed_track_catalogue.height:,}
    - **Proposed canonical song keys:** {_proposed_track_count:,}
    - **Labels needing a raw-title fallback because normalization is empty:** {_null_song_keys:,}
    """)
    return


@app.cell
def _(pl, proposed_track_catalogue):
    proposed_collapse_groups = (
        proposed_track_catalogue.group_by(
            "artist_id",
            "canonical_name",
            "safe_proposed_song_key",
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
    proposed_collapse_groups.head(100)
    return (proposed_collapse_groups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The proposal keeps unrecognized qualifiers separate. This lets us inspect plausible same-song variants that the conservative rules deliberately miss before deciding whether another rule is justified.
    """)
    return


@app.cell
def _(pl, proposed_qualified_tracks, qualifier_base_matches):
    unhandled_base_matches = (
        qualifier_base_matches.join(
            proposed_qualified_tracks.select(
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
    ## 7. Track MBID evidence

    Track MBIDs are inspected as supporting evidence rather than a merge rule. They may distinguish recordings, so different MBIDs inside one proposed song do not automatically mean the song-level grouping is wrong.
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
    - **Listening events with a track MBID:** {_events_with_track_mbid:,} / {resolved_events.height:,} ({_events_with_track_mbid / resolved_events.height * 100:.1f}%)
    - **Observed artist-track labels with at least one track MBID:** {_labels_with_track_mbid:,} / {_artist_track_labels:,} ({_labels_with_track_mbid / _artist_track_labels * 100:.1f}%)
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


@app.cell
def _(pl, proposed_track_catalogue, resolved_events):
    events_with_proposed_key = resolved_events.join(
        proposed_track_catalogue.select(
            "artist_id",
            "track_name",
            "safe_proposed_song_key",
        ),
        on=["artist_id", "track_name"],
        how="left",
        validate="m:1",
    )

    shared_mbid_across_song_keys = (
        events_with_proposed_key.filter(pl.col("track_mbid").is_not_null())
        .group_by("artist_id", "canonical_name", "track_mbid")
        .agg(
            pl.col("safe_proposed_song_key").n_unique().alias("song_key_count"),
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
def _(pl, proposed_track_catalogue, resolved_events):
    proposed_group_mbid_summary = (
        resolved_events.join(
            proposed_track_catalogue.select(
                "artist_id",
                "track_name",
                "safe_proposed_song_key",
            ),
            on=["artist_id", "track_name"],
            how="left",
            validate="m:1",
        )
        .group_by("artist_id", "canonical_name", "safe_proposed_song_key")
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
    proposed_group_mbid_summary.head(80)
    return (proposed_group_mbid_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation focus

    The proposed keys let us review three things before implementation: whether the recognized metadata rules create sensible song groups, which unhandled qualifiers deserve another rule, and whether track MBIDs reveal missed matches or recording-level differences that the title rules should respect.
    """)
    return


if __name__ == "__main__":
    app.run()

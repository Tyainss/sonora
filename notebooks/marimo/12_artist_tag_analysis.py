import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return mo, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Artist tag analysis

    How well do Last.fm tags describe the artists in my listening history?
    """)
    return


@app.cell
def _(paths, pl):
    artists = pl.read_parquet(paths.curated_artists)
    artist_tags = pl.read_parquet(paths.curated_artist_tags)

    if artists["artist_id"].n_unique() != artists.height:
        raise ValueError("The artist catalogue contains duplicate IDs")
    if artist_tags["artist_id"].n_unique() != artist_tags.height:
        raise ValueError("The tag dataset contains duplicate artist IDs")
    if set(artists["artist_id"]) != set(artist_tags["artist_id"]):
        raise ValueError("The tag dataset does not match the artist catalogue")

    tagged = artist_tags.with_columns(pl.col("tags").list.len().alias("tag_count"))
    return artists, tagged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many artists have tags?
    """)
    return


@app.cell
def _(artists, mo, pl, tagged):
    total_artists = artists.height
    with_tags = tagged.filter(pl.col("tag_count") > 0).height
    empty = tagged.filter(pl.col("status") == "empty").height
    mismatched = tagged.filter(pl.col("status") == "identity_mismatch").height
    five_or_more = tagged.filter(pl.col("tag_count") >= 5).height
    ten_tags = tagged.filter(pl.col("tag_count") == 10).height

    mo.md(f"""
    - **{with_tags:,} of {total_artists:,} artists ({with_tags / total_artists:.1%})** have tags.
    - **{five_or_more:,}** have at least five tags; **{ten_tags:,}** have ten.
    - **{empty:,}** have empty results; **{mismatched:,}** have an artist-name mismatch.

    Most artists have usable tags. The dataset contains at most ten tags per artist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does coverage depend on how often I listen to an artist?
    """)
    return


@app.cell
def _(paths, pl, tagged):
    listen_counts = (
        pl.scan_parquet(paths.curated_listening_events)
        .group_by("artist_id")
        .agg(pl.len().alias("total_listens"))
        .collect()
    )
    coverage = tagged.join(listen_counts, on="artist_id", how="left").with_columns(
        pl.col("total_listens").fill_null(0)
    )
    bands = [
        ("1-2", 1, 3),
        ("3-9", 3, 10),
        ("10-29", 10, 30),
        ("30-99", 30, 100),
        ("100+", 100, None),
    ]
    rows = []
    for label, lower, upper in bands:
        group = coverage.filter(pl.col("total_listens") >= lower)
        if upper is not None:
            group = group.filter(pl.col("total_listens") < upper)
        total = group.height
        with_tags = group.filter(pl.col("tag_count") > 0).height
        rows.append(
            {
                "Total listens": label,
                "Artists": total,
                "With tags (%)": round(100 * with_tags / total, 1) if total else None,
            }
        )
    coverage_table = pl.DataFrame(rows)
    coverage_table
    return coverage_table


@app.cell(hide_code=True)
def _(coverage_table, mo):
    _low = coverage_table["With tags (%)"][0]
    _high = coverage_table["With tags (%)"][-1]
    mo.md(f"""
    Artists with 100 or more listens have **{_high}%** tag coverage, compared with **{_low}%** for artists with one or two listens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which tags appear most often?
    """)
    return


@app.cell
def _(pl, tagged):
    _all_tags = (
        tagged.select("artist_id", "tags")
        .explode("tags")
        .drop_nulls("tags")
        .unnest("tags")
    )
    tag_frequency = (
        _all_tags.group_by("name")
        .agg(pl.col("artist_id").n_unique().alias("Artists"))
        .sort("Artists", descending=True)
    )
    tag_frequency.head(12)
    return (tag_frequency,)


@app.cell(hide_code=True)
def _(mo, tag_frequency):
    _unique_tags = tag_frequency.height
    _one_artist = tag_frequency.filter(pl.col("Artists") == 1).height
    mo.md(f"""
    There are **{_unique_tags:,} distinct tags**. **{_one_artist:,} ({_one_artist / _unique_tags:.1%})** appear on only one artist.

    Some frequent tags describe a musical style, while others describe nationality or other characteristics. Rare tags may be useful, but can also be very noisy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Are different names used for the same tag?
    """)
    return


@app.cell
def _(pl, tag_frequency):
    spelling_variants = (
        tag_frequency.with_columns(
            pl.col("name")
            .str.to_lowercase()
            .str.replace_all(r"[\s_-]", "")
            .alias("normalized")
        )
        .group_by("normalized")
        .agg(
            pl.col("name").sort().alias("Variants"),
            pl.len().alias("variant_count"),
            pl.col("Artists").sum().alias("Tag assignments"),
        )
        .filter(pl.col("variant_count") > 1)
        .sort("Tag assignments", descending=True)
        .head(8)
        .select("Variants", "Tag assignments")
    )
    spelling_variants
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are possible spelling variants, not automatically approved replacements. Common examples include `Hip-Hop` / `hip hop` and `trip-hop` / `trip hop`.

    Some tags also describe personal collections or events rather than music, such as `funk_add_to_lidarr_batch_18` and `FDC 2023 Composers`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which artist names need checking?
    """)
    return


@app.cell
def _(pl, tagged):
    identity_mismatches = tagged.filter(
        pl.col("status") == "identity_mismatch"
    ).select("canonical_name", "queried_name", "reported_name")
    identity_mismatches
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    Most artists have enough tags to try a first musical similarity comparison. Different spellings and non-musical labels could distort the results, so their effect is worth checking. The original tags and weights should remain available for comparison.
    """)
    return


if __name__ == "__main__":
    app.run()

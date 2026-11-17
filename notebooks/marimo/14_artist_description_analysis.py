import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS
    from sonora.similarity.descriptions import clean_artist_description

    paths = DEFAULT_DATA_PATHS
    return clean_artist_description, mo, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Artist description analysis

    How much useful text does Last.fm provide for describing artists?
    """)
    return


@app.cell
def _(clean_artist_description, paths, pl):
    artists = pl.read_parquet(paths.curated_artists)
    descriptions = pl.read_parquet(paths.curated_artist_descriptions)

    if artists["artist_id"].n_unique() != artists.height:
        raise ValueError("The artist catalogue contains duplicate IDs")
    if descriptions["artist_id"].n_unique() != descriptions.height:
        raise ValueError("The description dataset contains duplicate artist IDs")
    if set(artists["artist_id"]) != set(descriptions["artist_id"]):
        raise ValueError("The description dataset does not match the artist catalogue")

    descriptions = descriptions.with_columns(
        pl.col("summary")
        .map_elements(clean_artist_description, return_dtype=pl.String)
        .fill_null("")
        .alias("summary_text"),
        pl.col("content")
        .map_elements(clean_artist_description, return_dtype=pl.String)
        .fill_null("")
        .alias("content_text"),
    ).with_columns(
        pl.col("summary_text").str.count_matches(r"\b\w+\b").alias("summary_words"),
        pl.col("content_text").str.count_matches(r"\b\w+\b").alias("content_words"),
    )
    return (descriptions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many artists have descriptions?
    """)
    return


@app.cell
def _(descriptions, mo, pl):
    _total = descriptions.height
    _summary = descriptions.filter(pl.col("summary_words") > 0).height
    _content = descriptions.filter(pl.col("content_words") > 0).height
    _none = descriptions.filter(
        (pl.col("summary_words") == 0) & (pl.col("content_words") == 0)
    ).height

    mo.md(f"""
    - **{_content:,} of {_total:,} artists ({_content / _total:.1%})** have nonempty full-description text.
    - **{_summary:,}** have summaries; **{_none:,}** have neither.
    """)
    return


@app.cell
def _(descriptions, pl):
    status_counts = (
        descriptions.group_by("status")
        .agg(pl.len().alias("Artists"))
        .sort("Artists", descending=True)
    )
    status_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does description coverage depend on listening frequency?
    """)
    return


@app.cell
def _(descriptions, paths, pl):
    listens = (
        pl.scan_parquet(paths.curated_listening_events)
        .group_by("artist_id")
        .agg(pl.len().alias("total_listens"))
        .collect()
    )
    with_listens = descriptions.join(listens, on="artist_id", how="left").with_columns(
        pl.col("total_listens").fill_null(0)
    )
    groups = [
        ("0", 0, 1),
        ("1-2", 1, 3),
        ("3-9", 3, 10),
        ("10-29", 10, 30),
        ("30-99", 30, 100),
        ("100+", 100, None),
    ]
    rows = []
    for label, lower, upper in groups:
        group = with_listens.filter(pl.col("total_listens") >= lower)
        if upper is not None:
            group = group.filter(pl.col("total_listens") < upper)
        if not group.height:
            continue
        described = group.filter(pl.col("content_words") > 0).height
        rows.append(
            {
                "Total listens": label,
                "Artists": group.height,
                "With text (%)": round(100 * described / group.height, 1),
            }
        )
    pl.DataFrame(rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How much text is available?

    Word counts exclude HTML markup and Last.fm attribution text.
    """)
    return


@app.cell
def _(descriptions, pl):
    rows = []
    for _label, column in (("Summary", "summary_words"), ("Full description", "content_words")):
        lengths = descriptions.filter(pl.col(column) > 0)[column]
        rows.append(
            {
                "Field": _label,
                "Artists": lengths.len(),
                "Median words": round(lengths.median()) if not lengths.is_empty() else None,
                "90th percentile": round(lengths.quantile(0.9)) if not lengths.is_empty() else None,
                "Under 30 words": int((lengths < 30).sum()),
            }
        )
    pl.DataFrame(rows)
    return


@app.cell
def _(descriptions, mo, pl):
    _longer = descriptions.filter(
        pl.col("content_words") > pl.col("summary_words")
    ).height
    _long = descriptions.filter(pl.col("content_words") > 1000).height
    mo.md(f"""
    Full descriptions contain more text than summaries for **{_longer:,} artists**.
    **{_long:,} artists** have descriptions longer than 1,000 words.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What formatting and attribution text appears in the responses?
    """)
    return


@app.cell
def _(descriptions, mo, pl):
    _content = pl.col("content").fill_null("")
    _html = descriptions.filter(_content.str.contains(r"<[^>]+>")).height
    _lastfm = descriptions.filter(
        _content.str.contains("Read more on Last.fm", literal=True)
    ).height
    _license = descriptions.filter(
        _content.str.contains("User-contributed text", literal=True)
    ).height
    mo.md(f"""
    - **{_html:,}** full descriptions contain HTML tags.
    - **{_lastfm:,}** contain a Last.fm attribution link.
    - **{_license:,}** contain licence text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which descriptions are shortest?
    """)
    return


@app.cell
def _(descriptions, pl):
    _columns = ["canonical_name", "status", "summary_words", "content_words"]
    (
        descriptions.filter(pl.col("content_words") > 0)
        .sort("content_words")
        .select(_columns)
        .head(10)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What do individual descriptions say?
    """)
    return


@app.cell
def _(descriptions, mo):
    _counts = descriptions.group_by("canonical_name").len(name="name_count")
    _options = {}
    for _row in descriptions.join(
        _counts, on="canonical_name", how="left", validate="m:1"
    ).sort("canonical_name").iter_rows(named=True):
        _label = _row["canonical_name"]
        if _row["name_count"] > 1:
            _label = f"{_label} [{_row['artist_id'][:8]}]"
        _options[_label] = _row["artist_id"]

    artist_selector = mo.ui.dropdown(
        options=_options,
        value="Silvana Estrada" if "Silvana Estrada" in _options else next(iter(_options)),
        searchable=True,
        label="Artist",
        full_width=True,
    )
    artist_selector
    return (artist_selector,)


@app.cell
def _(artist_selector, descriptions, mo, pl):
    _artist = descriptions.filter(
        pl.col("artist_id") == artist_selector.value
    ).row(0, named=True)

    def _excerpt(value, max_characters):
        if not value:
            return "No description."
        return value[:max_characters] + ("..." if len(value) > max_characters else "")

    mo.vstack(
        [
            mo.md(
                f"**Status:** {_artist['status']}  |  "
                f"**Summary:** {_artist['summary_words']} words  |  "
                f"**Full:** {_artist['content_words']} words"
            ),
            mo.md("**Summary**"),
            mo.md(_excerpt(_artist["summary_text"], 1200)),
            mo.md("**Full description**"),
            mo.md(_excerpt(_artist["content_text"], 2400)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

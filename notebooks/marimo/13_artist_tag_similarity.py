import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS
    from sonora.similarity.tags import build_artist_tag_vectors, find_similar_artists

    paths = DEFAULT_DATA_PATHS
    return build_artist_tag_vectors, find_similar_artists, mo, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Do Last.fm tag weights improve artist similarity?

    Compare **binary tags** (all tags count equally) with **weighted tags** (using Last.fm weights).
    Both use cosine similarity and the original tag names.
    """)
    return


@app.cell
def _(build_artist_tag_vectors, paths, pl):
    artist_tags = pl.read_parquet(paths.curated_artist_tags)
    vectors = build_artist_tag_vectors(artist_tags)
    return artist_tags, vectors


@app.cell(hide_code=True)
def _(artist_tags, mo, vectors):
    mo.md(f"""
    **{vectors.artists.height:,} of {artist_tags.height:,} artists** have usable tags.
    Artists without tags are excluded from the rankings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which artists look similar?

    Select an artist to compare its top ten matches. Higher scores mean more similar artists.
    """)
    return


@app.cell
def _(mo, vectors):
    _name_counts = vectors.artists.group_by("canonical_name").len(name="name_count")
    _options = {}
    for _row in vectors.artists.join(
        _name_counts, on="canonical_name", how="left", validate="m:1"
    ).iter_rows(named=True):
        _label = _row["canonical_name"]
        if _row["name_count"] > 1:
            _label = f"{_label} [{_row['artist_id'][:8]}]"
        _options[_label] = _row["artist_id"]

    artist_selector = mo.ui.dropdown(
        options=_options,
        value="Portishead" if "Portishead" in _options else next(iter(_options)),
        searchable=True,
        label="Artist",
        full_width=True,
    )
    artist_selector
    return (artist_selector,)


@app.cell
def _(artist_selector, artist_tags, find_similar_artists, mo, pl, vectors):
    _selected_id = artist_selector.value
    _row = artist_tags.filter(pl.col("artist_id") == _selected_id).row(0, named=True)
    _top_tags = sorted(_row["tags"], key=lambda tag: -tag["weight"])[:8]
    _tag_text = ", ".join(f"{tag['name']} ({tag['weight']})" for tag in _top_tags)

    _binary = find_similar_artists(
        vectors, _selected_id, representation="binary", top_k=10
    )
    _weighted = find_similar_artists(
        vectors, _selected_id, representation="weighted", top_k=10
    )
    _shared = len(set(_binary["artist_id"]) & set(_weighted["artist_id"]))
    _count = max(_binary.height, _weighted.height)
    _rows = []
    for _index in range(_count):
        _binary_row = (
            _binary.row(_index, named=True) if _index < _binary.height else None
        )
        _weighted_row = (
            _weighted.row(_index, named=True) if _index < _weighted.height else None
        )
        _rows.append(
            {
                "Rank": _index + 1,
                "Binary": _binary_row["canonical_name"] if _binary_row else "",
                "Binary score": (
                    round(_binary_row["similarity"], 3) if _binary_row else None
                ),
                "Weighted": _weighted_row["canonical_name"] if _weighted_row else "",
                "Weighted score": (
                    round(_weighted_row["similarity"], 3) if _weighted_row else None
                ),
            }
        )

    mo.vstack(
        [
            mo.md(f"**Top tags:** {_tag_text}"),
            mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=max(1, _count))
            if _rows
            else mo.md("No other artists share tags with this artist."),
            mo.md(f"**{_shared} artists** appear in both top-ten lists."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does this tell us?

    Different rankings show that tag weights matter, but not which approach is better.
    Check familiar artists to see which matches make more musical sense, especially when broad or noisy tags appear.
    """)
    return


if __name__ == "__main__":
    app.run()

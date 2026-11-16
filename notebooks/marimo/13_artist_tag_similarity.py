import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS
    from sonora.similarity.tags import (
        build_artist_tag_vectors,
        clean_artist_tags,
        find_similar_artists,
    )

    paths = DEFAULT_DATA_PATHS
    return (
        build_artist_tag_vectors,
        clean_artist_tags,
        find_similar_artists,
        mo,
        paths,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How similar are artists?

    Compare artist similarity using Last.fm tags and descriptions.
    """)
    return


@app.cell
def _(build_artist_tag_vectors, paths, pl):
    artist_tags = pl.read_parquet(paths.curated_artist_tags)
    vectors = build_artist_tag_vectors(artist_tags)
    return artist_tags, vectors


@app.cell
def _(paths):
    from sonora.similarity.evaluation import load_artist_similarity_evaluation

    evaluation_snapshot = (
        load_artist_similarity_evaluation(paths=paths)
        if paths.artist_similarity_evaluation.exists()
        else None
    )
    candidate_ids = (
        set(evaluation_snapshot["candidate_artist_ids"])
        if evaluation_snapshot is not None
        else None
    )
    return candidate_ids, evaluation_snapshot


@app.cell
def _(find_similar_artists, pl):
    def rank_tag_candidates(
        vectors, artist_id, *, representation, candidate_ids=None, top_k=10
    ):
        if candidate_ids is None:
            return find_similar_artists(
                vectors, artist_id, representation=representation, top_k=top_k
            )
        matches = find_similar_artists(
            vectors,
            artist_id,
            representation=representation,
            top_k=vectors.artists.height,
        )
        return matches.filter(pl.col("artist_id").is_in(list(candidate_ids))).head(top_k)

    return (rank_tag_candidates,)


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

    Compare the top ten matches. When an evaluation snapshot exists, results
    are limited to its frozen top 500 artists.
    """)
    return


@app.cell(hide_code=True)
def _(candidate_ids, mo, pl, vectors):
    _artists = vectors.artists
    if candidate_ids is not None:
        _artists = _artists.filter(pl.col("artist_id").is_in(candidate_ids))
    _name_counts = _artists.group_by("canonical_name").len(name="name_count")
    _options = {}
    for _row in _artists.join(
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


@app.cell(hide_code=True)
def _(
    artist_selector,
    artist_tags,
    candidate_ids,
    mo,
    pl,
    rank_tag_candidates,
    vectors,
):
    _selected_id = artist_selector.value
    _row = artist_tags.filter(pl.col("artist_id") == _selected_id).row(0, named=True)
    _top_tags = sorted(_row["tags"], key=lambda tag: -tag["weight"])[:8]
    _tag_text = ", ".join(f"{tag['name']} ({tag['weight']})" for tag in _top_tags)

    _binary = rank_tag_candidates(
        vectors,
        _selected_id,
        representation="binary",
        candidate_ids=candidate_ids,
    )
    _weighted = rank_tag_candidates(
        vectors,
        _selected_id,
        representation="weighted",
        candidate_ids=candidate_ids,
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
    ## Does cleaning the tags change the matches?
    """)
    return


@app.cell
def _(artist_tags, build_artist_tag_vectors):
    cleaned_vectors = build_artist_tag_vectors(artist_tags, clean_tags=True)
    return (cleaned_vectors,)


@app.cell
def _(mo):
    representation_selector = mo.ui.dropdown(
        options={"Binary": "binary", "Weighted": "weighted"},
        value="Binary",
        label="Representation",
    )
    representation_selector
    return (representation_selector,)


@app.cell(hide_code=True)
def _(
    artist_selector,
    artist_tags,
    candidate_ids,
    clean_artist_tags,
    cleaned_vectors,
    mo,
    pl,
    rank_tag_candidates,
    representation_selector,
    vectors,
):
    _selected_id = artist_selector.value
    _tags = (
        artist_tags.filter(pl.col("artist_id") == _selected_id)
        .row(0, named=True)["tags"]
    )
    _cleaned_tags = clean_artist_tags(_tags)
    _raw_top = sorted(_tags, key=lambda tag: -tag["weight"])[:8]
    _cleaned_top = sorted(_cleaned_tags, key=lambda tag: -tag["weight"])[:8]

    def _describe(tags):
        return ", ".join(f"{tag['name']} ({tag['weight']})" for tag in tags) or "None"

    _representation = representation_selector.value
    _raw = rank_tag_candidates(
        vectors,
        _selected_id,
        representation=_representation,
        candidate_ids=candidate_ids,
    )
    if _selected_id in cleaned_vectors.artists["artist_id"].to_list():
        _cleaned = rank_tag_candidates(
            cleaned_vectors,
            _selected_id,
            representation=_representation,
            candidate_ids=candidate_ids,
        )
    else:
        _cleaned = _raw.head(0)

    _rows = []
    for _index in range(max(_raw.height, _cleaned.height)):
        _raw_row = _raw.row(_index, named=True) if _index < _raw.height else None
        _cleaned_row = (
            _cleaned.row(_index, named=True) if _index < _cleaned.height else None
        )
        _rows.append(
            {
                "Rank": _index + 1,
                "Raw": _raw_row["canonical_name"] if _raw_row else "",
                "Raw score": round(_raw_row["similarity"], 3) if _raw_row else None,
                "Cleaned": _cleaned_row["canonical_name"] if _cleaned_row else "",
                "Cleaned score": (
                    round(_cleaned_row["similarity"], 3) if _cleaned_row else None
                ),
            }
        )

    _shared = len(set(_raw["artist_id"]) & set(_cleaned["artist_id"]))
    mo.vstack(
        [
            mo.md(f"**Original tags:** {_describe(_raw_top)}"),
            mo.md(f"**Cleaned tags:** {_describe(_cleaned_top)}"),
            mo.ui.table(
                pl.DataFrame(_rows), selection=None, page_size=max(1, len(_rows))
            )
            if _rows
            else mo.md("No matching artists."),
            mo.md(f"**{_shared} artists** appear in both top-ten lists."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does TF-IDF change the matches?

    TF-IDF gives more importance to tags shared by fewer artists. Compare all three
    representations using cleaned tags.
    """)
    return


@app.cell(hide_code=True)
def _(
    artist_selector,
    candidate_ids,
    cleaned_vectors,
    mo,
    pl,
    rank_tag_candidates,
):
    _selected_id = artist_selector.value
    if _selected_id not in cleaned_vectors.artists["artist_id"].to_list():
        _output = mo.md("This artist has no usable cleaned tags.")
    else:
        _rankings = {
            _label: rank_tag_candidates(
                cleaned_vectors,
                _selected_id,
                representation=_representation,
                candidate_ids=candidate_ids,
            )
            for _label, _representation in (
                ("Binary", "binary"),
                ("Weighted", "weighted"),
                ("TF-IDF", "tfidf"),
            )
        }
        _rows = []
        for _rank in range(max(_ranking.height for _ranking in _rankings.values())):
            _row = {"Rank": _rank + 1}
            for _label, _ranking in _rankings.items():
                _match = (
                    _ranking.row(_rank, named=True) if _rank < _ranking.height else None
                )
                _row[_label] = _match["canonical_name"] if _match else ""
                _row[f"{_label} score"] = (
                    round(_match["similarity"], 3) if _match else None
                )
            _rows.append(_row)
        _output = (
            mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=len(_rows))
            if _rows
            else mo.md("No matching artists.")
        )
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does embedding artist tags improve similarity?

    Compare cleaned tag TF-IDF, BGE-M3 tag embeddings, and BGE-M3 summaries
    over the same artists. Tag embeddings use tag names only, not weights.
    """)
    return


@app.cell
def _(paths, pl):
    from sonora.similarity.embeddings import find_similar_summary_artists

    summary_embeddings = (
        pl.read_parquet(paths.artist_summary_embeddings)
        if paths.artist_summary_embeddings.exists()
        else None
    )
    return find_similar_summary_artists, summary_embeddings


@app.cell
def _(paths, pl):
    from sonora.similarity.embeddings import find_similar_tag_artists

    tag_embeddings = (
        pl.read_parquet(paths.artist_tag_embeddings)
        if paths.artist_tag_embeddings.exists()
        else None
    )
    return find_similar_tag_artists, tag_embeddings


@app.cell(hide_code=True)
def _(
    artist_selector,
    artist_tags,
    candidate_ids,
    cleaned_vectors,
    find_similar_summary_artists,
    find_similar_tag_artists,
    mo,
    pl,
    rank_tag_candidates,
    summary_embeddings,
    tag_embeddings,
):
    if summary_embeddings is None or tag_embeddings is None:
        _output = mo.md("Generate both summary and tag embeddings to compare results.")
    else:
        _shared_ids = (
            set(cleaned_vectors.artists["artist_id"])
            & set(summary_embeddings["artist_id"])
            & set(tag_embeddings["artist_id"])
        )
        if candidate_ids is not None:
            _shared_ids &= candidate_ids
        _selected_id = artist_selector.value
        if _selected_id not in _shared_ids:
            _output = mo.md("This artist does not have all three representations.")
        else:
            _tfidf = rank_tag_candidates(
                cleaned_vectors,
                _selected_id,
                representation="tfidf",
                candidate_ids=_shared_ids,
            )
            _tag_bge = find_similar_tag_artists(
                tag_embeddings, artist_tags, _selected_id, candidate_ids=_shared_ids
            )
            _summary_bge = find_similar_summary_artists(
                summary_embeddings, artist_tags, _selected_id, candidate_ids=_shared_ids
            )
            _rankings = {
                "TF-IDF": _tfidf,
                "BGE-M3 tags": _tag_bge,
                "BGE-M3 summary": _summary_bge,
            }
            _rows = []
            for _index in range(max(ranking.height for ranking in _rankings.values())):
                _row = {"Rank": _index + 1}
                for _label, _ranking in _rankings.items():
                    _match = (
                        _ranking.row(_index, named=True)
                        if _index < _ranking.height
                        else None
                    )
                    _row[_label] = _match["canonical_name"] if _match else ""
                    _row[f"{_label} score"] = (
                        round(_match["similarity"], 3) if _match else None
                    )
                _rows.append(_row)
            _output = (
                mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=len(_rows))
                if _rows
                else mo.md("No similar artists found.")
            )
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How well do the rankings match known similar artists?

    Compare TF-IDF, BGE-M3 tags, and their equal-weight RRF (k=60) against
    listener and independent LLM references. Fusion uses full rankings within
    the frozen top 500; only the final results are limited to ten artists.
    """)
    return


@app.cell(hide_code=True)
def _(
    artist_tags,
    cleaned_vectors,
    evaluation_snapshot,
    find_similar_tag_artists,
    mo,
    paths,
    pl,
    rank_tag_candidates,
    tag_embeddings,
):
    from sonora.similarity.evaluation import (
        load_artist_similarity_reference_sets,
        score_artist_similarity,
    )
    from sonora.similarity.fusion import reciprocal_rank_fusion

    reference_sets = None
    reference_rankings = {"TF-IDF": {}, "BGE-M3 tags": {}, "RRF": {}}
    reference_scores = None

    if evaluation_snapshot is None:
        _output = mo.md("Create the artist similarity evaluation snapshot first.")
    elif tag_embeddings is None:
        _output = mo.md("Generate BGE-M3 tag embeddings first.")
    else:
        reference_sets = load_artist_similarity_reference_sets(paths=paths)
        _candidate_ids = set(evaluation_snapshot["candidate_artist_ids"])
        _tag_ids = set(cleaned_vectors.artists["artist_id"])
        _bge_ids = set(tag_embeddings["artist_id"])
        for _reference in evaluation_snapshot["references"]:
            _artist_id = _reference["artist_id"]
            _full_rankings = {}
            if _artist_id in _tag_ids:
                _matches = rank_tag_candidates(
                    cleaned_vectors,
                    _artist_id,
                    representation="tfidf",
                    candidate_ids=_candidate_ids,
                    top_k=len(_candidate_ids),
                )
                _full_rankings["TF-IDF"] = _matches["artist_id"].to_list()
            if _artist_id in _bge_ids:
                _matches = find_similar_tag_artists(
                    tag_embeddings,
                    artist_tags,
                    _artist_id,
                    top_k=len(_candidate_ids),
                    candidate_ids=_candidate_ids,
                )
                _full_rankings["BGE-M3 tags"] = _matches["artist_id"].to_list()

            for _approach, _ranking in _full_rankings.items():
                reference_rankings[_approach][_artist_id] = _ranking[:10]
            if len(_full_rankings) == 2:
                reference_rankings["RRF"][_artist_id] = reciprocal_rank_fusion(
                    list(_full_rankings.values()), top_k=10
                )

        reference_scores = pl.concat(
            [
                score_artist_similarity(_snapshot, reference_rankings).with_columns(
                    pl.lit(_label).alias("Reference set")
                )
                for _label, _snapshot in reference_sets.items()
            ]
        )
        _totals = (
            reference_scores.group_by("Reference set", "Approach")
            .agg(
                pl.col("Available").sum().alias("Artists evaluated"),
                (pl.col("Top 5").filter(pl.col("Available")).mean() / 5)
                .round(3)
                .alias("Recall@5"),
                (pl.col("Top 10").filter(pl.col("Available")).mean() / 5)
                .round(3)
                .alias("Recall@10"),
                pl.col("NDCG@10").mean().round(3).alias("NDCG@10"),
            )
            .sort("Reference set", "Approach")
        )
        _output = mo.ui.table(_totals, selection=None, page_size=6)

    _output
    return reference_rankings, reference_scores, reference_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which approach works better across artists?

    Each approach shows **top 5 / top 10 / NDCG@10**. Counts are matches
    out of five references. RRF gain compares fusion with the better of TF-IDF
    and BGE-M3, using NDCG@10. Positive means fusion performs better.
    The largest changes appear first.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl, reference_scores):
    if reference_scores is None:
        _output = mo.md("No evaluation results available.")
    else:
        _sections = []
        for _set in reference_scores["Reference set"].unique(maintain_order=True):
            _scores = reference_scores.filter(pl.col("Reference set") == _set)
            _by_artist = {}
            for _score in _scores.iter_rows(named=True):
                _by_artist.setdefault(_score["Artist"], {})[_score["Approach"]] = _score

            def _metrics(score):
                if score is None or not score["Available"]:
                    return "Unavailable"
                return (
                    f"{score['Top 5']}/5 · {score['Top 10']}/5 · "
                    f"{score['NDCG@10']:.3f}"
                )

            _labels = ("TF-IDF", "BGE-M3 tags", "RRF")
            _rows = []
            _wins = {**dict.fromkeys(_labels, 0), "Tie": 0}
            for _artist, _approaches in _by_artist.items():
                _scores_by_approach = {
                    _label: _approaches.get(_label) for _label in _labels
                }
                _gain = None
                _winner = "Unavailable"
                if all(
                    _score is not None and _score["Available"]
                    for _score in _scores_by_approach.values()
                ):
                    _ndcg = {
                        _label: round(_score["NDCG@10"], 3)
                        for _label, _score in _scores_by_approach.items()
                    }
                    _best = max(_ndcg.values())
                    _leaders = [
                        _label for _label, _value in _ndcg.items() if _value == _best
                    ]
                    _winner = _leaders[0] if len(_leaders) == 1 else "Tie"
                    _wins[_winner] += 1
                    _gain = round(
                        _ndcg["RRF"] - max(_ndcg["TF-IDF"], _ndcg["BGE-M3 tags"]),
                        3,
                    )

                _rows.append(
                    {
                        "Artist": _artist,
                        **{
                            _label: _metrics(_scores_by_approach[_label])
                            for _label in _labels
                        },
                        "Best NDCG": _winner,
                        "RRF gain": _gain,
                    }
                )

            _rows.sort(
                key=lambda row: (
                    row["RRF gain"] is None,
                    -abs(row["RRF gain"] or 0),
                    row["Artist"],
                )
            )
            _compared = sum(_wins.values())
            _sections.extend(
                [
                    mo.md(
                        f"**{_set}** - NDCG@10 wins: RRF {_wins['RRF']}, "
                        f"BGE-M3 {_wins['BGE-M3 tags']}, TF-IDF {_wins['TF-IDF']}, "
                        f"ties {_wins['Tie']} ({_compared} artists compared)."
                    ),
                    mo.ui.table(
                        pl.DataFrame(_rows), selection=None, page_size=len(_rows)
                    ),
                ]
            )
        _output = mo.vstack(_sections)
    _output
    return


@app.cell(hide_code=True)
def _(mo, reference_sets):
    reference_set_selector = None
    if reference_sets is not None:
        reference_set_selector = mo.ui.dropdown(
            options=list(reference_sets),
            value=next(iter(reference_sets)),
            label="Reference set",
        )
    reference_set_selector
    return (reference_set_selector,)


@app.cell(hide_code=True)
def _(mo, reference_set_selector, reference_sets):
    reference_selector = None
    if reference_set_selector is not None:
        _snapshot = reference_sets[reference_set_selector.value]
        _options = {
            row["canonical_name"]: row["artist_id"]
            for row in _snapshot["references"]
        }
        reference_selector = mo.ui.dropdown(
            options=_options,
            value=next(iter(_options)),
            searchable=True,
            label="Reference artist",
            full_width=True,
        )
    reference_selector
    return (reference_selector,)


@app.cell(hide_code=True)
def _(
    artist_tags,
    mo,
    pl,
    reference_rankings,
    reference_scores,
    reference_selector,
    reference_set_selector,
    reference_sets,
):
    if reference_selector is None:
        _output = None
    else:
        _set_label = reference_set_selector.value
        _snapshot = reference_sets[_set_label]
        _artist_id = reference_selector.value
        _reference = next(
            row
            for row in _snapshot["references"]
            if row["artist_id"] == _artist_id
        )
        _expected = {match["artist_id"] for match in _reference["matches"]}
        _names = dict(artist_tags.select("artist_id", "canonical_name").iter_rows())
        _rows = []
        for _rank in range(10):
            _row = {"Rank": _rank + 1}
            for _approach in ("TF-IDF", "BGE-M3 tags", "RRF"):
                _matches = reference_rankings[_approach].get(_artist_id, [])
                _match_id = _matches[_rank] if _rank < len(_matches) else None
                _name = _names[_match_id] if _match_id is not None else ""
                _row[_approach] = (
                    f"✓ {_name}" if _match_id in _expected else _name
                )
            _rows.append(_row)

        _artist_scores = reference_scores.filter(
            (pl.col("Reference set") == _set_label)
            & (pl.col("Artist") == _reference["canonical_name"])
        ).sort("Approach")
        _counts = " | ".join(
            (
                f"{row['Approach']}: {row['Top 5']}/5 in top 5, "
                f"{row['Top 10']}/5 in top 10, "
                f"NDCG@10 {row['NDCG@10']:.3f}"
                if row["Available"]
                else f"{row['Approach']}: unavailable"
            )
            for row in _artist_scores.iter_rows(named=True)
        )
        _reference_names = ", ".join(
            f"{rank}. {match['canonical_name']}"
            for rank, match in enumerate(_reference["matches"], start=1)
        )
        _output = mo.vstack(
            [
                mo.md(f"**Reference matches:** {_reference_names}"),
                mo.md(_counts),
                mo.ui.table(pl.DataFrame(_rows), selection=None, page_size=10),
                mo.md("✓ = reference match. Rows 1–5 are the top five results."),
            ]
        )
    _output
    return


if __name__ == "__main__":
    app.run()

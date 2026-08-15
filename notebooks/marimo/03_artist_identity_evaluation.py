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
    # Artist identity evaluation

    A first look at which artist pairs the matcher finds and what evidence supports them.
    """)
    return


@app.cell
def _():
    from itertools import combinations

    import polars as pl

    from sonora.data.artist_candidates import (
        collect_artist_aliases,
        generate_artist_candidate_pairs,
    )
    from sonora.data.artist_evidence import score_artist_candidate_evidence
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return (
        collect_artist_aliases,
        combinations,
        generate_artist_candidate_pairs,
        paths,
        pl,
        score_artist_candidate_evidence,
    )


@app.cell
def _(mo, paths):
    dataset_path = paths.listening_events_clean
    mo.stop(
        not dataset_path.is_file(),
        mo.callout(
            f"Clean listening events not found at `{dataset_path}`. Build the interim dataset first.",
            kind="danger",
        ),
    )
    return (dataset_path,)


@app.cell(hide_code=True)
def _(dataset_path, mo):
    mo.md(f"""
    **Input:** `{dataset_path}`
    """)
    return


@app.cell
def _(
    collect_artist_aliases,
    dataset_path,
    generate_artist_candidate_pairs,
    pl,
    score_artist_candidate_evidence,
):
    events = pl.scan_parquet(dataset_path)
    aliases = collect_artist_aliases(events)
    candidates = generate_artist_candidate_pairs(aliases)
    evidence = score_artist_candidate_evidence(events, candidates)
    return aliases, candidates, evidence, events


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Candidate generation

    First, check how much the candidate step cuts down the number of artist pairs.
    """)
    return


@app.cell
def _(aliases, candidates, mo):
    _alias_count = aliases.height
    _candidate_count = candidates.height
    _all_pair_count = _alias_count * (_alias_count - 1) // 2
    _candidate_pct = (
        _candidate_count / _all_pair_count * 100 if _all_pair_count else 0.0
    )
    _reduction_pct = 100 - _candidate_pct

    mo.md(f"""
    - **Observed artist aliases:** {_alias_count:,}
    - **All possible alias pairs:** {_all_pair_count:,}
    - **Generated candidate pairs:** {_candidate_count:,}
    - **Candidate share of all pairs:** {_candidate_pct:.3f}%
    - **Pair reduction:** {_reduction_pct:.3f}%
    """)
    return


@app.cell
def _(candidates, pl):
    _blocking_counts = candidates.select(
        pl.col("blocking_exact_name_match").sum().alias("exact_name"),
        (pl.col("blocking_shared_token_count") > 0).sum().alias("shared_token"),
        (pl.col("blocking_shared_ngram_count") > 0).sum().alias("shared_ngram"),
    ).row(0, named=True)

    blocking_summary = pl.DataFrame(
        {
            "retrieval_signal": [
                "Same normalized name",
                "Shared word",
                "Shared character n-grams",
            ],
            "candidate_pairs": [
                _blocking_counts["exact_name"],
                _blocking_counts["shared_token"],
                _blocking_counts["shared_ngram"],
            ],
        }
    )

    blocking_summary
    return (blocking_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A pair can be found by more than one signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pairs from the earlier EDA

    These pairs stood out earlier and give us a useful first check of what the matcher is finding.
    """)
    return


@app.cell
def _(aliases, evidence, pl):
    _reference_pairs = [
        ("Samuel Uria", "Samuel Úria"),
        ("King Gizzard & The Lizard Wizard", "King Gizzard And The Lizard Wizard"),
        ("Conjunto Corona", "Corona"),
        ("Luisa Sobral", "Luísa Sobral"),
        ("Miguel Luz", "Mike Lyte"),
        ("Blu", "Blu & Exile"),
    ]
    _observed_names = set(aliases.get_column("observed_artist_name").to_list())
    _rows = []

    for _left, _right in _reference_pairs:
        _match = evidence.filter(
            (
                (pl.col("observed_artist_name_left") == _left)
                & (pl.col("observed_artist_name_right") == _right)
            )
            | (
                (pl.col("observed_artist_name_left") == _right)
                & (pl.col("observed_artist_name_right") == _left)
            )
        )
        _row = _match.row(0, named=True) if not _match.is_empty() else None
        _rows.append(
            {
                "reference_left": _left,
                "reference_right": _right,
                "left_observed": _left in _observed_names,
                "right_observed": _right in _observed_names,
                "retrieved_as_candidate": _row is not None,
                "name_levenshtein_similarity": (
                    _row["name_levenshtein_similarity"] if _row else None
                ),
                "name_wratio": _row["name_wratio"] if _row else None,
                "name_token_set_ratio": _row["name_token_set_ratio"] if _row else None,
                "shared_track_count": _row["shared_track_count"] if _row else None,
                "track_containment": _row["track_containment"] if _row else None,
                "shared_track_mbid_count": (
                    _row["shared_track_mbid_count"] if _row else None
                ),
                "shared_album_count": _row["shared_album_count"] if _row else None,
                "album_containment": _row["album_containment"] if _row else None,
                "artist_mbid_relation": (
                    _row["artist_mbid_relation"] if _row else None
                ),
            }
        )

    reference_pair_summary = pl.DataFrame(_rows)
    reference_pair_summary
    return (reference_pair_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Shared artist MBIDs

    If two aliases share an artist MBID, we would expect the candidate step to find them. This shows how often that happens.
    """)
    return


@app.cell
def _(candidates, combinations, events, pl):
    _mbid_aliases = (
        events.filter(pl.col("artist_mbid").is_not_null())
        .group_by("artist_mbid")
        .agg(pl.col("artist_name").unique().sort().alias("artist_aliases"))
        .filter(pl.col("artist_aliases").list.len() > 1)
        .collect()
    )

    _candidate_pair_keys = {
        frozenset((_left, _right))
        for _left, _right in candidates.select(
            "observed_artist_name_left",
            "observed_artist_name_right",
        ).iter_rows()
    }
    _mbid_pair_keys = {
        frozenset((_left, _right))
        for _aliases in _mbid_aliases.get_column("artist_aliases").to_list()
        for _left, _right in combinations(_aliases, 2)
    }
    _retrieved_mbid_pairs = _mbid_pair_keys & _candidate_pair_keys
    _mbid_recall = (
        len(_retrieved_mbid_pairs) / len(_mbid_pair_keys)
        if _mbid_pair_keys
        else None
    )

    mbid_candidate_recall = pl.DataFrame(
        {
            "metric": [
                "Artist MBIDs used by multiple names",
                "Name pairs sharing an artist MBID",
                "Shared-MBID pairs found as candidates",
                "Shared-MBID recall",
            ],
            "value": [
                str(_mbid_aliases.height),
                str(len(_mbid_pair_keys)),
                str(len(_retrieved_mbid_pairs)),
                f"{_mbid_recall:.1%}" if _mbid_recall is not None else "n/a",
            ],
        }
    )

    mbid_candidate_recall
    return (mbid_candidate_recall,)


@app.cell
def _(candidates, combinations, events, pl):
    _mbid_aliases = (
        events.filter(pl.col("artist_mbid").is_not_null())
        .group_by("artist_mbid")
        .agg(pl.col("artist_name").unique().sort().alias("artist_aliases"))
        .filter(pl.col("artist_aliases").list.len() > 1)
        .collect()
    )
    _candidate_pair_keys = {
        frozenset((_left, _right))
        for _left, _right in candidates.select(
            "observed_artist_name_left",
            "observed_artist_name_right",
        ).iter_rows()
    }
    _missed_rows = []
    for _artist_mbid, _aliases in _mbid_aliases.iter_rows():
        for _left, _right in combinations(_aliases, 2):
            if frozenset((_left, _right)) not in _candidate_pair_keys:
                _missed_rows.append(
                    {
                        "artist_mbid": _artist_mbid,
                        "observed_artist_name_left": _left,
                        "observed_artist_name_right": _right,
                    }
                )

    missed_shared_mbid_pairs = pl.DataFrame(
        _missed_rows,
        schema={
            "artist_mbid": pl.String,
            "observed_artist_name_left": pl.String,
            "observed_artist_name_right": pl.String,
        },
    )
    missed_shared_mbid_pairs.head(30)
    return (missed_shared_mbid_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If useful pairs are missing here, adding artist MBIDs as another candidate source may be better than making fuzzy matching broader.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evidence ranges

    These ranges give us a feel for the signals before we choose any thresholds.
    """)
    return


@app.cell
def _(evidence, pl):
    _metrics = [
        "name_levenshtein_similarity",
        "name_wratio",
        "name_token_set_ratio",
        "shared_track_count",
        "track_containment",
        "shared_track_mbid_count",
        "track_mbid_containment",
        "shared_album_count",
        "album_containment",
    ]
    _distribution_rows = []
    for _metric in _metrics:
        _values = evidence.get_column(_metric).drop_nulls()
        _distribution_rows.append(
            {
                "metric": _metric,
                "non_null": _values.len(),
                "min": _values.min(),
                "p25": _values.quantile(0.25),
                "median": _values.median(),
                "p75": _values.quantile(0.75),
                "max": _values.max(),
            }
        )

    evidence_distribution = pl.DataFrame(_distribution_rows)
    evidence_distribution
    return (evidence_distribution,)


@app.cell
def _(evidence, pl):
    identifier_relation_summary = (
        evidence.group_by("artist_mbid_relation")
        .agg(pl.len().alias("candidate_pairs"))
        .sort("candidate_pairs", descending=True)
    )

    identifier_relation_summary
    return (identifier_relation_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Pairs sharing tracks, albums, or IDs

    These pairs share tracks, albums, or IDs, so they are useful beyond simple name similarity.
    """)
    return


@app.cell
def _(evidence, pl):
    catalogue_supported_pairs = (
        evidence.filter(
            (pl.col("shared_track_count") > 0)
            | (pl.col("shared_track_mbid_count") > 0)
            | (pl.col("shared_album_count") > 0)
            | (pl.col("shared_artist_mbid_count") > 0)
        )
        .sort(
            [
                "shared_artist_mbid_count",
                "shared_track_mbid_count",
                "shared_track_count",
                "track_containment",
                "name_wratio",
            ],
            descending=True,
            nulls_last=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "name_levenshtein_similarity",
            "name_wratio",
            "shared_track_count",
            "track_containment",
            "shared_track_mbid_count",
            "track_mbid_containment",
            "shared_album_count",
            "album_containment",
            "shared_artist_mbid_count",
            "artist_mbid_relation",
        )
    )

    catalogue_supported_pairs.head(40)
    return (catalogue_supported_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Similar names without catalogue support

    These pairs show where similar names may be misleading on their own.
    """)
    return


@app.cell
def _(evidence, pl):
    lexical_only_pairs = (
        evidence.filter(
            (pl.col("shared_track_count") == 0)
            & (pl.col("shared_track_mbid_count") == 0)
            & (pl.col("shared_album_count") == 0)
            & (pl.col("shared_artist_mbid_count") == 0)
        )
        .sort(
            ["name_wratio", "name_levenshtein_similarity"],
            descending=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "name_levenshtein_similarity",
            "name_wratio",
            "name_token_set_ratio",
            "artist_mbid_relation",
            "blocking_exact_name_match",
            "blocking_shared_token_count",
            "blocking_shared_ngram_count",
        )
    )

    lexical_only_pairs.head(40)
    return (lexical_only_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Artist MBID conflicts

    These are worth checking when the names or catalogues otherwise look similar.
    """)
    return


@app.cell
def _(evidence, pl):
    artist_mbid_conflicts = (
        evidence.filter(pl.col("artist_mbid_relation") == "conflict")
        .sort(
            ["name_wratio", "track_containment", "shared_track_count"],
            descending=True,
            nulls_last=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "name_levenshtein_similarity",
            "name_wratio",
            "shared_track_count",
            "track_containment",
            "shared_track_mbid_count",
            "shared_album_count",
            "album_containment",
        )
    )

    artist_mbid_conflicts.head(40)
    return (artist_mbid_conflicts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Fuzzy metric differences

    These are the pairs where the fuzzy metrics disagree most. They should help us see whether one metric behaves better for this data.
    """)
    return


@app.cell
def _(evidence, pl):
    fuzzy_metric_disagreement = (
        evidence.with_columns(
            (pl.col("name_wratio") - pl.col("name_levenshtein_similarity"))
            .abs()
            .alias("wratio_vs_levenshtein_gap"),
            (pl.col("name_token_set_ratio") - pl.col("name_levenshtein_similarity"))
            .abs()
            .alias("token_set_vs_levenshtein_gap"),
        )
        .with_columns(
            pl.max_horizontal(
                "wratio_vs_levenshtein_gap",
                "token_set_vs_levenshtein_gap",
            ).alias("largest_metric_gap")
        )
        .sort("largest_metric_gap", descending=True)
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "name_levenshtein_similarity",
            "name_wratio",
            "name_token_set_ratio",
            "largest_metric_gap",
            "shared_track_count",
            "track_containment",
            "artist_mbid_relation",
        )
    )

    fuzzy_metric_disagreement.head(40)
    return (fuzzy_metric_disagreement,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to look at next

    - Are we finding the pairs we expect?
    - Are we creating too many obvious false positives?
    - Are shared-MBID pairs being missed?
    - Do the fuzzy metrics behave differently enough to matter?

    We will use these results later to choose thresholds and test track/album suffix handling.
    """)
    return


if __name__ == "__main__":
    app.run()

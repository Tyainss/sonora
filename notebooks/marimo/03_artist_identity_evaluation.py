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

    Checks which artist names are being paired and what supports each match.
    """)
    return


@app.cell
def _():
    import polars as pl

    from sonora.data.artist_candidates import (
        DEFAULT_ARTIST_CANDIDATE_CONFIG,
        collect_artist_aliases,
        generate_artist_candidate_pairs,
    )
    from sonora.data.artist_evidence import score_artist_candidate_evidence
    from sonora.data.artist_resolution import (
        DEFAULT_ARTIST_RESOLUTION_CONFIG,
        cluster_artist_aliases,
        decide_artist_identity,
    )
    from sonora.data.comparison_normalization import normalize_for_comparison
    from sonora.data.curate_artists import build_curated_artist_tables
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    candidate_config = DEFAULT_ARTIST_CANDIDATE_CONFIG
    resolution_config = DEFAULT_ARTIST_RESOLUTION_CONFIG
    return (
        build_curated_artist_tables,
        candidate_config,
        cluster_artist_aliases,
        collect_artist_aliases,
        decide_artist_identity,
        generate_artist_candidate_pairs,
        normalize_for_comparison,
        paths,
        pl,
        resolution_config,
        score_artist_candidate_evidence,
    )


@app.cell
def _(mo, paths):
    dataset_path = paths.listening_events_clean
    mo.stop(
        not dataset_path.is_file(),
        mo.callout(
            f"Clean listening events not found at `{dataset_path}`. Build it first.",
            kind="danger",
        ),
    )
    return (dataset_path,)


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
    candidates = generate_artist_candidate_pairs(events)
    evidence = score_artist_candidate_evidence(events, candidates)
    return aliases, candidates, evidence, events


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Candidate generation
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

    mo.md(f"""
    - **Observed artist names:** {_alias_count:,}
    - **All possible pairs:** {_all_pair_count:,}
    - **Candidate pairs:** {_candidate_count:,}
    - **Candidate share:** {_candidate_pct:.3f}%
    """)
    return


@app.cell
def _(candidates, pl):
    candidate_source_summary = pl.DataFrame(
        {
            "source": [
                "Same normalized name",
                "Shared name word",
                "Shared name n-grams",
                "Shared track title",
            ],
            "candidate_pairs": [
                candidates.get_column("candidate_from_exact_name").sum(),
                candidates.get_column("candidate_from_shared_token").sum(),
                candidates.get_column("candidate_from_shared_ngrams").sum(),
                candidates.get_column("candidate_from_shared_tracks").sum(),
            ],
        }
    )

    candidate_source_summary
    return (candidate_source_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reference pairs

    Manually checked pairs provide a small evaluation set for the automatic resolver. They are used only to inspect resolver behaviour and do not affect clustering or canonical artist tables.
    """)
    return


@app.cell
def _(aliases, evidence, pl):
    _reference_pairs = [
        ("Samuel Uria", "Samuel Úria", "same_artist"),
        (
            "King Gizzard & The Lizard Wizard",
            "King Gizzard And The Lizard Wizard",
            "same_artist",
        ),
        ("Conjunto Corona", "Corona", "same_artist"),
        ("Luisa Sobral", "Luísa Sobral", "same_artist"),
        ("Miguel Luz", "Mike Lyte", "same_artist"),
        ("Travi$ Scott", "Travis Scott", "same_artist"),
        ("(Sandy) Alex G", "Alex G", "same_artist"),
        ("Big Pun", "Big Punisher", "same_artist"),
        ("Blu", "Blu & Exile", "different_artist"),
        ("Kenny Segal", "Milo", "different_artist"),
        ("El-P", "Run the Jewels", "different_artist"),
        ("Kanye West", "¥$", "different_artist"),
        ("Adrianne Lenker", "Big Thief", "different_artist"),
        ("Aphex Twin", "Soul Glo", "different_artist"),
    ]
    _observed_names = set(aliases.get_column("observed_artist_name").to_list())
    _rows = []

    for _left, _right, _expected_relationship in _reference_pairs:
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
                "artist_left": _left,
                "artist_right": _right,
                "expected_relationship": _expected_relationship,
                "left_observed": _left in _observed_names,
                "right_observed": _right in _observed_names,
                "candidate": _row is not None,
                "from_name": (
                    (
                        _row["candidate_from_exact_name"]
                        or _row["candidate_from_shared_token"]
                        or _row["candidate_from_shared_ngrams"]
                    )
                    if _row
                    else None
                ),
                "from_tracks": _row["candidate_from_shared_tracks"] if _row else None,
                "levenshtein": (
                    _row["name_levenshtein_similarity"] if _row else None
                ),
                "wratio": _row["name_wratio"] if _row else None,
                "shared_tracks": _row["shared_track_count"] if _row else None,
                "track_containment": _row["track_containment"] if _row else None,
                "shared_albums": _row["shared_album_count"] if _row else None,
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
    `Miguel Luz` / `Mike Lyte` is a rename case, so track overlap is more useful than the name itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Shared track titles
    """)
    return


@app.cell
def _(candidate_config, events, normalize_for_comparison, pl):
    shared_track_titles = (
        events.select(
            normalize_for_comparison(pl.col("track_name")).alias("track_name"),
            pl.col("artist_name"),
        )
        .drop_nulls("track_name")
        .unique()
        .group_by("track_name")
        .agg(pl.col("artist_name").n_unique().alias("artist_count"))
        .filter(pl.col("artist_count") > 1)
        .with_columns(
            (pl.col("artist_count") <= candidate_config.max_track_block_size).alias(
                "used_for_candidates"
            )
        )
        .sort("artist_count", descending=True)
        .collect()
    )

    shared_track_titles.head(30)
    return (shared_track_titles,)


@app.cell(hide_code=True)
def _(candidate_config, mo):
    mo.md(f"""
    Very common titles are noisy, so tracks used by more than **{candidate_config.max_track_block_size} artists** are ignored here.
    """)
    return


@app.cell
def _(evidence, pl):
    catalogue_only_candidates = (
        evidence.filter(
            pl.col("candidate_from_shared_tracks")
            & ~pl.col("candidate_from_exact_name")
            & ~pl.col("candidate_from_shared_token")
            & ~pl.col("candidate_from_shared_ngrams")
        )
        .sort(
            ["shared_track_count", "track_containment"],
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
            "shared_album_count",
            "album_containment",
            "artist_mbid_relation",
        )
    )

    catalogue_only_candidates.head(40)
    return (catalogue_only_candidates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Track matching finds the `Miguel Luz` / `Mike Lyte` rename that name matching misses. Most of the other strong track-only pairs are collaborations or closely related artists, such as `El-P` / `Run the Jewels`, `Kenny Segal` / `Milo`, `Kanye West` / `¥$`, and `Adrianne Lenker` / `Big Thief`. Shared tracks are useful for recall, but they are not enough to merge artists on their own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evidence ranges
    """)
    return


@app.cell
def _(evidence, pl):
    _metrics = [
        "name_levenshtein_similarity",
        "name_wratio",
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
    No observed artist names share an artist MBID in this dataset, so MBIDs do not add candidate recall here. Conflicts are still useful evidence against a merge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Catalogue-supported pairs
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
    The strongest rows include plausible alias pairs such as the `Samuel Úria`, `King Gizzard`, `Conjunto Corona`, `Luísa Sobral`, `Conan Osiris`, `JAY-Z`, and `Ichiko Aoba` variants. `El-P` / `Run the Jewels` is a useful edge case because Last.fm credits can blur a real solo artist with a group he belongs to. `Aphex Twin` / `Soul Glo` shows that catalogue overlap can also be accidental.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Similar names without catalogue support
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
            "artist_mbid_relation",
            "candidate_from_exact_name",
            "candidate_from_shared_token",
            "candidate_from_shared_ngrams",
        )
    )

    lexical_only_pairs.head(40)
    return (lexical_only_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some real matches have no catalogue overlap at all. Examples include `JAY-Z`, `The Legendary Tigerman`, `Chet Baker`, and `Miles Davis` variants. Name evidence therefore still needs to stand on its own for some aliases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Artist MBID conflicts
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
            "shared_album_count",
            "album_containment",
        )
    )

    artist_mbid_conflicts.head(40)
    return (artist_mbid_conflicts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most strong MBID conflicts are different artists. `Conjunto Corona` / `Corona` is the important exception here: the catalogue evidence is strong enough that an MBID conflict should count against a merge without automatically ruling it out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Levenshtein vs WRatio
    """)
    return


@app.cell
def _(evidence, pl):
    fuzzy_metric_difference = (
        evidence.with_columns(
            (pl.col("name_wratio") - pl.col("name_levenshtein_similarity"))
            .abs()
            .alias("metric_gap")
        )
        .sort("metric_gap", descending=True)
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "name_levenshtein_similarity",
            "name_wratio",
            "metric_gap",
            "shared_track_count",
            "track_containment",
            "artist_mbid_relation",
        )
    )

    fuzzy_metric_difference.head(40)
    return (fuzzy_metric_difference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Levenshtein is useful for small spelling changes. WRatio also handles cases where words move around or one name contains another. We keep both. Token-set ratio is excluded because it gives perfect scores to many obvious subset matches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evidence summary

    Name matching handles most spelling and formatting variants. Track matching adds useful recall for cases such as `Miguel Luz` / `Mike Lyte`, but it also picks up collaborations and related artists, so catalogue overlap needs other evidence before a merge. Levenshtein and WRatio both add useful information, while artist MBIDs are mainly negative evidence in this dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Automatic identity decisions
    """)
    return


@app.cell
def _(pl, resolution_config):
    resolution_settings = pl.DataFrame(
        {
            "setting": [
                "Near-exact Levenshtein",
                "Near-exact WRatio",
                "Name + catalogue WRatio",
                "Name + catalogue shared tracks",
                "Name + catalogue containment",
                "Catalogue merge shared tracks",
                "Catalogue merge containment",
                "Catalogue merge shared-play share",
                "MBID conflict override WRatio",
                "MBID conflict override shared tracks",
                "MBID conflict override containment",
                "Possible-match Levenshtein",
                "Possible-match WRatio",
                "Possible-match shared tracks",
                "Possible-match containment",
            ],
            "value": [
                resolution_config.near_exact_levenshtein,
                resolution_config.near_exact_wratio,
                resolution_config.name_catalogue_wratio,
                resolution_config.name_catalogue_min_shared_tracks,
                resolution_config.name_catalogue_min_track_containment,
                resolution_config.catalogue_merge_min_shared_tracks,
                resolution_config.catalogue_merge_min_track_containment,
                resolution_config.catalogue_merge_min_scrobble_share,
                resolution_config.conflict_override_wratio,
                resolution_config.conflict_override_min_shared_tracks,
                resolution_config.conflict_override_min_track_containment,
                resolution_config.possible_name_levenshtein,
                resolution_config.possible_name_wratio,
                resolution_config.possible_catalogue_min_shared_tracks,
                resolution_config.possible_catalogue_min_track_containment,
            ],
        }
    )
    resolution_settings
    return (resolution_settings,)


@app.cell
def _(decide_artist_identity, evidence, resolution_config):
    decisions = decide_artist_identity(evidence, config=resolution_config)
    return (decisions,)


@app.cell
def _(decisions, pl):
    decision_summary = (
        decisions.group_by("identity_decision", "decision_rule")
        .agg(pl.len().alias("pairs"))
        .sort(["identity_decision", "pairs"], descending=[False, True])
    )
    decision_summary
    return (decision_summary,)


@app.cell
def _(decisions, mo, pl):
    merge_pairs = (
        decisions.filter(pl.col("identity_decision") == "merge")
        .sort(
            ["decision_rule", "name_wratio", "track_containment"],
            descending=[False, True, True],
            nulls_last=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "decision_rule",
            "name_levenshtein_similarity",
            "name_wratio",
            "shared_track_count",
            "track_containment",
            "shared_album_count",
            "artist_mbid_relation",
        )
    )
    mo.ui.table(merge_pairs, pagination=False, selection=None, max_height=600)
    return (merge_pairs,)


@app.cell
def _(decisions, mo, pl):
    possible_matches = (
        decisions.filter(pl.col("identity_decision") == "possible_match")
        .sort(
            ["name_wratio", "track_containment", "shared_track_count"],
            descending=True,
            nulls_last=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "decision_rule",
            "name_levenshtein_similarity",
            "name_wratio",
            "shared_track_count",
            "track_containment",
            "shared_album_count",
            "artist_mbid_relation",
        )
    )
    mo.ui.table(
        possible_matches.head(60),
        pagination=False,
        selection=None,
        max_height=700,
    )
    return (possible_matches,)


@app.cell
def _(decisions, pl, reference_pair_summary):
    _rows = []
    for _reference in reference_pair_summary.iter_rows(named=True):
        _left = _reference["artist_left"]
        _right = _reference["artist_right"]
        _match = decisions.filter(
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
        _decision = _row["identity_decision"] if _row else None
        _expected_relationship = _reference["expected_relationship"]
        _automatic_outcome = (
            "resolved"
            if _expected_relationship == "same_artist" and _decision == "merge"
            else (
                "false_merge"
                if _expected_relationship == "different_artist" and _decision == "merge"
                else (
                    "unresolved"
                    if _expected_relationship == "same_artist"
                    else "kept_separate"
                )
            )
        )
        _rows.append(
            {
                "artist_left": _left,
                "artist_right": _right,
                "expected_relationship": _expected_relationship,
                "decision": _decision,
                "rule": _row["decision_rule"] if _row else None,
                "automatic_outcome": _automatic_outcome,
            }
        )

    reference_pair_decisions = pl.DataFrame(_rows)
    reference_pair_decisions
    return (reference_pair_decisions,)


@app.cell
def _(pl, reference_pair_decisions):
    reference_outcome_summary = (
        reference_pair_decisions.group_by("automatic_outcome")
        .agg(pl.len().alias("pairs"))
        .sort("automatic_outcome")
    )
    reference_outcome_summary
    return (reference_outcome_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Miguel Luz` / `Mike Lyte` has strong overlap on both sides (74% / 62% of scrobbles), while `Kenny Segal` / `Milo` is one-sided (100% / 21%). The bidirectional catalogue rule requires strong overlap on both sides.
    """)
    return


@app.cell
def _(decisions, pl):
    catalogue_only_decisions = (
        decisions.filter(
            pl.col("decision_rule").is_in(
                ["bidirectional_catalogue", "strong_catalogue"]
            )
        )
        .sort(
            ["shared_track_count", "track_containment"],
            descending=True,
        )
        .select(
            "observed_artist_name_left",
            "observed_artist_name_right",
            "left_scrobble_count",
            "right_scrobble_count",
            "left_track_count",
            "right_track_count",
            "shared_track_count",
            "track_containment",
            "shared_track_scrobble_share_left",
            "shared_track_scrobble_share_right",
            "shared_album_count",
            "album_containment",
            "artist_mbid_relation",
        )
    )
    catalogue_only_decisions
    return (catalogue_only_decisions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Artist clusters
    """)
    return


@app.cell
def _(
    aliases,
    cluster_artist_aliases,
    decisions,
    events,
    paths,
    pl,
    resolution_config,
):
    existing_aliases = (
        pl.read_parquet(paths.curated_artist_aliases)
        if paths.curated_artist_aliases.is_file()
        else None
    )
    clusters, blocked_merges = cluster_artist_aliases(
        events,
        aliases.select("observed_artist_name"),
        decisions,
        config=resolution_config,
        existing_aliases=existing_aliases,
    )
    return blocked_merges, clusters, existing_aliases


@app.cell
def _(clusters, mo, pl):
    multi_alias_clusters = (
        clusters.group_by("cluster_key")
        .agg(
            pl.col("observed_artist_name").sort().alias("artist_names"),
            pl.len().alias("alias_count"),
            pl.col("existing_artist_id").drop_nulls().first().alias("artist_id"),
        )
        .filter(pl.col("alias_count") > 1)
        .sort(["alias_count", "cluster_key"], descending=[True, False])
    )
    mo.ui.table(
        multi_alias_clusters,
        pagination=False,
        selection=None,
        max_height=500,
    )
    return (multi_alias_clusters,)


@app.cell
def _(blocked_merges):
    blocked_merges
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Build artist tables
    """)
    return


@app.cell
def _(mo):
    build_artist_tables = mo.ui.run_button(label="Build artist tables")
    build_artist_tables
    return (build_artist_tables,)


@app.cell
def _(build_artist_tables, build_curated_artist_tables, mo, paths):
    mo.stop(not build_artist_tables.value)

    _build = build_curated_artist_tables(paths=paths)
    mo.md(f"""
    **Artists:** {_build.artists.height:,}  \n    **Aliases:** {_build.artist_aliases.height:,}
    """)
    return


if __name__ == "__main__":
    app.run()

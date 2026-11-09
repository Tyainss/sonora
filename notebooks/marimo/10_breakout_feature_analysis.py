import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import calendar
    import datetime as dt
    import json

    import marimo as mo
    import polars as pl

    from sonora.breakout import score_recent_listening_share
    from sonora.breakout.evaluation import (
        build_holdout_boundaries,
        last_evaluable_scoring_date,
    )
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return (
        build_holdout_boundaries,
        calendar,
        dt,
        json,
        last_evaluable_scoring_date,
        mo,
        paths,
        pl,
        score_recent_listening_share,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breakout feature analysis

    A **breakout** is the first time an artist becomes an important, repeated part of my listening. Looking at an artist's share of all my listens over the previous 7 days often detects this change only after it has started. I want to find clues that appear earlier.

    ## How to read these comparisons

    On each date, I compare an artist that later broke out with **other artists I played the same number of times in the previous 7 days**, on that exact date. The other artists did not break out in the following 60 days. Their **median** (middle) value is the reference.

    For example, suppose an artist I later played much more had **3 recent listens** and **12 earlier listens**. If the other artists also had 3 recent listens but typically had **30 earlier listens**, the result is **"Fewer previous listens" (12 versus 30)**.

    **"More", "fewer", "earlier", "longer" and "shorter" always mean relative to those other artists, not fixed thresholds.** "Same" means the numerical values are equal. For tracks, it means the *number* of different songs is equal, not the songs themselves.

    In the tables, **Artist-days** counts daily comparisons (so one artist can appear many times). In the feature tables, **Breakout artists** counts each artist once, based on the median of its daily differences; an artist can have both higher and lower values on different dates. The first song-count table instead shows how many artists appear in each recent-play group. The main comparisons include days with no recent listens. Those days are examined separately below.

    These results describe patterns among artists that later broke out. **The percentages are not breakout probabilities**, and model evaluation is still needed to know whether a feature helps predictions.
    """)
    return


@app.cell
def _(
    build_holdout_boundaries,
    calendar,
    dt,
    json,
    last_evaluable_scoring_date,
    paths,
    pl,
):
    targets = pl.read_parquet(paths.daily_breakout_targets)
    breakout_events = pl.read_parquet(paths.breakout_events)
    listening_events = pl.read_parquet(paths.curated_listening_events)
    metadata = json.loads(
        paths.breakout_build_metadata.read_text(encoding="utf-8")
    )

    built_users = [user for user in metadata["users"] if user["status"] == "built"]
    if len(built_users) != 1:
        raise ValueError("This analysis notebook expects one built user")

    user_metadata = built_users[0]
    user_id = user_metadata["user_id"]
    observation_coverage_end = dt.date.fromisoformat(
        user_metadata["last_complete_date"]
    ) + dt.timedelta(days=1)
    holdout_event_end = last_evaluable_scoring_date(observation_coverage_end)

    _month_index = holdout_event_end.year * 12 + holdout_event_end.month - 1 - 6
    _year, _month_zero_based = divmod(_month_index, 12)
    _month = _month_zero_based + 1
    _day = min(holdout_event_end.day, calendar.monthrange(_year, _month)[1])
    holdout_event_start = dt.date(_year, _month, _day) + dt.timedelta(days=1)

    holdout = build_holdout_boundaries(
        holdout_event_start,
        generation="current",
    )
    if holdout.event_end != holdout_event_end:
        raise ValueError("Derived holdout does not align with observation coverage")

    user_targets = targets.filter(
        (pl.col("user_id") == user_id)
        & (pl.col("scoring_date") <= holdout.development_end)
        & pl.col("has_full_horizon")
    )
    user_breakouts = breakout_events.filter(pl.col("user_id") == user_id)
    user_listening = listening_events.filter(pl.col("user_id") == user_id)
    return holdout, user_breakouts, user_listening, user_targets


@app.cell(hide_code=True)
def _(holdout, mo):
    mo.md(f"""
    The most recent data is reserved for testing. These comparisons use dates up to **{holdout.development_end}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does listening to more different songs matter?

    I count **distinct songs played in the previous 7 days**, rather than total plays. For example, 4 plays of 2 songs means 2 distinct tracks. I compare that number with other artists that also had 4 plays that week.

    A tie is unavoidable when there were **0 plays** (both have 0 songs) or **1 play** (both have 1 song). The group with **at least 2 plays** is more informative.
    """)
    return


@app.cell
def _(pl, score_recent_listening_share, user_listening, user_targets):
    recent_scores = score_recent_listening_share(user_listening, user_targets)

    candidate_days = user_targets.select(
        "user_id",
        "artist_id",
        pl.col("scoring_date").alias("date"),
    )
    track_daily = (
        user_listening.select("user_id", "artist_id", "track_id", "listened_at")
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "artist_id", "date")
        .agg(pl.col("track_id").unique().alias("track_ids"))
    )
    track_timeline = (
        pl.concat(
            [
                track_daily,
                candidate_days.with_columns(
                    pl.lit([], dtype=pl.List(pl.String)).alias("track_ids")
                ),
            ]
        )
        .group_by("user_id", "artist_id", "date")
        .agg(pl.col("track_ids").explode(empty_as_null=True).drop_nulls().unique().alias("track_ids"))
        .sort("user_id", "artist_id", "date")
    )
    distinct_tracks_7d = (
        track_timeline.rolling(
            index_column="date",
            period="7d",
            closed="left",
            group_by=["user_id", "artist_id"],
        )
        .agg(
            pl.col("track_ids")
            .explode(empty_as_null=True)
            .drop_nulls()
            .n_unique()
            .alias("distinct_tracks_7d")
        )
        .join(candidate_days, on=["user_id", "artist_id", "date"], how="semi")
        .rename({"date": "scoring_date"})
    )

    recent_evidence = recent_scores.join(
        distinct_tracks_7d,
        on=["user_id", "artist_id", "scoring_date"],
        validate="1:1",
    )
    return recent_evidence, recent_scores


@app.cell
def _(pl, recent_evidence, user_breakouts, user_targets):
    evidence = (
        user_targets.select(
            "user_id",
            "artist_id",
            "scoring_date",
            "target_60d",
        )
        .join(
            recent_evidence.select(
                "user_id",
                "artist_id",
                "scoring_date",
                "artist_scrobbles_7d",
                "listening_share_7d",
                "distinct_tracks_7d",
                "prior_scrobbles",
            ),
            on=["user_id", "artist_id", "scoring_date"],
            validate="1:1",
        )
        .join(
            user_breakouts.select(
                "user_id",
                "artist_id",
                "breakout_period_start",
            ),
            on=["user_id", "artist_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("target_60d")
                & (pl.col("scoring_date") < pl.col("breakout_period_start"))
            ).fill_null(False).alias("pre_breakout")
        )
    )

    positives = evidence.filter(pl.col("pre_breakout"))
    negatives = evidence.filter(~pl.col("target_60d"))

    comparable = positives.join(
        negatives.group_by("user_id", "scoring_date", "artist_scrobbles_7d").agg(
            pl.col("distinct_tracks_7d").median().alias("negative_median_tracks"),
            pl.len().alias("matched_negatives"),
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (pl.col("distinct_tracks_7d") - pl.col("negative_median_tracks")).alias(
            "track_difference"
        ),
        pl.when(pl.col("distinct_tracks_7d") > pl.col("negative_median_tracks"))
        .then(pl.lit("More tracks"))
        .when(pl.col("distinct_tracks_7d") < pl.col("negative_median_tracks"))
        .then(pl.lit("Fewer tracks"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return comparable, evidence


@app.cell
def _(comparable, mo, pl):
    track_listen_groups = (
        comparable.with_columns(
            pl.when(pl.col("artist_scrobbles_7d") == 0)
            .then(pl.lit("0 plays"))
            .when(pl.col("artist_scrobbles_7d") == 1)
            .then(pl.lit("1 play"))
            .otherwise(pl.lit("2+ plays"))
            .alias("Recent plays")
        )
        .group_by("Recent plays")
        .agg(
            pl.len().alias("Artist-days"),
            pl.struct("user_id", "artist_id")
            .n_unique()
            .alias("Artists in group"),
            (pl.col("comparison").eq("More tracks").mean() * 100)
            .round(1)
            .alias("More songs (%)"),
            (pl.col("comparison").eq("Same").mean() * 100)
            .round(1)
            .alias("Same count (%)"),
            (pl.col("comparison").eq("Fewer tracks").mean() * 100)
            .round(1)
            .alias("Fewer songs (%)"),
        )
        .sort("Recent plays")
    )
    mo.ui.table(track_listen_groups, selection=None)
    return (track_listen_groups,)


@app.cell(hide_code=True)
def _(mo, pl, track_listen_groups):
    _many = track_listen_groups.filter(pl.col("Recent plays") == "2+ plays")
    _days = _many["Artist-days"][0] if _many.height else 0
    _same = _many["Same count (%)"][0] if _many.height else 0.0
    _more = _many["More songs (%)"][0] if _many.height else 0.0
    _fewer = _many["Fewer songs (%)"][0] if _many.height else 0.0

    mo.md(f"""
    With **2 or more recent plays**, the number of distinct songs was the same in **{_same:.0f}%** of **{_days:,}** artist-days. Future breakout artists had more songs in **{_more:.0f}%** and fewer in **{_fewer:.0f}%**.

    **Conclusion:** the ties with 0 or 1 play are unavoidable. With 2+ plays, artists that later broke out were about as likely to have more songs as fewer. Track variety is not a priority to test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does previous listening matter?

    I count every play for the artist **before the most recent 7 days**. For example, 12 previous listens versus a typical 30 means **"Fewer previous listens"**. This measures how familiar the artist already was.
    """)
    return


@app.cell
def _(evidence, pl):
    familiarity_evidence = evidence.with_columns(
        (
            pl.col("prior_scrobbles").cast(pl.Int64)
            - pl.col("artist_scrobbles_7d").cast(pl.Int64)
        ).alias("older_scrobbles")
    )

    familiarity_positives = familiarity_evidence.filter(pl.col("pre_breakout"))
    familiarity_negatives = familiarity_evidence.filter(~pl.col("target_60d"))

    familiarity_comparable = familiarity_positives.join(
        familiarity_negatives.group_by(
            "user_id",
            "scoring_date",
            "artist_scrobbles_7d",
        ).agg(
            pl.col("older_scrobbles")
            .median()
            .alias("negative_median_older_scrobbles"),
            pl.len().alias("matched_negatives"),
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("older_scrobbles")
            - pl.col("negative_median_older_scrobbles")
        ).alias("older_scrobble_difference"),
        pl.when(
            pl.col("older_scrobbles")
            > pl.col("negative_median_older_scrobbles")
        )
        .then(pl.lit("More previous listens"))
        .when(
            pl.col("older_scrobbles")
            < pl.col("negative_median_older_scrobbles")
        )
        .then(pl.lit("Fewer previous listens"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return familiarity_comparable, familiarity_evidence


@app.cell
def _(familiarity_comparable, mo, pl):
    familiarity_row_summary = (
        familiarity_comparable.group_by("comparison")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
        .sort("comparison")
        .rename(
            {
                "comparison": "Previous listens",
                "len": "Cases",
                "share": "Share",
            }
        )
    )

    familiarity_event_summary = (
        familiarity_comparable.group_by("user_id", "artist_id")
        .agg(
            pl.col("older_scrobble_difference")
            .median()
            .alias("median_difference")
        )
        .with_columns(
            pl.when(pl.col("median_difference") > 0)
            .then(pl.lit("More previous listens"))
            .when(pl.col("median_difference") < 0)
            .then(pl.lit("Fewer previous listens"))
            .otherwise(pl.lit("Same"))
            .alias("comparison")
        )
        .group_by("comparison")
        .len()
        .sort("comparison")
        .rename(
            {
                "comparison": "Previous listens",
                "len": "Breakout artists",
            }
        )
    )

    _comparison_table = (
        familiarity_row_summary.join(
            familiarity_event_summary, on="Previous listens", how="full", coalesce=True
        )
        .with_columns(
            pl.col("Breakout artists").fill_null(0),
            pl.col("Cases").fill_null(0),
            pl.col("Share").fill_null(0.0),
        )
        .with_columns((pl.col("Share") * 100).round(1))
        .rename({"Cases": "Artist-days", "Share": "Artist-days (%)"})
        .sort("Previous listens")
    )
    mo.ui.table(_comparison_table, selection=None)
    return familiarity_event_summary, familiarity_row_summary


@app.cell(hide_code=True)
def _(
    familiarity_comparable,
    familiarity_event_summary,
    familiarity_row_summary,
    mo,
    pl,
):
    _rows = familiarity_comparable.height
    _artists = familiarity_comparable.select("user_id", "artist_id").n_unique()

    def _row_share(label):
        _match = familiarity_row_summary.filter(pl.col("Previous listens") == label)
        return _match["Share"][0] if _match.height else 0.0

    def _event_count(label):
        _match = familiarity_event_summary.filter(pl.col("Previous listens") == label)
        return _match["Breakout artists"][0] if _match.height else 0

    mo.md(f"""
    Future breakout artists had **more plays before the recent week** than other artists in **{_row_share('More previous listens'):.0%}** of daily comparisons. Across the whole period, **{_event_count('More previous listens')} of {_artists}** breakout artists typically had more previous listens.

    **Conclusion:** how often I listened to an artist in the past looks worth testing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does it matter when I first heard an artist?

    I count the days since I **first listened** to each artist. If it was 300 days ago, compared with a typical 150 days for the other artists, the label is **"First heard earlier"**.
    """)
    return


@app.cell
def _(familiarity_evidence, pl, user_listening):
    first_listens = (
        user_listening.select("user_id", "artist_id", "listened_at")
        .with_columns(pl.col("listened_at").dt.date().alias("first_listen_date"))
        .group_by("user_id", "artist_id")
        .agg(pl.col("first_listen_date").min())
    )

    relationship_evidence = (
        familiarity_evidence.join(
            first_listens,
            on=["user_id", "artist_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("scoring_date") - pl.col("first_listen_date"))
            .dt.total_days()
            .alias("days_since_first_listen")
        )
    )

    relationship_positives = relationship_evidence.filter(pl.col("pre_breakout"))
    relationship_negatives = relationship_evidence.filter(~pl.col("target_60d"))

    relationship_comparable = relationship_positives.join(
        relationship_negatives.group_by(
            "user_id",
            "scoring_date",
            "artist_scrobbles_7d",
        ).agg(
            pl.col("days_since_first_listen")
            .median()
            .alias("negative_median_days_known")
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("days_since_first_listen")
            - pl.col("negative_median_days_known")
        ).alias("days_known_difference"),
        pl.when(pl.col("days_since_first_listen") > pl.col("negative_median_days_known"))
        .then(pl.lit("First heard earlier"))
        .when(pl.col("days_since_first_listen") < pl.col("negative_median_days_known"))
        .then(pl.lit("First heard more recently"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return (relationship_comparable,)


@app.cell
def _(mo, pl, relationship_comparable):
    relationship_row_summary = (
        relationship_comparable.group_by("comparison")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
        .sort("comparison")
        .rename(
            {
                "comparison": "Time since first listen",
                "len": "Cases",
                "share": "Share",
            }
        )
    )

    relationship_event_summary = (
        relationship_comparable.group_by("user_id", "artist_id")
        .agg(pl.col("days_known_difference").median().alias("median_difference"))
        .with_columns(
            pl.when(pl.col("median_difference") > 0)
            .then(pl.lit("First heard earlier"))
            .when(pl.col("median_difference") < 0)
            .then(pl.lit("First heard more recently"))
            .otherwise(pl.lit("Same"))
            .alias("comparison")
        )
        .group_by("comparison")
        .len()
        .sort("comparison")
        .rename(
            {
                "comparison": "Time since first listen",
                "len": "Breakout artists",
            }
        )
    )

    _comparison_table = (
        relationship_row_summary.join(
            relationship_event_summary, on="Time since first listen", how="full", coalesce=True
        )
        .with_columns(
            pl.col("Breakout artists").fill_null(0),
            pl.col("Cases").fill_null(0),
            pl.col("Share").fill_null(0.0),
        )
        .with_columns((pl.col("Share") * 100).round(1))
        .rename({"Cases": "Artist-days", "Share": "Artist-days (%)"})
        .sort("Time since first listen")
    )
    mo.ui.table(_comparison_table, selection=None)
    return (relationship_row_summary,)


@app.cell(hide_code=True)
def _(mo, pl, relationship_comparable, relationship_row_summary):
    def _row_share(label):
        _match = relationship_row_summary.filter(pl.col("Time since first listen") == label)
        return _match["Share"][0] if _match.height else 0.0

    _active = relationship_comparable.filter(pl.col("artist_scrobbles_7d") > 0)
    _earlier = _active.filter(
        pl.col("comparison") == "First heard earlier"
    ).height / _active.height

    mo.md(f"""
    Future breakout artists had an **earlier first-listen date** in **{_row_share('First heard earlier'):.0%}** of daily comparisons. On days when I played the artist recently, that fell to **{_earlier:.0%}**.

    **Conclusion:** discovering an artist longer ago does not, by itself, look like a useful clue.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does a comeback after a long gap matter?

    I look at the most recent listen **before** the last 7-day period begins and count the days from that listen to the start of the period. For example, an 8-day gap compared with a typical 20-day gap is **"Shorter gap"**. This is not the time since the artist's very last listen, which could have happened yesterday.
    """)
    return


@app.cell
def _(familiarity_evidence, pl, user_listening):
    older_cutoffs = (
        familiarity_evidence.select(
            "user_id",
            "artist_id",
            "scoring_date",
            "target_60d",
            "pre_breakout",
            "artist_scrobbles_7d",
            "older_scrobbles",
        )
        .filter(pl.col("older_scrobbles") > 0)
        .with_columns(
            (pl.col("scoring_date") - pl.duration(days=7)).alias("older_cutoff")
        )
        .sort("user_id", "artist_id", "older_cutoff")
    )

    listening_dates = (
        user_listening.select("user_id", "artist_id", "listened_at")
        .with_columns(pl.col("listened_at").dt.date().alias("listen_date"))
        .select("user_id", "artist_id", "listen_date")
        .unique()
        .sort("user_id", "artist_id", "listen_date")
    )

    gap_evidence = (
        older_cutoffs.join_asof(
            listening_dates,
            left_on="older_cutoff",
            right_on="listen_date",
            by=["user_id", "artist_id"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("older_cutoff") - pl.col("listen_date"))
            .dt.total_days()
            .alias("days_since_older_listen")
        )
    )

    gap_positives = gap_evidence.filter(pl.col("pre_breakout"))
    gap_negatives = gap_evidence.filter(~pl.col("target_60d"))

    gap_comparable = gap_positives.join(
        gap_negatives.group_by(
            "user_id",
            "scoring_date",
            "artist_scrobbles_7d",
        ).agg(
            pl.col("days_since_older_listen")
            .median()
            .alias("negative_median_gap_days")
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("days_since_older_listen")
            - pl.col("negative_median_gap_days")
        ).alias("gap_difference"),
        pl.when(pl.col("days_since_older_listen") > pl.col("negative_median_gap_days"))
        .then(pl.lit("Longer gap"))
        .when(pl.col("days_since_older_listen") < pl.col("negative_median_gap_days"))
        .then(pl.lit("Shorter gap"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return gap_comparable, listening_dates


@app.cell
def _(gap_comparable, mo, pl):
    gap_row_summary = (
        gap_comparable.group_by("comparison")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
        .sort("comparison")
        .rename(
            {
                "comparison": "Gap before current period",
                "len": "Cases",
                "share": "Share",
            }
        )
    )

    gap_event_summary = (
        gap_comparable.group_by("user_id", "artist_id")
        .agg(pl.col("gap_difference").median().alias("median_difference"))
        .with_columns(
            pl.when(pl.col("median_difference") > 0)
            .then(pl.lit("Longer gap"))
            .when(pl.col("median_difference") < 0)
            .then(pl.lit("Shorter gap"))
            .otherwise(pl.lit("Same"))
            .alias("comparison")
        )
        .group_by("comparison")
        .len()
        .sort("comparison")
        .rename(
            {
                "comparison": "Gap before current period",
                "len": "Breakout artists",
            }
        )
    )

    _comparison_table = (
        gap_row_summary.join(
            gap_event_summary, on="Gap before current period", how="full", coalesce=True
        )
        .with_columns(
            pl.col("Breakout artists").fill_null(0),
            pl.col("Cases").fill_null(0),
            pl.col("Share").fill_null(0.0),
        )
        .with_columns((pl.col("Share") * 100).round(1))
        .rename({"Cases": "Artist-days", "Share": "Artist-days (%)"})
        .sort("Gap before current period")
    )
    mo.ui.table(_comparison_table, selection=None)
    return gap_event_summary, gap_row_summary


@app.cell(hide_code=True)
def _(gap_event_summary, gap_row_summary, mo, pl):
    def _row_share(label):
        _match = gap_row_summary.filter(pl.col("Gap before current period") == label)
        return _match["Share"][0] if _match.height else 0.0

    def _event_count(label):
        _match = gap_event_summary.filter(pl.col("Gap before current period") == label)
        return _match["Breakout artists"][0] if _match.height else 0

    _artists = gap_event_summary["Breakout artists"].sum()

    mo.md(f"""
    Future breakout artists had **shorter gaps before the recent week** in **{_row_share('Shorter gap'):.0%}** of daily comparisons. Across the period, **{_event_count('Shorter gap')} of {_artists}** artists typically had shorter gaps.

    **Conclusion:** a shorter gap may be useful, although it could reflect the fact that I had listened to those artists more often. It is worth testing alongside previous listening count.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does listening across more past days matter?

    I count the number of **different calendar days** when I listened to an artist before the most recent week. An artist played 20 times across 3 days has 3 previous listening days, even though they had 20 plays.
    """)
    return


@app.cell
def _(familiarity_evidence, listening_dates, pl):
    active_day_timeline = (
        listening_dates.sort("user_id", "artist_id", "listen_date")
        .with_columns(
            pl.col("listen_date")
            .cum_count()
            .over(["user_id", "artist_id"])
            .alias("cumulative_active_days")
        )
    )

    active_day_cutoffs = (
        familiarity_evidence.select(
            "user_id",
            "artist_id",
            "scoring_date",
            "target_60d",
            "pre_breakout",
            "artist_scrobbles_7d",
        )
        .with_columns(
            (pl.col("scoring_date") - pl.duration(days=7)).alias("older_cutoff")
        )
        .sort("user_id", "artist_id", "older_cutoff")
    )

    active_day_evidence = (
        active_day_cutoffs.join_asof(
            active_day_timeline,
            left_on="older_cutoff",
            right_on="listen_date",
            by=["user_id", "artist_id"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            pl.col("cumulative_active_days")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("older_active_days")
        )
    )

    active_day_positives = active_day_evidence.filter(pl.col("pre_breakout"))
    active_day_negatives = active_day_evidence.filter(~pl.col("target_60d"))

    active_day_comparable = active_day_positives.join(
        active_day_negatives.group_by(
            "user_id",
            "scoring_date",
            "artist_scrobbles_7d",
        ).agg(
            pl.col("older_active_days")
            .median()
            .alias("negative_median_older_active_days")
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("older_active_days")
            - pl.col("negative_median_older_active_days")
        ).alias("active_day_difference"),
        pl.when(
            pl.col("older_active_days")
            > pl.col("negative_median_older_active_days")
        )
        .then(pl.lit("More past listening days"))
        .when(
            pl.col("older_active_days")
            < pl.col("negative_median_older_active_days")
        )
        .then(pl.lit("Fewer past listening days"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return (active_day_comparable,)


@app.cell
def _(active_day_comparable, mo, pl):
    active_day_row_summary = (
        active_day_comparable.group_by("comparison")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
        .sort("comparison")
        .rename(
            {
                "comparison": "Past listening days",
                "len": "Cases",
                "share": "Share",
            }
        )
    )

    active_day_event_summary = (
        active_day_comparable.group_by("user_id", "artist_id")
        .agg(pl.col("active_day_difference").median().alias("median_difference"))
        .with_columns(
            pl.when(pl.col("median_difference") > 0)
            .then(pl.lit("More past listening days"))
            .when(pl.col("median_difference") < 0)
            .then(pl.lit("Fewer past listening days"))
            .otherwise(pl.lit("Same"))
            .alias("comparison")
        )
        .group_by("comparison")
        .len()
        .sort("comparison")
        .rename(
            {
                "comparison": "Past listening days",
                "len": "Breakout artists",
            }
        )
    )

    _comparison_table = (
        active_day_row_summary.join(
            active_day_event_summary, on="Past listening days", how="full", coalesce=True
        )
        .with_columns(
            pl.col("Breakout artists").fill_null(0),
            pl.col("Cases").fill_null(0),
            pl.col("Share").fill_null(0.0),
        )
        .with_columns((pl.col("Share") * 100).round(1))
        .rename({"Cases": "Artist-days", "Share": "Artist-days (%)"})
        .sort("Past listening days")
    )
    mo.ui.table(_comparison_table, selection=None)
    return active_day_event_summary, active_day_row_summary


@app.cell(hide_code=True)
def _(active_day_event_summary, active_day_row_summary, mo, pl):
    _total = active_day_event_summary["Breakout artists"].sum()
    _more = active_day_event_summary.filter(
        pl.col("Past listening days") == "More past listening days"
    )["Breakout artists"][0]
    _fewer = active_day_event_summary.filter(
        pl.col("Past listening days") == "Fewer past listening days"
    )["Breakout artists"][0]
    _more_share = active_day_row_summary.filter(
        pl.col("Past listening days") == "More past listening days"
    )["Share"][0]
    _fewer_share = active_day_row_summary.filter(
        pl.col("Past listening days") == "Fewer past listening days"
    )["Share"][0]

    mo.md(f"""
    Future breakout artists had **more separate listening days** in **{_more_share:.0%}** of comparisons and fewer in **{_fewer_share:.0%}**. Across {_total} artists, {_more} typically had more days and {_fewer} had fewer.

    **Conclusion:** there is no clear difference here, so separate listening days are not a priority.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does an artist's strongest earlier week matter?

    I find the highest share of all my listening that an artist received during any earlier 7-day period. For example, **5 artist plays out of 10 total** (50%) is a higher peak than **30 out of 100** (30%). A week with very few total plays can therefore produce a high share.

    Only past weeks ending **before the current 7-day period starts** count, using the full available listening history.
    """)
    return


@app.cell
def _(familiarity_evidence, pl, recent_scores, user_listening):
    days_after_listen = pl.DataFrame({"days_after_listen": list(range(1, 8))})

    daily_artist = (
        user_listening.select("user_id", "artist_id", "listened_at")
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "artist_id", "date")
        .agg(pl.len().alias("artist_plays"))
    )
    daily_user = daily_artist.group_by("user_id", "date").agg(
        pl.col("artist_plays").sum().alias("user_plays")
    )

    artist_weeks = (
        daily_artist.join(days_after_listen, how="cross")
        .with_columns(
            (pl.col("date") + pl.duration(days=pl.col("days_after_listen")))
            .cast(pl.Date)
            .alias("history_date")
        )
        .group_by("user_id", "artist_id", "history_date")
        .agg(pl.col("artist_plays").sum().alias("artist_plays_7d"))
    )
    user_weeks = (
        daily_user.join(days_after_listen, how="cross")
        .with_columns(
            (pl.col("date") + pl.duration(days=pl.col("days_after_listen")))
            .cast(pl.Date)
            .alias("history_date")
        )
        .group_by("user_id", "history_date")
        .agg(pl.col("user_plays").sum().alias("user_plays_7d"))
    )

    full_peak_timeline = (
        artist_weeks.join(
            user_weeks,
            on=["user_id", "history_date"],
            validate="m:1",
        )
        .with_columns(
            (pl.col("artist_plays_7d") / pl.col("user_plays_7d"))
            .alias("week_share")
        )
        .sort("user_id", "artist_id", "history_date")
        .with_columns(
            pl.col("week_share")
            .cum_max()
            .over(["user_id", "artist_id"])
            .alias("running_peak_share_7d")
        )
        .select("user_id", "artist_id", "history_date", "running_peak_share_7d")
    )
    scored_peak_timeline = (
        recent_scores.select(
            "user_id", "artist_id", "scoring_date", "listening_share_7d"
        )
        .sort("user_id", "artist_id", "scoring_date")
        .with_columns(
            pl.col("listening_share_7d")
            .cum_max()
            .over(["user_id", "artist_id"])
            .alias("running_peak_share_7d")
        )
        .rename({"scoring_date": "history_date"})
    )

    peak_cutoffs = (
        familiarity_evidence.select(
            "user_id", "artist_id", "scoring_date", "target_60d",
            "pre_breakout", "artist_scrobbles_7d", "older_scrobbles",
        )
        .with_columns(
            (pl.col("scoring_date") - pl.duration(days=7)).alias("history_cutoff")
        )
        .sort("user_id", "artist_id", "history_cutoff")
    )

    peak_evidence = (
        peak_cutoffs.join_asof(
            full_peak_timeline,
            left_on="history_cutoff",
            right_on="history_date",
            by=["user_id", "artist_id"],
            strategy="backward",
            allow_exact_matches=True,
            check_sortedness=False,
        )
        .with_columns(
            pl.col("running_peak_share_7d")
            .fill_null(0.0)
            .alias("prior_peak_share_7d")
        )
        .drop("running_peak_share_7d", "history_date")
    )
    scored_peak_evidence = (
        peak_cutoffs.join_asof(
            scored_peak_timeline,
            left_on="history_cutoff",
            right_on="history_date",
            by=["user_id", "artist_id"],
            strategy="backward",
            allow_exact_matches=True,
            check_sortedness=False,
        )
        .select(
            "user_id", "artist_id", "scoring_date",
            pl.col("running_peak_share_7d")
            .fill_null(0.0)
            .alias("scored_peak_share_7d"),
        )
    )
    peak_history_check = (
        peak_evidence.join(
            scored_peak_evidence,
            on=["user_id", "artist_id", "scoring_date"],
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("prior_peak_share_7d") - pl.col("scored_peak_share_7d")
            ).alias("history_difference")
        )
    )

    peak_positives = peak_evidence.filter(pl.col("pre_breakout"))
    peak_negatives = peak_evidence.filter(~pl.col("target_60d"))
    peak_comparable = peak_positives.join(
        peak_negatives.group_by(
            "user_id", "scoring_date", "artist_scrobbles_7d"
        ).agg(
            pl.col("prior_peak_share_7d")
            .median()
            .alias("negative_median_prior_peak_share")
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("prior_peak_share_7d")
            - pl.col("negative_median_prior_peak_share")
        ).alias("peak_difference"),
        pl.when(
            pl.col("prior_peak_share_7d")
            > pl.col("negative_median_prior_peak_share")
        )
        .then(pl.lit("Higher past peak"))
        .when(
            pl.col("prior_peak_share_7d")
            < pl.col("negative_median_prior_peak_share")
        )
        .then(pl.lit("Lower past peak"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return peak_comparable, peak_evidence, peak_history_check


@app.cell(hide_code=True)
def _(mo, peak_history_check, pl):
    _changed = peak_history_check.filter(
        pl.col("history_difference").abs() > 1e-10
    ).height
    mo.md(
        f"**{_changed:,} of {peak_history_check.height:,}** artist-days have "
        "a different past peak using the full available listening history, rather than only scoring dates."
    )
    return


@app.cell
def _(mo, peak_comparable, pl):
    peak_row_summary = (
        peak_comparable.group_by("comparison")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
        .sort("comparison")
        .rename(
            {
                "comparison": "Past peak",
                "len": "Cases",
                "share": "Share",
            }
        )
    )

    peak_event_summary = (
        peak_comparable.group_by("user_id", "artist_id")
        .agg(pl.col("peak_difference").median().alias("median_difference"))
        .with_columns(
            pl.when(pl.col("median_difference") > 0)
            .then(pl.lit("Higher past peak"))
            .when(pl.col("median_difference") < 0)
            .then(pl.lit("Lower past peak"))
            .otherwise(pl.lit("Same"))
            .alias("comparison")
        )
        .group_by("comparison")
        .len()
        .sort("comparison")
        .rename(
            {
                "comparison": "Past peak",
                "len": "Breakout artists",
            }
        )
    )

    _comparison_table = (
        peak_row_summary.join(
            peak_event_summary, on="Past peak", how="full", coalesce=True
        )
        .with_columns(
            pl.col("Breakout artists").fill_null(0),
            pl.col("Cases").fill_null(0),
            pl.col("Share").fill_null(0.0),
        )
        .with_columns((pl.col("Share") * 100).round(1))
        .rename({"Cases": "Artist-days", "Share": "Artist-days (%)"})
        .sort("Past peak")
    )
    mo.ui.table(_comparison_table, selection=None)
    return peak_event_summary, peak_row_summary


@app.cell(hide_code=True)
def _(mo, peak_event_summary, peak_row_summary, pl):
    _total = peak_event_summary["Breakout artists"].sum()
    _higher_events = peak_event_summary.filter(
        pl.col("Past peak") == "Higher past peak"
    )
    _higher_rows = peak_row_summary.filter(
        pl.col("Past peak") == "Higher past peak"
    )
    _higher = _higher_events["Breakout artists"][0] if _higher_events.height else 0
    _higher_share = _higher_rows["Share"][0] if _higher_rows.height else 0.0

    mo.md(f"""
    Future breakout artists had received a **larger share of my listening in a previous week** in **{_higher_share:.0%}** of daily comparisons, including **{_higher} of {_total}** artists overall.

    **Conclusion:** past listening peaks look interesting, but this could be because those artists already had more total previous plays.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do strong earlier weeks matter when previous listening is the same?

    This time, I compare artists with the **same number of recent plays and the same total number of previous plays**. For example, two artists might each have 3 recent and 40 previous plays, but one had a much stronger individual week earlier on.

    The question is whether a strong past week says anything beyond the **total** previous listening count. Exact matches are harder to find here, so the number of comparisons matters.
    """)
    return


@app.cell
def _(peak_evidence, pl):
    known_artists = peak_evidence.filter(pl.col("older_scrobbles") > 0)
    peak_history_positives = known_artists.filter(pl.col("pre_breakout"))
    peak_history_negatives = known_artists.filter(~pl.col("target_60d"))

    peak_history_comparable = peak_history_positives.join(
        peak_history_negatives.group_by(
            "user_id",
            "scoring_date",
            "artist_scrobbles_7d",
            "older_scrobbles",
        ).agg(
            pl.col("prior_peak_share_7d")
            .median()
            .alias("other_peak_median")
        ),
        on=["user_id", "scoring_date", "artist_scrobbles_7d", "older_scrobbles"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (
            pl.col("prior_peak_share_7d") - pl.col("other_peak_median")
        ).alias("peak_difference"),
        pl.when(pl.col("prior_peak_share_7d") > pl.col("other_peak_median"))
        .then(pl.lit("Higher past peak"))
        .when(pl.col("prior_peak_share_7d") < pl.col("other_peak_median"))
        .then(pl.lit("Lower past peak"))
        .otherwise(pl.lit("Same"))
        .alias("comparison"),
    )
    return peak_history_comparable, peak_history_positives


@app.cell
def _(mo, peak_history_comparable, peak_history_positives, pl):
    cases = (
        peak_history_comparable.group_by("comparison")
        .agg(pl.len().alias("Cases"))
        .with_columns(
            (pl.col("Cases") / pl.col("Cases").sum()).alias("Share")
        )
    )
    artists = (
        peak_history_comparable.group_by("user_id", "artist_id")
        .agg(pl.col("peak_difference").median().alias("difference"))
        .with_columns(
            pl.when(pl.col("difference") > 0)
            .then(pl.lit("Higher past peak"))
            .when(pl.col("difference") < 0)
            .then(pl.lit("Lower past peak"))
            .otherwise(pl.lit("Same"))
            .alias("comparison"),
        )
        .group_by("comparison")
        .len()
        .rename({"len": "Breakout artists"})
    )
    summary = (
        cases.join(artists, on="comparison", how="left")
        .fill_null(0)
        .rename({"comparison": "Past peak"})
        .sort("Past peak")
    )
    mo.vstack(
        [
            mo.md(
                f"**{peak_history_comparable.height:,} of "
                f"{peak_history_positives.height:,}** cases could be compared "
                "with the same recent and previous listens."
            ),
            mo.ui.table(summary, selection=None),
        ]
    )
    return (summary,)


@app.cell(hide_code=True)
def _(mo, peak_history_comparable, peak_history_positives, pl, summary):
    _higher = summary.filter(pl.col("Past peak") == "Higher past peak")
    _share = _higher["Share"][0] if _higher.height else 0.0
    _artists = _higher["Breakout artists"][0] if _higher.height else 0
    _total = summary["Breakout artists"].sum()
    _coverage = peak_history_comparable.height / peak_history_positives.height

    mo.md(f"""
    Even after matching the **same recent and previous play counts**, future breakout artists had higher past peaks in **{_share:.0%}** of the daily comparisons, across **{_artists} of {_total}** artists. An exact match was possible for only **{_coverage:.0%}** of the available comparisons.

    **Conclusion:** there is limited evidence that a strong past week adds much beyond the number of previous plays. The small number of closely matched artists makes this uncertain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do days without recent listening change the results?

    Many artists that later broke out had **no plays in the most recent 7 days** on the dates examined. I repeat the key comparisons on days with **at least one recent play**, so inactivity does not dominate the results.
    """)
    return


@app.cell
def _(
    evidence,
    familiarity_comparable,
    mo,
    peak_comparable,
    peak_history_comparable,
    pl,
):
    _before = evidence.filter(pl.col("pre_breakout"))
    _without = _before.filter(pl.col("artist_scrobbles_7d") == 0)
    _features = [
        ("Previous listens", familiarity_comparable, "More previous listens"),
        ("Past peak", peak_comparable, "Higher past peak"),
        ("Past peak, same previous listens", peak_history_comparable, "Higher past peak"),
    ]
    _rows = []
    for _name, _frame, _direction in _features:
        _active = _frame.filter(pl.col("artist_scrobbles_7d") > 0)
        _rows.append({
            "Feature": _name,
            "Compared result": _direction,
            "All days (%)": round(100 * _frame.filter(
                pl.col("comparison") == _direction
            ).height / _frame.height, 1) if _frame.height else None,
            "With recent listens (%)": round(100 * _active.filter(
                pl.col("comparison") == _direction
            ).height / _active.height, 1) if _active.height else None,
            "With recent listens (days)": _active.height,
            "Artists": _active.select("user_id", "artist_id").n_unique(),
        })
    recent_listening_check = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(
            f"**{_without.height:,} of {_before.height:,}** days before breakout "
            "had no listens for that artist in the previous week."
        ),
        mo.ui.table(recent_listening_check, selection=None),
    ])
    return (recent_listening_check,)


@app.cell(hide_code=True)
def _(mo, pl, recent_listening_check):
    _older = recent_listening_check.filter(pl.col("Feature") == "Previous listens")
    _peak = recent_listening_check.filter(pl.col("Feature") == "Past peak")
    _same = recent_listening_check.filter(
        pl.col("Feature") == "Past peak, same previous listens"
    )
    mo.md(f"""
    On dates when I listened to the artist recently, future breakout artists had more **previous plays** in **{_older['With recent listens (%)'][0]:.0f}%** of comparisons and a **higher past peak** in **{_peak['With recent listens (%)'][0]:.0f}%**.

    When previous play counts were also equal, the past peak was higher in only **{_same['With recent listens (%)'][0]:.0f}%** of **{_same['With recent listens (days)'][0]}** comparisons across **{_same['Artists'][0]}** artists.

    **Conclusion:** previous play count is the clearer feature candidate. A past peak might add something, but this comparison gives limited support for it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How early do these differences appear?

    I group the dates when an artist had recent plays by **how many days remained before breakout**. This shows whether a difference was visible weeks beforehand or only close to the breakout.

    Each percentage is the share of artist-days with the stated result, not a prediction success rate. The same artist can contribute several dates.
    """)
    return


@app.cell
def _(
    familiarity_comparable,
    mo,
    peak_comparable,
    peak_history_comparable,
    pl,
    user_breakouts,
):
    _starts = user_breakouts.select(
        "user_id", "artist_id", "breakout_period_start"
    )
    _features = [
        ("Previous listens", familiarity_comparable, "More previous listens"),
        ("Past peak", peak_comparable, "Higher past peak"),
        ("Past peak, same previous listens", peak_history_comparable, "Higher past peak"),
    ]
    _groups = []
    for _name, _frame, _direction in _features:
        _dated = (
            _frame.filter(pl.col("artist_scrobbles_7d") > 0)
            .join(_starts, on=["user_id", "artist_id"], validate="m:1")
            .with_columns(
                (pl.col("breakout_period_start") - pl.col("scoring_date"))
                .dt.total_days()
                .alias("days_before_breakout")
            )
            .filter(pl.col("days_before_breakout") > 0)
            .with_columns(
                pl.when(pl.col("days_before_breakout") <= 7)
                .then(pl.lit("1–7 days"))
                .when(pl.col("days_before_breakout") <= 30)
                .then(pl.lit("8–30 days"))
                .when(pl.col("days_before_breakout") <= 60)
                .then(pl.lit("31–60 days"))
                .otherwise(pl.lit("Over 60 days"))
                .alias("Before breakout")
            )
        )
        _grouped = _dated.group_by("Before breakout").agg(
            pl.len().alias("Days"),
            pl.struct("user_id", "artist_id").n_unique().alias("Artists"),
            (pl.col("comparison").eq(_direction).mean() * 100)
            .round(1)
            .alias("Matching result (%)"),
        ).with_columns(pl.lit(_name).alias("Feature"))
        _groups.append(_grouped)
    _order = {"1–7 days": 0, "8–30 days": 1, "31–60 days": 2, "Over 60 days": 3}
    early_comparison = (
        pl.concat(_groups)
        .with_columns(
            pl.col("Before breakout")
            .replace_strict(_order, return_dtype=pl.Int8)
            .alias("order")
        )
        .sort("Feature", "order")
        .drop("order")
        .select("Feature", "Before breakout", "Days", "Artists", "Matching result (%)")
    )
    mo.ui.table(early_comparison, selection=None)
    return (early_comparison,)


@app.cell(hide_code=True)
def _(early_comparison, mo, pl):
    _older = early_comparison.filter(pl.col("Feature") == "Previous listens")
    _early = _older.filter(pl.col("Before breakout") == "31–60 days")
    _late = _older.filter(pl.col("Before breakout") == "1–7 days")
    mo.md(f"""
    **31–60 days before breakout**, future breakout artists had more previous plays in **{_early['Matching result (%)'][0]:.0f}%** of the daily comparisons, compared with **{_late['Matching result (%)'][0]:.0f}%** during the final 7 days.

    **Conclusion:** previous listening may be useful well before breakout begins. These figures do not show whether a daily ranked prediction will improve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature overview

    This brings the main comparisons together. For each row, the percentage means **the share of artist-days when the stated result was true** compared with other artists. Only days with at least one recent play are included (except the distinct-song row, which requires two). Higher percentages do not automatically mean better predictions.
    """)
    return


@app.cell
def _(
    active_day_comparable,
    comparable,
    familiarity_comparable,
    gap_comparable,
    mo,
    peak_comparable,
    peak_history_comparable,
    pl,
    relationship_comparable,
):
    _features = [
        (
            "More different songs (2+ recent plays)",
            comparable.filter(pl.col("artist_scrobbles_7d") >= 2),
            "More tracks",
        ),
        ("More previous listens", familiarity_comparable, "More previous listens"),
        ("First heard earlier", relationship_comparable, "First heard earlier"),
        ("Shorter gap before recent week", gap_comparable, "Shorter gap"),
        ("More listening days", active_day_comparable, "More past listening days"),
        ("Higher past peak", peak_comparable, "Higher past peak"),
        ("Higher peak, same previous listens", peak_history_comparable, "Higher past peak"),
    ]
    _rows = []
    for _name, _frame, _direction in _features:
        _active = _frame.filter(pl.col("artist_scrobbles_7d") > 0)
        _rows.append({
            "Feature": _name,
            "Artist-days": _active.height,
            "Breakout artists": _active.select("user_id", "artist_id").n_unique(),
            "Days with this result (%)": round(
                100 * _active.filter(pl.col("comparison") == _direction).height
                / _active.height, 1
            ) if _active.height else None,
        })
    feature_overview = pl.DataFrame(_rows)
    mo.ui.table(feature_overview, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    **Conclusion:** previous listening count is the clearest feature to test. Past peaks and gaps between listens are possible additions, but may largely describe the same listening habits.

    Track variety also shows no clear difference once days with fewer than 2 plays are excluded. Time since first listen and the number of previous listening days are lower priorities.

    These comparisons identify **possible features**, but they do not establish whether breakout predictions would improve.
    """)
    return


if __name__ == "__main__":
    app.run()

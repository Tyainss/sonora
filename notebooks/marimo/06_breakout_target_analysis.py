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
    # Breakout target analysis

    We want to turn breakout into something we can label from listening history: the first move from background listening into strong, repeated importance.
    """)
    return


@app.cell
def _():
    import altair as alt
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return alt, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Artist trajectories

    Individual histories help us see the patterns the definition needs to handle: short bursts, gradual buildup, long gaps, and sustained adoption.
    """)
    return


@app.cell
def _(paths, pl):
    listening_events = pl.scan_parquet(paths.curated_listening_events)
    artists = pl.scan_parquet(paths.curated_artists)
    return artists, listening_events


@app.cell
def _(listening_events, pl):
    daily_user_listening = (
        listening_events
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "date")
        .agg(pl.len().alias("user_scrobbles"))
    )

    daily_artist_listening = (
        listening_events
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "artist_id", "date")
        .agg(pl.len().alias("artist_scrobbles"))
        .join(
            daily_user_listening,
            on=["user_id", "date"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("artist_scrobbles")
                / pl.col("user_scrobbles")
                * 100
            )
            .round(2)
            .alias("artist_share_pct")
        )
    )
    return daily_artist_listening, daily_user_listening


@app.cell
def _(daily_artist_listening, pl):
    artist_active_days = (
        daily_artist_listening
        .select("user_id", "artist_id", "date")
        .with_columns(
            pl.col("date")
            .shift(1)
            .over(["user_id", "artist_id"], order_by="date")
            .alias("previous_active_date")
        )
        .with_columns(
            (
                pl.col("date") - pl.col("previous_active_date")
            )
            .dt.total_days()
            .alias("gap_days")
        )
    )
    return (artist_active_days,)


@app.cell
def _(artist_active_days, artists, daily_artist_listening, pl):
    artist_trajectory_summary = (
        daily_artist_listening
        .group_by("user_id", "artist_id")
        .agg(
            pl.col("artist_scrobbles").sum().alias("scrobbles"),
            pl.len().alias("active_days"),
            pl.col("date").min().alias("first_seen"),
            pl.col("date").max().alias("last_seen"),
            pl.col("artist_scrobbles").max().alias("max_day_scrobbles"),
            pl.col("artist_share_pct").max().alias("max_daily_share_pct"),
        )
        .join(
            artist_active_days
            .group_by("user_id", "artist_id")
            .agg(pl.col("gap_days").max().alias("max_gap_days")),
            on=["user_id", "artist_id"],
            how="left",
            validate="1:1",
        )
        .join(artists, on="artist_id", how="left", validate="m:1")
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            "scrobbles",
            "active_days",
            "first_seen",
            "last_seen",
            "max_day_scrobbles",
            "max_daily_share_pct",
            "max_gap_days",
        )
        .sort("scrobbles", descending=True)
        .collect()
    )

    artist_trajectory_summary
    return (artist_trajectory_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Single days can be very concentrated on one artist, so daily peaks are too noisy for breakout on their own.
    """)
    return


@app.cell
def _(artist_trajectory_summary, mo):
    _name_counts = (
        artist_trajectory_summary
        .group_by("canonical_name")
        .len(name="name_count")
    )
    _artist_options = {}
    for _row in (
        artist_trajectory_summary
        .join(_name_counts, on="canonical_name", how="left", validate="m:1")
        .iter_rows(named=True)
    ):
        _label = _row["canonical_name"]
        if _row["name_count"] > 1:
            _label = f"{_label} [{_row['artist_id'][:8]}]"
        _artist_options[_label] = _row["artist_id"]

    artist_selector = mo.ui.dropdown(
        options=_artist_options,
        value=next(iter(_artist_options)),
        searchable=True,
        label="Artist",
        full_width=True,
    )
    artist_selector
    return (artist_selector,)


@app.cell
def _(artist_selector, artist_trajectory_summary, mo):
    _selected_summary = (
        artist_trajectory_summary
        .filter(artist_trajectory_summary["artist_id"] == artist_selector.value)
        .row(0, named=True)
    )

    _gap = _selected_summary["max_gap_days"]
    _gap_text = f"{_gap:,} days" if _gap is not None else "-"

    mo.md(f"""
    **{_selected_summary['canonical_name']}**  
    **Scrobbles:** {_selected_summary['scrobbles']:,} · **Active days:** {_selected_summary['active_days']:,} · **First seen:** {_selected_summary['first_seen']} · **Last seen:** {_selected_summary['last_seen']} · **Longest gap:** {_gap_text}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weekly view is more useful because it shows both how much the artist was played and whether that listening was spread across several days.
    """)
    return


@app.cell
def _(artist_selector, daily_artist_listening, daily_user_listening, pl):
    _weekly_user_listening = (
        daily_user_listening
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("user_id", "week")
        .agg(pl.col("user_scrobbles").sum().alias("user_scrobbles"))
    )

    selected_artist_history = (
        daily_artist_listening
        .filter(pl.col("artist_id") == artist_selector.value)
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("user_id", "artist_id", "week")
        .agg(
            pl.col("artist_scrobbles").sum().alias("artist_scrobbles"),
            pl.len().alias("active_days"),
        )
        .join(
            _weekly_user_listening,
            on=["user_id", "week"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("artist_scrobbles")
                / pl.col("user_scrobbles")
                * 100
            )
            .round(2)
            .alias("artist_share_pct")
        )
        .sort("week")
        .collect()
    )
    return (selected_artist_history,)


@app.cell
def _(alt, artist_selector, selected_artist_history):
    _base = alt.Chart(selected_artist_history).encode(
        x=alt.X("week:T", title="Week"),
        tooltip=[
            alt.Tooltip("week:T", title="Week"),
            alt.Tooltip("artist_scrobbles:Q", title="Artist scrobbles"),
            alt.Tooltip("active_days:Q", title="Active days"),
            alt.Tooltip("user_scrobbles:Q", title="All scrobbles"),
            alt.Tooltip(
                "artist_share_pct:Q",
                title="Listening share (%)",
                format=".2f",
            ),
        ],
    )

    _scrobbles_chart = (
        _base
        .mark_bar()
        .encode(
            y=alt.Y("artist_scrobbles:Q", title="Scrobbles"),
        )
        .properties(height=180)
    )

    _share_chart = (
        _base
        .mark_circle()
        .encode(
            y=alt.Y(
                "artist_share_pct:Q",
                title="Weekly listening share (%)",
                scale=alt.Scale(zero=True),
            ),
            size=alt.Size(
                "active_days:Q",
                title="Active days",
                scale=alt.Scale(range=[20, 180]),
            ),
        )
        .properties(height=180)
    )

    alt.vconcat(
        _scrobbles_chart,
        _share_chart,
        spacing=10,
        title=artist_selector.selected_key,
    ).interactive(bind_y=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Importance and repetition

    For each artist-week, we compare how much of the week's listening went to the artist with how many different days they were played.

    `typical_share_pct` is the middle of each group. `top_10_share_pct` shows a stronger level reached by only about 10% of those weeks.
    """)
    return


@app.cell
def _(artists, daily_artist_listening, daily_user_listening, pl):
    weekly_user_listening = (
        daily_user_listening
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("user_id", "week")
        .agg(pl.col("user_scrobbles").sum().alias("user_scrobbles"))
    )

    weekly_artist_listening = (
        daily_artist_listening
        .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("user_id", "artist_id", "week")
        .agg(
            pl.col("artist_scrobbles").sum().alias("artist_scrobbles"),
            pl.len().alias("active_days"),
        )
        .join(
            weekly_user_listening,
            on=["user_id", "week"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("artist_scrobbles")
                / pl.col("user_scrobbles")
                * 100
            )
            .round(2)
            .alias("artist_share_pct")
        )
        .join(artists, on="artist_id", how="left", validate="m:1")
    )
    return (weekly_artist_listening,)


@app.cell
def _(pl, weekly_artist_listening):
    weekly_repetition_summary = (
        weekly_artist_listening
        .group_by("user_id", "active_days")
        .agg(
            pl.len().alias("artist_weeks"),
            pl.col("artist_share_pct").median().round(2).alias("typical_share_pct"),
            pl.col("artist_share_pct").quantile(0.75).round(2).alias("top_25_share_pct"),
            pl.col("artist_share_pct").quantile(0.90).round(2).alias("top_10_share_pct"),
            pl.col("artist_share_pct").max().round(2).alias("max_share_pct"),
            pl.col("artist_scrobbles").median().alias("median_scrobbles"),
        )
        .sort("user_id", "active_days")
        .collect()
    )

    weekly_repetition_summary
    return (weekly_repetition_summary,)


@app.cell
def _(alt, weekly_repetition_summary):
    (
        alt.Chart(weekly_repetition_summary)
        .transform_fold(
            ["typical_share_pct", "top_25_share_pct", "top_10_share_pct"],
            as_=["Statistic", "Listening share"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "active_days:O",
                title="Active days in week",
            ),
            y=alt.Y(
                "Listening share:Q",
                title="Weekly listening share (%)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color("Statistic:N", title=None),
            tooltip=[
                alt.Tooltip("active_days:O", title="Active days"),
                alt.Tooltip("Statistic:N", title="Statistic"),
                alt.Tooltip(
                    "Listening share:Q",
                    title="Listening share (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=320)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Listening spread across more days is usually much stronger. Typical weekly share rises from **0.21% with one active day** to **6.00% with five**, **8.80% with six**, and **17.98% with seven**.

    But a one-day week can still reach **26.79%**. We therefore need both importance and repetition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A strong week is not enough

    To see how often a big week turns into something lasting, we use the first week where an artist lands in the user's strongest 10% of listening weeks, then count how many of the next four weeks include that artist.

    The top 10% is only a marker for this exploration. It is not the breakout threshold.
    """)
    return


@app.cell
def _(pl, weekly_artist_listening):
    _user_week_limits = (
        weekly_artist_listening
        .group_by("user_id")
        .agg(
            pl.col("week").min().alias("first_week"),
            pl.col("week").max().alias("last_week"),
        )
    )

    _ranked_artist_weeks = (
        weekly_artist_listening
        .with_columns(
            (
                pl.col("artist_share_pct").rank("average").over("user_id")
                / pl.len().over("user_id")
            ).alias("importance_percentile"),
        )
        .with_columns(
            (pl.col("importance_percentile") * 10)
            .ceil()
            .clip(1, 10)
            .cast(pl.Int8)
            .alias("importance_decile"),
        )
        .join(
            _user_week_limits,
            on="user_id",
            how="left",
            validate="m:1",
        )
        .filter(
            pl.col("week")
            <= pl.col("last_week") - pl.duration(days=28)
        )
        .with_columns(
            (
                pl.col("week") - pl.col("first_week")
            )
            .dt.total_days()
            .alias("user_history_days")
        )
    )

    _first_strong_week_dates = (
        _ranked_artist_weeks
        .filter(pl.col("importance_decile") == 10)
        .group_by("user_id", "artist_id")
        .agg(pl.col("week").min())
    )

    first_strong_weeks = (
        _ranked_artist_weeks
        .join(
            _first_strong_week_dates,
            on=["user_id", "artist_id", "week"],
            how="inner",
            validate="1:1",
        )
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
            "artist_scrobbles",
            "artist_share_pct",
            "active_days",
            "user_history_days",
        )
    )
    return (first_strong_weeks,)


@app.cell
def _(first_strong_weeks, pl, weekly_artist_listening):
    _future_artist_weeks = pl.concat(
        [
            weekly_artist_listening.select(
                "user_id",
                "artist_id",
                (
                    pl.col("week") - pl.duration(days=7 * _weeks_ahead)
                ).alias("week"),
                pl.lit(_weeks_ahead).alias("weeks_ahead"),
            )
            for _weeks_ahead in range(1, 5)
        ]
    )

    first_strong_week_persistence = (
        first_strong_weeks
        .join(
            _future_artist_weeks,
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="1:m",
        )
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
            "artist_scrobbles",
            "artist_share_pct",
            "active_days",
            "user_history_days",
        )
        .agg(
            pl.col("weeks_ahead")
            .is_not_null()
            .sum()
            .alias("active_weeks_next_4")
        )
        .sort(
            ["active_weeks_next_4", "artist_share_pct"],
            descending=[True, True],
        )
        .collect()
    )

    first_strong_week_persistence
    return (first_strong_week_persistence,)


@app.cell
def _(first_strong_week_persistence, pl):
    first_strong_persistence_summary = (
        first_strong_week_persistence
        .group_by("user_id", "active_weeks_next_4")
        .agg(
            pl.len().alias("artists"),
            pl.col("artist_share_pct")
            .median()
            .round(2)
            .alias("median_strong_week_share_pct"),
        )
        .with_columns(
            (
                pl.col("artists")
                / pl.col("artists").sum().over("user_id")
                * 100
            )
            .round(1)
            .alias("artist_pct")
        )
        .sort("user_id", "active_weeks_next_4")
    )

    first_strong_persistence_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A big first week often fades quickly: **36% of artists disappear for all four following weeks**, while only **8% appear in every one**.

    A single strong week is therefore not enough to call a breakout.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The start of the listening history

    The first months are harder to trust because tracking may have started after some artists were already important.

    We compare first strong weeks by how much listening history came before them.
    """)
    return


@app.cell
def _(first_strong_week_persistence, pl):
    first_strong_by_user_history = (
        first_strong_week_persistence
        .with_columns(
            pl.when(pl.col("user_history_days") < 30)
            .then(pl.lit("<1 month"))
            .when(pl.col("user_history_days") < 90)
            .then(pl.lit("1–3 months"))
            .when(pl.col("user_history_days") < 180)
            .then(pl.lit("3–6 months"))
            .when(pl.col("user_history_days") < 365)
            .then(pl.lit("6–12 months"))
            .otherwise(pl.lit("12+ months"))
            .alias("history_available")
        )
        .group_by("user_id", "history_available")
        .agg(
            pl.len().alias("artists"),
            pl.col("active_weeks_next_4")
            .mean()
            .round(2)
            .alias("avg_active_weeks_next_4"),
            (
                (pl.col("active_weeks_next_4") >= 2).mean() * 100
            )
            .round(1)
            .alias("active_in_2plus_next_4_pct"),
        )
        .with_columns(
            pl.col("history_available")
            .replace(
                {
                    "<1 month": 1,
                    "1–3 months": 2,
                    "3–6 months": 3,
                    "6–12 months": 4,
                    "12+ months": 5,
                }
            )
            .cast(pl.Int8)
            .alias("_history_order")
        )
        .sort("user_id", "_history_order")
        .drop("_history_order")
    )

    first_strong_by_user_history
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `avg_active_weeks_next_4` is simply how many of the next four weeks the artist appears in on average.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Early "first strong weeks" are much more likely to continue. In the first three months, artists appear in about **2.5 of the next four weeks** on average; after a year, that falls to about **1.1**.

    That suggests some early cases were already established when tracking began.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To keep the rest of this dataset analysis cleaner, we compare a few possible cutoffs for how much user history should exist before a candidate event.
    """)
    return


@app.cell
def _(first_strong_week_persistence, pl):
    _warmup_days = {
        "3 months": 90,
        "6 months": 180,
        "9 months": 270,
        "12 months": 365,
    }

    _warmup_rows = []
    for _label, _days in _warmup_days.items():
        _eligible = first_strong_week_persistence.filter(
            pl.col("user_history_days") >= _days
        )

        _warmup_rows.append(
            {
                "warm_up": _label,
                "candidates": _eligible.height,
                "retained_pct": round(
                    _eligible.height
                    / first_strong_week_persistence.height
                    * 100,
                    1,
                ),
                "active_in_2plus_next_4_pct": _eligible.select(
                    (
                        (pl.col("active_weeks_next_4") >= 2).mean()
                        * 100
                    )
                    .round(1)
                ).item(),
            }
        )

    warmup_summary = pl.DataFrame(_warmup_rows)

    warmup_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No cutoff clearly stands out. For the rest of the target analysis, we will only study possible breakout dates after **12 months** of user history. It still keeps **468 of 616 candidates (76.0%)**.

    This is only an analysis choice for this dataset, not a rule that Sonora users need 12 months of history. Earlier artist history still counts when deciding whether the artist had already broken out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Listening before the strong week

    A breakout can happen suddenly or build up from lighter listening. For the rest of this analysis, we only look at strong weeks after the first 12 months, then count how many of the previous 12 weeks included the artist.

    The 12-week lookback is only for this comparison.
    """)
    return


@app.cell
def _(first_strong_week_persistence, pl, weekly_artist_listening):
    _prior_artist_weeks = pl.concat(
        [
            weekly_artist_listening.select(
                "user_id",
                "artist_id",
                (
                    pl.col("week") + pl.duration(days=7 * _weeks_before)
                ).alias("week"),
                pl.col("artist_share_pct").alias("prior_share_pct"),
            )
            for _weeks_before in range(1, 13)
        ]
    )

    first_strong_recent_history = (
        first_strong_week_persistence
        .lazy()
        .filter(pl.col("user_history_days") >= 365)
        .join(
            _prior_artist_weeks,
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="1:m",
        )
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
            "active_weeks_next_4",
        )
        .agg(
            pl.col("prior_share_pct")
            .is_not_null()
            .sum()
            .alias("active_weeks_prev_12")
        )
        .collect()
    )
    return (first_strong_recent_history,)


@app.cell
def _(first_strong_recent_history, pl):
    prior_activity_summary = (
        first_strong_recent_history
        .with_columns(
            pl.when(pl.col("active_weeks_prev_12") == 0)
            .then(pl.lit("0"))
            .when(pl.col("active_weeks_prev_12") == 1)
            .then(pl.lit("1"))
            .otherwise(pl.lit("2+"))
            .alias("prior_active_weeks")
        )
        .group_by("user_id", "prior_active_weeks")
        .agg(
            pl.len().alias("artists"),
            pl.col("active_weeks_next_4")
            .mean()
            .round(2)
            .alias("avg_active_weeks_next_4"),
            (
                (pl.col("active_weeks_next_4") >= 2).mean() * 100
            )
            .round(1)
            .alias("active_in_2plus_next_4_pct"),
        )
        .with_columns(
            pl.col("prior_active_weeks")
            .replace({"0": 0, "1": 1, "2+": 2})
            .cast(pl.Int8)
            .alias("_prior_order")
        )
        .sort("user_id", "_prior_order")
        .drop("_prior_order")
    )

    prior_activity_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some earlier listening makes continued interest more common: **24.3%** of artists with no activity in the previous 12 weeks appear in at least two of the next four, compared with **47.6%** after one prior week and **53.2%** after two or more.

    But sudden adoption still happens. A breakout must allow both gradual buildup and a sharper jump.

    Earlier listening, even isolated strong periods, does not rule out a later first breakout. The artist only stops being eligible once an earlier period actually meets the final breakout rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Real examples

    The tables are easier to judge when we look at real histories. We pick one example of four patterns from the data:

    - **Short burst:** a strong week, then little or no follow-up.
    - **Sudden adoption:** little recent listening, then several active weeks.
    - **Gradual buildup:** repeated lighter listening before the stronger period.
    - **Delayed adoption:** little immediate follow-up, then a stronger period later.

    These rules only choose examples; they are not breakout rules. Each artist is picked from near the middle of its group rather than by hand.

    The vertical line marks the first top-10% week. Larger points mean the artist was played on more days that week.
    """)
    return


@app.cell
def _(first_strong_recent_history, pl, weekly_artist_listening):
    _base_candidates = (
        first_strong_recent_history
        .join(
            weekly_artist_listening
            .select(
                "user_id",
                "artist_id",
                "week",
                "artist_share_pct",
                "active_days",
            )
            .collect(),
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="1:1",
        )
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            pl.col("week").alias("first_strong_week"),
            pl.col("artist_share_pct").alias("first_strong_share_pct"),
            pl.col("active_days").alias("first_strong_active_days"),
            "active_weeks_prev_12",
            "active_weeks_next_4",
        )
    )

    _later_candidates = (
        weekly_artist_listening
        .join(
            _base_candidates
            .filter(pl.col("active_weeks_next_4") <= 1)
            .drop("canonical_name")
            .lazy(),
            on=["user_id", "artist_id"],
            how="inner",
            validate="m:1",
        )
        .with_columns(
            (
                (pl.col("week") - pl.col("first_strong_week"))
                .dt.total_days()
                // 7
            )
            .cast(pl.Int16)
            .alias("relative_week")
        )
        .filter(pl.col("relative_week").is_between(5, 40))
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            "first_strong_week",
            "first_strong_share_pct",
            "first_strong_active_days",
            "active_weeks_prev_12",
            "active_weeks_next_4",
        )
        .agg(
            pl.len().alias("later_active_weeks"),
            pl.col("artist_share_pct")
            .max()
            .round(2)
            .alias("later_peak_share_pct"),
        )
        .filter(
            (pl.col("later_active_weeks") >= 4)
            & (
                pl.col("later_peak_share_pct")
                > pl.col("first_strong_share_pct")
            )
        )
        .with_columns(
            pl.lit("Delayed adoption").alias("example"),
            pl.lit(4, dtype=pl.Int8).alias("example_order"),
        )
        .collect()
    )

    _immediate_candidates = (
        _base_candidates
        .join(
            _later_candidates.select(
                "user_id",
                "artist_id",
                "first_strong_week",
            ),
            on=["user_id", "artist_id", "first_strong_week"],
            how="anti",
        )
        .with_columns(
            pl.when(
                (pl.col("active_weeks_prev_12") == 0)
                & (pl.col("active_weeks_next_4") == 0)
            )
            .then(pl.lit("Short burst"))
            .when(
                (pl.col("active_weeks_prev_12") == 0)
                & (pl.col("active_weeks_next_4") == 4)
            )
            .then(pl.lit("Sudden adoption"))
            .when(
                (pl.col("active_weeks_prev_12") >= 2)
                & (pl.col("active_weeks_next_4") == 4)
            )
            .then(pl.lit("Gradual buildup"))
            .otherwise(None)
            .alias("example")
        )
        .filter(pl.col("example").is_not_null())
        .with_columns(
            pl.col("example")
            .replace(
                {
                    "Short burst": 1,
                    "Sudden adoption": 2,
                    "Gradual buildup": 3,
                }
            )
            .cast(pl.Int8)
            .alias("example_order"),
            pl.lit(None, dtype=pl.UInt32).alias("later_active_weeks"),
            pl.lit(None, dtype=pl.Float64).alias("later_peak_share_pct"),
        )
    )

    _candidate_columns = [
        "example",
        "example_order",
        "user_id",
        "artist_id",
        "canonical_name",
        "first_strong_week",
        "first_strong_share_pct",
        "first_strong_active_days",
        "active_weeks_prev_12",
        "active_weeks_next_4",
        "later_active_weeks",
        "later_peak_share_pct",
    ]

    trajectory_candidates = pl.concat(
        [
            _immediate_candidates.select(_candidate_columns),
            _later_candidates.select(_candidate_columns),
        ]
    )

    trajectory_examples = (
        trajectory_candidates
        .with_columns(
            pl.len().over("example").alias("candidate_count"),
            pl.col("first_strong_share_pct")
            .median()
            .over("example")
            .alias("_group_median_share"),
        )
        .with_columns(
            (
                pl.col("first_strong_share_pct")
                - pl.col("_group_median_share")
            )
            .abs()
            .alias("_share_distance")
        )
        .sort(
            ["example_order", "_share_distance", "canonical_name"]
        )
        .group_by("example", maintain_order=True)
        .first()
        .drop("_group_median_share", "_share_distance")
        .sort("example_order")
    )

    trajectory_examples
    return (trajectory_examples,)


@app.cell(hide_code=True)
def _(pl, trajectory_examples, weekly_artist_listening):
    _relative_weeks = pl.DataFrame(
        {"relative_week": list(range(-12, 41))}
    )

    _trajectory_grid = (
        trajectory_examples
        .select(
            "example",
            "example_order",
            "user_id",
            "artist_id",
            "canonical_name",
            "first_strong_week",
        )
        .join(_relative_weeks, how="cross")
        .with_columns(
            (
                pl.col("first_strong_week")
                + pl.duration(days=7) * pl.col("relative_week")
            ).alias("week")
        )
    )

    _weekly_example_history = (
        weekly_artist_listening
        .filter(
            pl.col("artist_id").is_in(
                trajectory_examples["artist_id"].to_list()
            )
        )
        .select(
            "user_id",
            "artist_id",
            "week",
            "artist_scrobbles",
            "artist_share_pct",
            "active_days",
        )
        .collect()
    )

    trajectory_example_history = (
        _trajectory_grid
        .join(
            _weekly_example_history,
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("artist_scrobbles").fill_null(0),
            pl.col("artist_share_pct").fill_null(0.0),
            pl.col("active_days").fill_null(0),
            pl.concat_str(
                pl.col("example_order").cast(pl.String),
                pl.lit(". "),
                "example",
                pl.lit(" - "),
                "canonical_name",
            ).alias("example_label"),
        )
        .sort("example_order", "relative_week")
    )
    return (trajectory_example_history,)


@app.cell
def _(alt, trajectory_example_history):
    _base = alt.Chart(trajectory_example_history).encode(
        x=alt.X(
            "relative_week:Q",
            title="Weeks from first strong week",
            scale=alt.Scale(domain=[-12, 40]),
        ),
        y=alt.Y(
            "artist_share_pct:Q",
            title="Weekly listening share (%)",
            scale=alt.Scale(zero=True),
        ),
        tooltip=[
            alt.Tooltip("canonical_name:N", title="Artist"),
            alt.Tooltip("week:T", title="Week"),
            alt.Tooltip(
                "artist_share_pct:Q",
                title="Listening share (%)",
                format=".2f",
            ),
            alt.Tooltip("artist_scrobbles:Q", title="Scrobbles"),
            alt.Tooltip("active_days:Q", title="Active days"),
        ],
    )

    _line = _base.mark_line()
    _points = (
        _base
        .transform_filter("datum.active_days > 0")
        .mark_circle()
        .encode(
            size=alt.Size(
                "active_days:Q",
                title="Active days",
                scale=alt.Scale(range=[20, 140]),
            )
        )
    )
    _first_strong_rule = (
        alt.Chart(trajectory_example_history)
        .transform_filter("datum.relative_week == 0")
        .mark_rule(strokeDash=[4, 4])
        .encode(x=alt.X("relative_week:Q"))
    )

    (
        alt.layer(_line, _points, _first_strong_rule)
        .facet(
            facet=alt.Facet(
                "example_label:N",
                title=None,
                sort="ascending",
            ),
            columns=2,
        )
        .properties(spacing=20)
    )
    return


@app.cell(hide_code=True)
def _(mo, trajectory_examples):
    _names = {
        _row["example"]: _row["canonical_name"]
        for _row in trajectory_examples.iter_rows(named=True)
    }

    mo.md(f"""
    **{_names['Short burst']}** fades after the first strong week. **{_names['Sudden adoption']}** jumps from little recent activity into repeated listening, while **{_names['Gradual buildup']}** was already appearing beforehand.

    **{_names['Delayed adoption']}** becomes more active later and reaches a stronger peak. These examples show why the first big week is useful to inspect, but cannot define breakout by itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repeated low-level listening

    Repetition is not enough either. We also show the artist with the most active weeks while never reaching the user's strongest 10% of listening weeks.
    """)
    return


@app.cell
def _(pl, weekly_artist_listening):
    background_example_summary = (
        weekly_artist_listening
        .with_columns(
            (
                pl.col("artist_share_pct").rank("average").over("user_id")
                / pl.len().over("user_id")
                * 100
            ).alias("importance_percentile")
        )
        .group_by("user_id", "artist_id", "canonical_name")
        .agg(
            pl.len().alias("active_weeks"),
            pl.col("week").min().alias("first_week"),
            pl.col("week").max().alias("last_week"),
            pl.col("artist_share_pct")
            .median()
            .round(2)
            .alias("median_share_pct"),
            pl.col("artist_share_pct")
            .max()
            .round(2)
            .alias("max_share_pct"),
            pl.col("importance_percentile")
            .max()
            .round(1)
            .alias("max_importance_percentile"),
        )
        .filter(pl.col("max_importance_percentile") < 90)
        .sort("active_weeks", descending=True)
        .head(1)
        .collect()
    )

    background_example_summary
    return (background_example_summary,)


@app.cell
def _(background_example_summary, pl, weekly_artist_listening):
    background_example_history = (
        weekly_artist_listening
        .filter(
            pl.col("artist_id")
            == background_example_summary["artist_id"][0]
        )
        .select(
            "canonical_name",
            "week",
            "artist_scrobbles",
            "artist_share_pct",
            "active_days",
        )
        .sort("week")
        .collect()
    )
    return (background_example_history,)


@app.cell
def _(alt, background_example_history):
    (
        alt.Chart(background_example_history)
        .mark_circle()
        .encode(
            x=alt.X("week:T", title="Week"),
            y=alt.Y(
                "artist_share_pct:Q",
                title="Weekly listening share (%)",
                scale=alt.Scale(zero=True),
            ),
            size=alt.Size(
                "active_days:Q",
                title="Active days",
                scale=alt.Scale(range=[20, 140]),
            ),
            tooltip=[
                alt.Tooltip("canonical_name:N", title="Artist"),
                alt.Tooltip("week:T", title="Week"),
                alt.Tooltip(
                    "artist_share_pct:Q",
                    title="Listening share (%)",
                    format=".2f",
                ),
                alt.Tooltip("artist_scrobbles:Q", title="Scrobbles"),
                alt.Tooltip("active_days:Q", title="Active days"),
            ],
        )
        .properties(height=240)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This artist appears regularly for a long time without a comparable rise in importance. Breakout therefore needs a real increase in importance as well as repeated listening.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. What should "important" mean?

    Weekly listening share is easy to read: **5% means one in twenty scrobbles that week were the artist**. But the same percentage can be unusually high for one user and ordinary for another.

    A better fit for Sonora is to compare an artist's weekly share with that user's own earlier artist-weeks. The table shows, at the start of each year, the share needed to be among the strongest 20%, 10%, or 5% of weeks seen before then.

    These are examples to understand the scale, not thresholds we are choosing.
    """)
    return


@app.cell
def _(pl, weekly_artist_listening):
    from datetime import date as _date

    _weekly_importance_history = (
        weekly_artist_listening
        .select("user_id", "week", "artist_share_pct")
        .collect()
    )

    _importance_rows = []
    for _user_id in _weekly_importance_history["user_id"].unique().to_list():
        _user_history = (
            _weekly_importance_history
            .filter(pl.col("user_id") == _user_id)
        )
        _first_week = _user_history["week"].min()
        _last_week = _user_history["week"].max()

        for _year in range(_first_week.year + 1, _last_week.year + 1):
            _cutoff = _date(_year, 1, 1)

            if (_cutoff - _first_week).days < 365:
                continue

            _past_shares = (
                _user_history
                .filter(pl.col("week") < _cutoff)
                ["artist_share_pct"]
            )

            _importance_rows.append(
                {
                    "user_id": _user_id,
                    "year": _year,
                    "top_20pct_share": round(
                        float(_past_shares.quantile(0.80)),
                        2,
                    ),
                    "top_10pct_share": round(
                        float(_past_shares.quantile(0.90)),
                        2,
                    ),
                    "top_5pct_share": round(
                        float(_past_shares.quantile(0.95)),
                        2,
                    ),
                }
            )

    importance_reference_summary = (
        pl.DataFrame(_importance_rows)
        .sort("user_id", "year")
    )

    importance_reference_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same relative level does not always correspond to the same raw share. For example, the share needed to be in the strongest **10%** of earlier artist-weeks rises from **2.39% in 2022** to about **2.90% in 2026**.

    So we keep **weekly listening share** as the measure of importance, but judge whether it is unusually high against the user's own listening.

    The next question is how much of that earlier history should count.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How much earlier history should count?

    Using every past year could make the comparison slow to adapt as someone's listening changes.

    We keep the strongest 10% as an example and compare three baselines: all earlier history, the previous 24 months, and the previous 12 months. The 10% level itself is still not decided.
    """)
    return


@app.cell
def _(pl, weekly_artist_listening):
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    _importance_history = (
        weekly_artist_listening
        .select("user_id", "week", "artist_share_pct")
        .collect()
    )

    _baseline_rows = []
    for _user_id in _importance_history["user_id"].unique().to_list():
        _user_history = _importance_history.filter(
            pl.col("user_id") == _user_id
        )
        _first_week = _user_history["week"].min()
        _last_week = _user_history["week"].max()

        for _year in range(_first_week.year + 1, _last_week.year + 1):
            _cutoff = _date(_year, 1, 1)

            if (_cutoff - _first_week).days < 730:
                continue

            for _baseline, _days in [
                ("all_history", None),
                ("previous_24_months", 730),
                ("previous_12_months", 365),
            ]:
                _past = _user_history.filter(pl.col("week") < _cutoff)

                if _days is not None:
                    _past = _past.filter(
                        pl.col("week")
                        >= _cutoff - _timedelta(days=_days)
                    )

                _baseline_rows.append(
                    {
                        "user_id": _user_id,
                        "year": _year,
                        "baseline": _baseline,
                        "top_10_share_pct": round(
                            float(
                                _past["artist_share_pct"].quantile(0.90)
                            ),
                            2,
                        ),
                    }
                )

    importance_baseline_summary = (
        pl.DataFrame(_baseline_rows)
        .pivot(
            on="baseline",
            index=["user_id", "year"],
            values="top_10_share_pct",
        )
        .sort("user_id", "year")
    )

    importance_baseline_summary
    return


if __name__ == "__main__":
    app.run()

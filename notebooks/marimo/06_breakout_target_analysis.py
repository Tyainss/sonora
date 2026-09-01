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
    import datetime as dt
    import altair as alt
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return alt, dt, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Artist trajectories

    Individual histories help us see the patterns the definition needs to handle: short bursts, gradual buildup, long gaps, and stronger periods that repeat.
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

    To see how often a big week becomes more than a one-off spike, we use the first week where an artist lands in the user's strongest 10% of listening weeks, then count how many of the next four weeks include that artist.

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
    Some earlier listening makes follow-up more common: **24.3%** of artists with no activity in the previous 12 weeks appear in at least two of the next four, compared with **47.6%** after one prior week and **53.2%** after two or more.

    But a breakout can still happen suddenly. The definition must allow both gradual buildup and a sharper jump.

    Earlier listening, even isolated strong periods, does not rule out a later first breakout. The artist only stops being eligible once an earlier period actually meets the final breakout rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Real examples

    The tables are easier to judge when we look at real histories. We pick one example of four patterns from the data:

    - **Short burst:** a strong week, then little or no follow-up.
    - **Sudden rise:** little recent listening, then several active weeks.
    - **Gradual buildup:** repeated lighter listening before the stronger period.
    - **Delayed rise:** little immediate follow-up, then a stronger period later.

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
            pl.lit("Delayed rise").alias("example"),
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
            .then(pl.lit("Sudden rise"))
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
                    "Sudden rise": 2,
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
    **{_names['Short burst']}** fades after the first strong week. **{_names['Sudden rise']}** jumps from little recent activity into repeated listening, while **{_names['Gradual buildup']}** was already appearing beforehand.

    **{_names['Delayed rise']}** becomes more active later and reaches a stronger peak. These examples show why the first big week is useful to inspect, but cannot define breakout by itself.
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
def _(dt, pl, weekly_artist_listening):
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
            _cutoff = dt.date(_year, 1, 1)

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
def _(dt, pl, weekly_artist_listening):
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
            _cutoff = dt.date(_year, 1, 1)

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
                        >= _cutoff - dt.timedelta(days=_days)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recent history gives a somewhat different picture from using everything the user has ever listened to. The difference becomes especially visible from 2024 onward, while the 12- and 24-month comparisons stay fairly close.

    We will use the **previous 24 months** as the working comparison window: recent enough to follow changing listening habits, while using more history than a single year.

    This is a maximum lookback, not a requirement that a user already has 24 months of data. We will revisit the window when we test how sensitive the final breakout rule is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. How high should importance be?

    We compare three possible levels: being among the strongest 20%, 10%, or 5% of artist-weeks from the previous 24 months.

    For each level, we take the first week an artist reaches it and look at how strong that week is and whether the artist keeps appearing afterwards.
    """)
    return


@app.cell
def _(dt, pl, weekly_artist_listening):
    _importance_history = (
        weekly_artist_listening
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
            "artist_share_pct",
            "active_days",
        )
        .collect()
    )

    _cutoff_rows = []
    for _user_id in _importance_history["user_id"].unique().to_list():
        _user_history = _importance_history.filter(
            pl.col("user_id") == _user_id
        )
        _first_week = _user_history["week"].min()
        _last_week = _user_history["week"].max()

        for _week in _user_history["week"].unique().sort().to_list():
            if (_week - _first_week).days < 365:
                continue

            _past = _user_history.filter(
                (pl.col("week") < _week)
                & (
                    pl.col("week")
                    >= _week - dt.timedelta(days=730)
                )
            )["artist_share_pct"]

            _cutoff_rows.append(
                {
                    "user_id": _user_id,
                    "week": _week,
                    "top_20_cutoff": float(_past.quantile(0.80)),
                    "top_10_cutoff": float(_past.quantile(0.90)),
                    "top_5_cutoff": float(_past.quantile(0.95)),
                }
            )

    _importance_cutoffs = pl.DataFrame(_cutoff_rows)

    weekly_importance = _importance_history.join(
        _importance_cutoffs,
        on=["user_id", "week"],
        how="inner",
        validate="m:1",
    )

    _future_activity = pl.concat(
        [
            _importance_history.select(
                "user_id",
                "artist_id",
                (
                    pl.col("week")
                    - dt.timedelta(days=7 * _weeks_ahead)
                ).alias("week"),
                pl.lit(_weeks_ahead).alias("_weeks_ahead"),
            )
            for _weeks_ahead in range(1, 5)
        ]
    )

    _summary_rows = []
    for _label, _cutoff in [
        ("Top 20%", "top_20_cutoff"),
        ("Top 10%", "top_10_cutoff"),
        ("Top 5%", "top_5_cutoff"),
    ]:
        _first_qualifying = (
            weekly_importance
            .filter(pl.col("artist_share_pct") >= pl.col(_cutoff))
            .with_columns(
                pl.col("week")
                .max()
                .over("user_id")
                .alias("_last_week")
            )
            .filter(
                pl.col("week")
                <= pl.col("_last_week") - dt.timedelta(days=28)
            )
            .drop("_last_week")
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )

        _followup = (
            _first_qualifying
            .join(
                _future_activity,
                on=["user_id", "artist_id", "week"],
                how="left",
                validate="1:m",
            )
            .group_by("user_id", "artist_id")
            .agg(
                pl.col("artist_share_pct").first(),
                pl.col("active_days").first(),
                pl.col("_weeks_ahead")
                .is_not_null()
                .sum()
                .alias("active_weeks_next_4"),
            )
        )

        _summary_rows.append(
            {
                "importance_level": _label,
                "typical_cutoff_pct": round(
                    float(_importance_cutoffs[_cutoff].median()),
                    2,
                ),
                "artists": _followup.height,
                "typical_share_pct": round(
                    float(_followup["artist_share_pct"].median()),
                    2,
                ),
                "typical_active_days": float(
                    _followup["active_days"].median()
                ),
                "no_activity_next_4_pct": round(
                    float(
                        (
                            _followup["active_weeks_next_4"] == 0
                        ).mean()
                        * 100
                    ),
                    1,
                ),
                "active_in_2plus_next_4_pct": round(
                    float(
                        (
                            _followup["active_weeks_next_4"] >= 2
                        ).mean()
                        * 100
                    ),
                    1,
                ),
            }
        )

    importance_level_summary = pl.DataFrame(_summary_rows)

    importance_level_summary
    return (weekly_importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The top 20% level still includes many fairly light weeks: the typical first qualifying week has only one active day, and more than half of the artists disappear for the next four weeks.

    The top 5% level is much stronger, but already asks for a fairly intense week before repetition is considered.

    We will use the **top 10%** as the working definition of an important week. It marks a clear rise in listening without requiring the first week itself to be extreme.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. How much repetition is enough?

    A breakout can be short-lived. We only need enough repetition to show that the first important week was more than a one-off spike.

    We compare requiring a second important week within a 2-, 3-, 4-, or 5-week period. Nothing after that period is required.
    """)
    return


@app.cell
def _(dt, pl, weekly_importance):
    _user_last_week = (
        weekly_importance
        .group_by("user_id")
        .agg(pl.col("week").max().alias("_last_week"))
    )

    _important_weeks = (
        weekly_importance
        .filter(
            pl.col("artist_share_pct")
            >= pl.col("top_10_cutoff")
        )
        .join(
            _user_last_week,
            on="user_id",
            how="left",
            validate="m:1",
        )
    )

    _repetition_rows = []
    for _followup_weeks in range(1, 5):
        _anchors = (
            _important_weeks
            .filter(
                pl.col("week")
                <= pl.col("_last_week")
                - dt.timedelta(days=7 * _followup_weeks)
            )
            .drop("_last_week")
        )

        _future_importance = pl.concat(
            [
                weekly_importance.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("week")
                        - dt.timedelta(days=7 * _weeks_ahead)
                    ).alias("week"),
                    (
                        pl.col("artist_share_pct")
                        >= pl.col("top_10_cutoff")
                    ).alias("_future_is_important"),
                )
                for _weeks_ahead in range(1, _followup_weeks + 1)
            ]
        )

        _qualifying = (
            _anchors
            .join(
                _future_importance,
                on=["user_id", "artist_id", "week"],
                how="left",
                validate="1:m",
            )
            .group_by("user_id", "artist_id", "week")
            .agg(
                pl.col("_future_is_important")
                .fill_null(False)
                .sum()
                .alias("_extra_important_weeks")
            )
            .filter(pl.col("_extra_important_weeks") >= 1)
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )

        _repetition_rows.append(
            {
                "period_weeks": _followup_weeks + 1,
                "candidate_artists": _qualifying.height,
            }
        )

    repetition_window_summary = pl.DataFrame(_repetition_rows)

    repetition_window_summary
    return (repetition_window_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Requiring important listening in consecutive weeks finds **162 artists**. Allowing a 3-week period raises that to **200**, while a fourth week adds only **10 more**. Extending to five weeks adds another **13**.

    We will use **two important weeks within a 3-week period**. This allows one quiet week between them while keeping both strong weeks part of the same short period.

    The artist does not need to stay important afterwards.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. The breakout event

    We can now combine the pieces:

    - an **important week** is in the user's strongest 10% of artist-weeks, using up to the previous 24 months;
    - the artist must have **at least two important weeks within a 3-week period**;
    - the first important week in the first qualifying period is the breakout week;
    - earlier weak listening and isolated strong weeks are allowed.

    For this dataset, we do not assign breakout dates during the uncertain first year. Artists that already show the same repeated-important pattern there are treated as pre-existing for this analysis.

    This does not make 12 months a requirement for future users. How to handle limited listening history in production remains open.
    """)
    return


@app.cell(hide_code=True)
def _(dt, pl, weekly_artist_listening, weekly_importance):
    _user_last_week = (
        weekly_importance
        .group_by("user_id")
        .agg(pl.col("week").max().alias("_last_week"))
    )

    _important_anchors = (
        weekly_importance
        .filter(
            pl.col("artist_share_pct")
            >= pl.col("top_10_cutoff")
        )
        .join(
            _user_last_week,
            on="user_id",
            how="left",
            validate="m:1",
        )
        .filter(
            pl.col("week")
            <= pl.col("_last_week") - dt.timedelta(days=14)
        )
        .drop("_last_week")
    )

    _future_important = pl.concat(
        [
            weekly_importance.select(
                "user_id",
                "artist_id",
                (
                    pl.col("week")
                    - dt.timedelta(days=7 * _weeks_ahead)
                ).alias("week"),
                pl.lit(_weeks_ahead).alias("_weeks_ahead"),
                (
                    pl.col("artist_share_pct")
                    >= pl.col("top_10_cutoff")
                ).alias("_future_is_important"),
            )
            for _weeks_ahead in range(1, 3)
        ]
    )

    _candidate_events = (
        _important_anchors
        .join(
            _future_important,
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="1:m",
        )
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
            "artist_share_pct",
            "active_days",
        )
        .agg(
            pl.col("_future_is_important")
            .fill_null(False)
            .sum()
            .alias("_extra_important_weeks"),
            pl.col("_weeks_ahead")
            .filter(pl.col("_future_is_important"))
            .min()
            .alias("weeks_until_second_important"),
        )
        .filter(pl.col("_extra_important_weeks") >= 1)
        .sort("week")
        .group_by("user_id", "artist_id", maintain_order=True)
        .first()
    )

    _weekly_history = weekly_artist_listening.collect()
    _user_first_week = (
        _weekly_history
        .group_by("user_id")
        .agg(pl.col("week").min().alias("_first_week"))
    )

    _history_with_first_week = (
        _weekly_history
        .join(
            _user_first_week,
            on="user_id",
            how="left",
            validate="m:1",
        )
    )

    _warmup_weeks = (
        _history_with_first_week
        .filter(
            pl.col("week")
            < pl.col("_first_week") + pl.duration(days=365)
        )
    )

    _warmup_cutoff = (
        _warmup_weeks
        .group_by("user_id")
        .agg(
            pl.col("artist_share_pct")
            .quantile(0.90)
            .alias("_warmup_top_10_cutoff")
        )
    )

    _warmup_importance = (
        _history_with_first_week
        .join(
            _warmup_cutoff,
            on="user_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("artist_share_pct")
                >= pl.col("_warmup_top_10_cutoff")
            ).alias("_is_warmup_important")
        )
    )

    _warmup_anchors = (
        _warmup_importance
        .filter(
            (
                pl.col("week")
                < pl.col("_first_week") + pl.duration(days=365)
            )
            & pl.col("_is_warmup_important")
        )
    )

    _warmup_future = pl.concat(
        [
            _warmup_importance.select(
                "user_id",
                "artist_id",
                (
                    pl.col("week")
                    - dt.timedelta(days=7 * _weeks_ahead)
                ).alias("week"),
                pl.col("_is_warmup_important")
                .alias("_future_is_warmup_important"),
            )
            for _weeks_ahead in range(1, 3)
        ]
    )

    warmup_established_artists = (
        _warmup_anchors
        .join(
            _warmup_future,
            on=["user_id", "artist_id", "week"],
            how="left",
            validate="1:m",
        )
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            "week",
        )
        .agg(
            pl.col("_future_is_warmup_important")
            .fill_null(False)
            .sum()
            .alias("_extra_important_weeks")
        )
        .filter(pl.col("_extra_important_weeks") >= 1)
        .sort("week")
        .group_by(
            "user_id",
            "artist_id",
            "canonical_name",
            maintain_order=True,
        )
        .first()
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            pl.col("week").alias("established_in_warmup_week"),
        )
    )

    _removed_as_already_established = (
        _candidate_events
        .join(
            warmup_established_artists.select(
                "user_id",
                "artist_id",
            ),
            on=["user_id", "artist_id"],
            how="inner",
        )
    )

    breakout_events = (
        _candidate_events
        .join(
            warmup_established_artists.select(
                "user_id",
                "artist_id",
            ),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            "user_id",
            "artist_id",
            "canonical_name",
            pl.col("week").alias("breakout_week"),
            pl.col("artist_share_pct").alias("breakout_share_pct"),
            pl.col("active_days").alias("breakout_active_days"),
            "weeks_until_second_important",
        )
        .sort("breakout_week")
    )

    breakout_rule_summary = pl.DataFrame(
        {
            "artists_established_in_first_year": [
                warmup_established_artists.height
            ],
            "later_candidates_removed": [
                _removed_as_already_established.height
            ],
            "breakout_events": [breakout_events.height],
        }
    )

    breakout_rule_summary
    return breakout_events, breakout_rule_summary, warmup_established_artists


@app.cell
def _(breakout_events, pl):
    breakout_events_by_year = (
        breakout_events
        .with_columns(
            pl.col("breakout_week").dt.year().alias("year")
        )
        .group_by("year")
        .agg(pl.len().alias("breakout_events"))
        .sort("year")
    )

    breakout_events_by_year
    return (breakout_events_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All breakout artists

    The full list is shown below so we can inspect the labels directly.
    """)
    return


@app.cell
def _(breakout_events, pl):
    breakout_artist_list = (
        breakout_events
        .with_columns(
            pl.col("breakout_week").dt.year().alias("year")
        )
        .select(
            "year",
            pl.col("canonical_name").alias("artist"),
            "breakout_week",
        )
        .sort("year", "breakout_week", "artist")
    )

    breakout_artist_list
    return (breakout_artist_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **92 artists** already show this repeated-important pattern during the first year, so we do not treat later strong periods from those artists as first breakouts. This removes **37 later candidates**.

    The working rule leaves **163 breakout events** after the first-year analysis period.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. How sensitive is the rule?

    We vary the main choices around the working rule. `working_events_kept_pct` shows how many of the **163** working breakout artists are still found under each alternative.
    """)
    return


@app.cell(hide_code=True)
def _(breakout_events, dt, pl, weekly_artist_listening):
    _weekly_history = weekly_artist_listening.collect()
    _importance_cache = {}
    _warmup_cache = {}

    def _build_importance(_quantile, _lookback_days):
        _cache_key = (_quantile, _lookback_days)
        if _cache_key in _importance_cache:
            return _importance_cache[_cache_key]

        _rows = []
        for _user_id in _weekly_history["user_id"].unique().to_list():
            _user_history = _weekly_history.filter(
                pl.col("user_id") == _user_id
            )
            _first_week = _user_history["week"].min()

            for _week in _user_history["week"].unique().sort().to_list():
                if (_week - _first_week).days < 365:
                    continue

                _past = _user_history.filter(pl.col("week") < _week)
                if _lookback_days is not None:
                    _past = _past.filter(
                        pl.col("week")
                        >= _week - dt.timedelta(days=_lookback_days)
                    )

                _rows.append(
                    {
                        "user_id": _user_id,
                        "week": _week,
                        "_cutoff": float(
                            _past["artist_share_pct"].quantile(_quantile)
                        ),
                    }
                )

        _scored = (
            _weekly_history
            .join(
                pl.DataFrame(_rows),
                on=["user_id", "week"],
                how="inner",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("artist_share_pct") >= pl.col("_cutoff")
                ).alias("_is_important")
            )
        )
        _importance_cache[_cache_key] = _scored
        return _scored

    def _warmup_established(_quantile, _followup_weeks, _extra_weeks):
        _cache_key = (_quantile, _followup_weeks, _extra_weeks)
        if _cache_key in _warmup_cache:
            return _warmup_cache[_cache_key]

        _first = (
            _weekly_history
            .group_by("user_id")
            .agg(pl.col("week").min().alias("_first_week"))
        )
        _history = _weekly_history.join(
            _first,
            on="user_id",
            how="left",
            validate="m:1",
        )
        _warmup = _history.filter(
            pl.col("week")
            < pl.col("_first_week") + pl.duration(days=365)
        )
        _cutoff = (
            _warmup
            .group_by("user_id")
            .agg(
                pl.col("artist_share_pct")
                .quantile(_quantile)
                .alias("_cutoff")
            )
        )
        _scored = (
            _history
            .join(
                _cutoff,
                on="user_id",
                how="left",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("artist_share_pct") >= pl.col("_cutoff")
                ).alias("_is_important")
            )
        )
        _anchors = _scored.filter(
            (
                pl.col("week")
                < pl.col("_first_week") + pl.duration(days=365)
            )
            & pl.col("_is_important")
        )
        _future = pl.concat(
            [
                _scored.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("week")
                        - dt.timedelta(days=7 * _weeks_ahead)
                    ).alias("week"),
                    pl.col("_is_important").alias("_future_important"),
                )
                for _weeks_ahead in range(1, _followup_weeks + 1)
            ]
        )
        _established = (
            _anchors
            .join(
                _future,
                on=["user_id", "artist_id", "week"],
                how="left",
                validate="1:m",
            )
            .group_by("user_id", "artist_id", "week")
            .agg(
                pl.col("_future_important")
                .fill_null(False)
                .sum()
                .alias("_extra_important_weeks")
            )
            .filter(
                pl.col("_extra_important_weeks") >= _extra_weeks
            )
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )
        _warmup_cache[_cache_key] = _established
        return _established

    def _events_for(
        _quantile,
        _lookback_days,
        _followup_weeks,
        _extra_weeks,
    ):
        _scored = _build_importance(_quantile, _lookback_days)
        _last = (
            _scored
            .group_by("user_id")
            .agg(pl.col("week").max().alias("_last_week"))
        )
        _anchors = (
            _scored
            .filter(pl.col("_is_important"))
            .join(
                _last,
                on="user_id",
                how="left",
                validate="m:1",
            )
            .filter(
                pl.col("week")
                <= pl.col("_last_week")
                - dt.timedelta(days=7 * _followup_weeks)
            )
            .drop("_last_week")
        )
        _future = pl.concat(
            [
                _scored.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("week")
                        - dt.timedelta(days=7 * _weeks_ahead)
                    ).alias("week"),
                    pl.col("_is_important").alias("_future_important"),
                )
                for _weeks_ahead in range(1, _followup_weeks + 1)
            ]
        )
        _events = (
            _anchors
            .join(
                _future,
                on=["user_id", "artist_id", "week"],
                how="left",
                validate="1:m",
            )
            .group_by("user_id", "artist_id", "week")
            .agg(
                pl.col("_future_important")
                .fill_null(False)
                .sum()
                .alias("_extra_important_weeks")
            )
            .filter(
                pl.col("_extra_important_weeks") >= _extra_weeks
            )
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )
        _already_established = _warmup_established(
            _quantile,
            _followup_weeks,
            _extra_weeks,
        )
        return (
            _events
            .join(
                _already_established.select("user_id", "artist_id"),
                on=["user_id", "artist_id"],
                how="anti",
            )
            .select("user_id", "artist_id")
        )

    _working = breakout_events.select("user_id", "artist_id")
    _working_keys = set(_working.iter_rows())

    _alternatives = [
        ("Working rule", 0.90, 730, 2, 1),
        ("12-month baseline", 0.90, 365, 2, 1),
        ("All earlier history", 0.90, None, 2, 1),
        ("Top 20% importance", 0.80, 730, 2, 1),
        ("Top 5% importance", 0.95, 730, 2, 1),
        ("2 important weeks within 2 weeks", 0.90, 730, 1, 1),
        ("2 important weeks within 4 weeks", 0.90, 730, 3, 1),
        ("2 important weeks within 5 weeks", 0.90, 730, 4, 1),
        ("3 important weeks within 3 weeks", 0.90, 730, 2, 2),
    ]

    _sensitivity_rows = []
    for _label, _quantile, _lookback, _followup, _extra in _alternatives:
        _events = _events_for(
            _quantile,
            _lookback,
            _followup,
            _extra,
        )
        _keys = set(_events.iter_rows())
        _sensitivity_rows.append(
            {
                "rule": _label,
                "events": _events.height,
                "working_events_kept_pct": round(
                    len(_working_keys & _keys)
                    / len(_working_keys)
                    * 100,
                    1,
                ),
            }
        )

    sensitivity_summary = pl.DataFrame(_sensitivity_rows)

    sensitivity_summary
    return (sensitivity_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rule is not very sensitive to the exact history window or whether the repetition period is two, three, four, or five weeks. The stricter **top 5%** level and requiring **three important weeks** remove far more events, which is expected because they change the meaning of breakout itself.

    We will keep the **24-month baseline, top 10% importance, and two important weeks within three weeks** as the working definition and revisit these choices when we have evidence from more users.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Working definition

    A breakout is the artist's **first period with at least two important weeks within three weeks**, where importance means being in the user's strongest 10% of artist-weeks compared with up to the previous 24 months.

    The breakout week is the first important week in that period. We only know that it qualifies once the second important week happens, which matters when we later turn this event into daily prediction labels.

    This gives us the breakout event. The next phase is to define the daily prediction rows and candidate eligibility without leaking future information.
    """)
    return


if __name__ == "__main__":
    app.run()

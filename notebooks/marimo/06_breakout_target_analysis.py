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

    Looks at how artist listening changes over time, with a focus on the difference between brief interest and stronger, repeated importance.
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

    We start by looking at individual artist histories to see what different listening patterns actually look like: isolated bursts, gradual growth, repeated low-level listening, and stronger adoption.
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
    Several of the most-listened artists reach **100% daily listening share** at least once, so the strongest single day is too noisy to say much about sustained importance on its own.
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
    _gap_text = f"{_gap:,} days" if _gap is not None else "—"

    mo.md(f"""
    **{_selected_summary['canonical_name']}**  
    **Scrobbles:** {_selected_summary['scrobbles']:,} · **Active days:** {_selected_summary['active_days']:,} · **First seen:** {_selected_summary['first_seen']} · **Last seen:** {_selected_summary['last_seen']} · **Longest gap:** {_gap_text}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A weekly view makes it easier to see whether strong listening is spread across several days or concentrated in a short burst.
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
    ## 2. Importance and repeated listening

    A breakout should mean more than one busy listening day. We compare an artist's share of weekly listening with the number of days they were played to see how importance and repetition relate.
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
            pl.col("artist_share_pct").median().round(2).alias("median_share_pct"),
            pl.col("artist_share_pct").quantile(0.75).round(2).alias("p75_share_pct"),
            pl.col("artist_share_pct").quantile(0.90).round(2).alias("p90_share_pct"),
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
            ["median_share_pct", "p75_share_pct", "p90_share_pct"],
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
    Weeks with listening spread across more days are usually much more important. Median weekly share rises from **0.21% with one active day** to **6.00% with five**, **8.80% with six**, and **17.98% with seven**.

    A single-day week can still reach **26.79%** of all listening, though, so neither importance nor repetition is enough on its own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. First strong weeks

    Some artists have one big week and disappear, while others keep showing up afterwards. We take the first time each artist reaches the user's top 10% of artist-weeks and check what happens over the next four weeks.

    The top 10% cut is only a convenient way to surface relatively strong listening here; the breakout threshold is still open.
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
            pl.col("week")
            .rank("ordinal")
            .over(["user_id", "artist_id"])
            .cast(pl.Int32)
            .alias("artist_active_week_number"),
            (
                pl.col("artist_share_pct").rank("average").over("user_id")
                / pl.len().over("user_id")
            ).alias("importance_percentile"),
        )
        .with_columns(
            (pl.col("artist_active_week_number") - 1)
            .alias("prior_active_weeks"),
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
            "prior_active_weeks",
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
                pl.col("artist_share_pct").alias("future_share_pct"),
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
            "prior_active_weeks",
            "user_history_days",
        )
        .agg(
            pl.col("future_share_pct")
            .is_not_null()
            .sum()
            .alias("active_weeks_next_4"),
            (
                (pl.col("weeks_ahead") == 1)
                & pl.col("future_share_pct").is_not_null()
            )
            .any()
            .alias("active_next_week"),
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
            pl.col("active_days")
            .median()
            .alias("median_active_days"),
            pl.col("prior_active_weeks")
            .median()
            .alias("median_prior_active_weeks"),
        )
        .sort("user_id", "active_weeks_next_4")
    )

    first_strong_persistence_summary
    return (first_strong_persistence_summary,)


@app.cell
def _(alt, first_strong_persistence_summary):
    (
        alt.Chart(first_strong_persistence_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "active_weeks_next_4:O",
                title="Active weeks in following 4 weeks",
            ),
            y=alt.Y(
                "artists:Q",
                title="Artists",
            ),
            detail="user_id:N",
            tooltip=[
                alt.Tooltip(
                    "active_weeks_next_4:O",
                    title="Active weeks next 4",
                ),
                alt.Tooltip("artists:Q", title="Artists"),
                alt.Tooltip(
                    "median_strong_week_share_pct:Q",
                    title="Median strong-week share (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "median_active_days:Q",
                    title="Median active days",
                    format=".1f",
                ),
                alt.Tooltip(
                    "median_prior_active_weeks:Q",
                    title="Median prior active weeks",
                    format=".1f",
                ),
            ],
        )
        .properties(height=320)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A first strong week often does **not** turn into sustained listening. **36% of artists disappear for all four following weeks**, while only **8% appear in every one of them**.

    Artists that keep returning tend to have a somewhat stronger and more spread-out first strong week, but there is still plenty of overlap. A big week by itself does not cleanly separate a lasting adoption from a short burst.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. History before a first strong week

    The beginning of a user's listening history is harder to interpret. An artist may look like a new breakout simply because tracking started after they were already important.

    We compare first strong weeks by how much user history existed beforehand to see how large that early-history effect is.
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
            (
                pl.col("active_next_week").mean() * 100
            )
            .round(1)
            .alias("active_next_week_pct"),
            pl.col("active_weeks_next_4")
            .mean()
            .round(2)
            .alias("mean_active_weeks_next_4"),
            (
                (pl.col("active_weeks_next_4") >= 2).mean() * 100
            )
            .round(1)
            .alias("active_2plus_weeks_pct"),
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
    return (first_strong_by_user_history,)


@app.cell
def _(alt, first_strong_by_user_history):
    (
        alt.Chart(first_strong_by_user_history)
        .mark_bar()
        .encode(
            x=alt.X(
                "history_available:N",
                title="User history available",
                sort=[
                    "<1 month",
                    "1–3 months",
                    "3–6 months",
                    "6–12 months",
                    "12+ months",
                ],
            ),
            y=alt.Y(
                "mean_active_weeks_next_4:Q",
                title="Active weeks in following 4 weeks",
                scale=alt.Scale(domain=[0, 4]),
            ),
            detail="user_id:N",
            tooltip=[
                alt.Tooltip(
                    "history_available:N",
                    title="History available",
                ),
                alt.Tooltip("artists:Q", title="Artists"),
                alt.Tooltip(
                    "active_next_week_pct:Q",
                    title="Active next week (%)",
                    format=".1f",
                ),
                alt.Tooltip(
                    "mean_active_weeks_next_4:Q",
                    title="Active weeks next 4",
                    format=".2f",
                ),
                alt.Tooltip(
                    "active_2plus_weeks_pct:Q",
                    title="Active in 2+ weeks (%)",
                    format=".1f",
                ),
            ],
        )
        .properties(height=320)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First strong weeks near the start of this history are much more likely to keep appearing afterwards. In the first three months, artists are active in about **2.5 of the following four weeks** on average; after a year of observed history, that falls to about **1.1 weeks**.

    This is a strong sign of left-censoring near the beginning of the data: some early "first" strong weeks are probably artists that were already established before tracking started. We therefore need a user-history warm-up rule, but one user's history is not enough to choose its final length yet.
    """)
    return


if __name__ == "__main__":
    app.run()

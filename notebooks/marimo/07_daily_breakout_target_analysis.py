import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt

    import marimo as mo
    import polars as pl

    from sonora.breakout import (
        DEFAULT_HORIZON_DAYS,
        DEFAULT_RECENCY_DAYS,
        aggregate_daily_listening,
        build_listening_gaps,
        detect_first_breakouts,
        eligible_segments,
        historical_user_bounds,
        make_daily_targets,
    )
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return DEFAULT_HORIZON_DAYS, DEFAULT_RECENCY_DAYS, aggregate_daily_listening, historical_user_bounds, detect_first_breakouts, eligible_segments, build_listening_gaps, make_daily_targets, dt, mo, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Daily breakout target analysis

    A daily watchlist ranks artists by their chance of a first confirmed breakout within the next **60 days**. An artist is eligible after a listen in the previous **90 days**.

    An important period has an artist listening share in the user's **top 10%**, compared with up to **24 months of earlier history**. The comparisons below explain the period length, listening-day requirement, and daily target.
    """)
    return


@app.cell
def _(paths, pl):
    listening_events = pl.scan_parquet(paths.curated_listening_events)
    artists = pl.scan_parquet(paths.curated_artists).collect()
    return artists, listening_events


@app.cell
def _(aggregate_daily_listening, historical_user_bounds, listening_events, pl):
    daily_user_listening, daily_artist_listening = aggregate_daily_listening(listening_events)
    user_limits = historical_user_bounds(daily_user_listening)
    first_artist_listens = daily_artist_listening.group_by("user_id", "artist_id").agg(pl.col("date").min().alias("first_listen_date"))
    return daily_artist_listening, daily_user_listening, first_artist_listens, user_limits



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Breakout period definition

    ### Calendar-week boundary

    A calendar week can split a run of listening across two weeks. Starting the week on different days shows how much that changes the detected breakouts. Mondayâ€“Sunday weeks provide the comparison point.
    """)
    return


@app.cell
def _(artists, dt, pl):
    def build_fixed_week_events(
        daily_user,
        daily_artist,
        start_weekday=0,
    ):
        def _week_start(_date):
            return _date - dt.timedelta(
                days=(_date.weekday() - start_weekday) % 7
            )

        _weekly_user = (
            daily_user
            .with_columns(
                pl.col("date")
                .map_elements(_week_start, return_dtype=pl.Date)
                .alias("week")
            )
            .group_by("user_id", "week")
            .agg(pl.col("user_scrobbles").sum())
        )

        _weekly_artist = (
            daily_artist
            .with_columns(
                pl.col("date")
                .map_elements(_week_start, return_dtype=pl.Date)
                .alias("week")
            )
            .group_by("user_id", "artist_id", "week")
            .agg(
                pl.col("artist_scrobbles").sum(),
                pl.len().alias("active_days"),
            )
            .join(
                _weekly_user,
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

        _cutoff_rows = []
        for _user_id in _weekly_artist["user_id"].unique().to_list():
            _user_history = _weekly_artist.filter(
                pl.col("user_id") == _user_id
            )
            _first_week = _user_history["week"].min()

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
                        "top_10_cutoff": float(_past.quantile(0.90)),
                    }
                )

        _importance = (
            _weekly_artist
            .join(
                pl.DataFrame(_cutoff_rows),
                on=["user_id", "week"],
                how="inner",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("artist_share_pct")
                    >= pl.col("top_10_cutoff")
                ).alias("is_important")
            )
        )

        _last_week = (
            _importance
            .group_by("user_id")
            .agg(pl.col("week").max().alias("_last_week"))
        )

        _anchors = (
            _importance
            .filter(pl.col("is_important"))
            .join(
                _last_week,
                on="user_id",
                how="left",
                validate="m:1",
            )
            .filter(
                pl.col("week")
                <= pl.col("_last_week") - pl.duration(days=14)
            )
            .drop("_last_week")
        )

        _future = pl.concat(
            [
                _importance.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("week")
                        - pl.duration(days=7 * _weeks_ahead)
                    ).alias("week"),
                    pl.lit(_weeks_ahead).alias("_weeks_ahead"),
                    pl.col("is_important").alias("_future_important"),
                )
                for _weeks_ahead in (1, 2)
            ]
        )

        _candidates = (
            _anchors
            .join(
                _future,
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
                pl.col("_future_important")
                .fill_null(False)
                .sum()
                .alias("_extra_important_periods"),
                pl.col("_weeks_ahead")
                .filter(pl.col("_future_important"))
                .min()
                .alias("periods_until_second"),
            )
            .filter(pl.col("_extra_important_periods") >= 1)
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )

        _user_first_week = (
            _weekly_artist
            .group_by("user_id")
            .agg(pl.col("week").min().alias("_first_week"))
        )

        _history_with_start = _weekly_artist.join(
            _user_first_week,
            on="user_id",
            how="left",
            validate="m:1",
        )

        _warmup_cutoffs = (
            _history_with_start
            .filter(
                pl.col("week")
                < pl.col("_first_week") + pl.duration(days=365)
            )
            .group_by("user_id")
            .agg(
                pl.col("artist_share_pct")
                .quantile(0.90)
                .alias("_warmup_cutoff")
            )
        )

        _warmup_scored = (
            _history_with_start
            .join(
                _warmup_cutoffs,
                on="user_id",
                how="left",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("artist_share_pct")
                    >= pl.col("_warmup_cutoff")
                ).alias("_warmup_important")
            )
        )

        _warmup_anchors = _warmup_scored.filter(
            (
                pl.col("week")
                < pl.col("_first_week") + pl.duration(days=365)
            )
            & pl.col("_warmup_important")
        )

        _warmup_future = pl.concat(
            [
                _warmup_scored.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("week")
                        - pl.duration(days=7 * _weeks_ahead)
                    ).alias("week"),
                    pl.col("_warmup_important")
                    .alias("_future_warmup_important"),
                )
                for _weeks_ahead in (1, 2)
            ]
        )

        _warmup_established = (
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
                pl.col("_future_warmup_important")
                .fill_null(False)
                .sum()
                .alias("_extra_important_periods")
            )
            .filter(pl.col("_extra_important_periods") >= 1)
            .sort("week")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
            .select(
                "user_id",
                "artist_id",
                "canonical_name",
                pl.col("week").alias("established_period_start"),
            )
        )

        _events = (
            _candidates
            .join(
                _warmup_established.select("user_id", "artist_id"),
                on=["user_id", "artist_id"],
                how="anti",
            )
            .with_columns(
                (
                    pl.col("week")
                    + pl.duration(days=7) * pl.col("periods_until_second")
                ).alias("second_important_period_start")
            )
            .with_columns(
                (
                    pl.col("second_important_period_start")
                    + pl.duration(days=7)
                ).alias("confirmation_date")
            )
            .select(
                "user_id",
                "artist_id",
                "canonical_name",
                pl.col("week").alias("breakout_period_start"),
                (
                    pl.col("week") + pl.duration(days=6)
                ).alias("first_important_period_end"),
                "second_important_period_start",
                "confirmation_date",
                pl.col("artist_share_pct").alias("first_share_pct"),
                pl.col("active_days").alias("first_active_days"),
                "periods_until_second",
            )
            .sort("breakout_period_start")
        )

        _second_stats = (
            _weekly_artist
            .select(
                "user_id",
                "artist_id",
                pl.col("week").alias("second_important_period_start"),
                pl.col("active_days").alias("second_active_days"),
            )
        )

        _events = _events.join(
            _second_stats,
            on=[
                "user_id",
                "artist_id",
                "second_important_period_start",
            ],
            how="left",
            validate="1:1",
        )

        return _weekly_artist, _importance, _warmup_established, _events

    return (build_fixed_week_events,)


@app.cell
def _(build_fixed_week_events, daily_artist_listening, daily_user_listening):
    _, _, fixed_warmup_established, fixed_breakout_events = build_fixed_week_events(
        daily_user_listening,
        daily_artist_listening,
        start_weekday=0,
    )
    return fixed_breakout_events, fixed_warmup_established


@app.cell
def _(
    build_fixed_week_events,
    daily_artist_listening,
    daily_user_listening,
    fixed_breakout_events,
    pl,
):
    _weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _monday_keys = set(
        fixed_breakout_events.select("user_id", "artist_id").iter_rows()
    )

    _alignment_rows = []
    for _weekday, _name in enumerate(_weekday_names):
        _, _, _, _events = build_fixed_week_events(
            daily_user_listening,
            daily_artist_listening,
            start_weekday=_weekday,
        )
        _keys = set(_events.select("user_id", "artist_id").iter_rows())
        _shared = len(_monday_keys & _keys)
        _alignment_rows.append(
            {
                "week_starts": _name,
                "events": len(_keys),
                "monday_events_kept_pct": round(
                    _shared / len(_monday_keys) * 100,
                    1,
                ),
                "new_vs_monday": len(_keys - _monday_keys),
                "missing_vs_monday": len(_monday_keys - _keys),
            }
        )

    weekday_alignment_summary = pl.DataFrame(_alignment_rows)
    weekday_alignment_summary.rename({
        "week_starts": "Week starts on", "events": "Breakouts",
        "monday_events_kept_pct": "Monday-based breakouts kept (%)",
        "new_vs_monday": "Additional breakouts", "missing_vs_monday": "Monday-based breakouts lost",
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changing the start weekday changes which breakouts are found. A breakout rule should depend less on where the calendar week starts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rolling periods

    A rolling period can start on any day. The comparison covers calendar weeks, **rolling 7-day periods**, and **rolling 5-day periods**.

    For 7-day periods, a breakout needs two important periods whose starts are 7â€“14 days apart. They cannot overlap, and together they fit within 21 days.
    """)
    return





@app.cell
def _(
    detect_first_breakouts,
    daily_artist_listening,
    daily_user_listening,
    user_limits,
):
    rolling_7_breakout_events, rolling_7_warmup_established = detect_first_breakouts(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=7,
        min_active_days=1,
    )

    rolling_5_breakout_events, rolling_5_warmup_established = detect_first_breakouts(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=5,
        min_active_days=1,
    )
    return (
        rolling_5_breakout_events,
        rolling_5_warmup_established,
        rolling_7_breakout_events,
        rolling_7_warmup_established,
    )


@app.cell
def _(
    fixed_breakout_events,
    fixed_warmup_established,
    pl,
    rolling_5_breakout_events,
    rolling_5_warmup_established,
    rolling_7_breakout_events,
    rolling_7_warmup_established,
):
    _fixed_keys = set(
        fixed_breakout_events.select("user_id", "artist_id").iter_rows()
    )

    _method_inputs = [
        (
            "Fixed Mondayâ€“Sunday 7 days",
            fixed_breakout_events,
            fixed_warmup_established,
        ),
        (
            "Rolling 7 days",
            rolling_7_breakout_events,
            rolling_7_warmup_established,
        ),
        (
            "Rolling 5 days",
            rolling_5_breakout_events,
            rolling_5_warmup_established,
        ),
    ]

    _rows = []
    for _method, _events, _warmup in _method_inputs:
        _keys = set(_events.select("user_id", "artist_id").iter_rows())
        _shared = len(_fixed_keys & _keys)
        _rows.append(
            {
                "method": _method,
                "breakout_events": len(_keys),
                "fixed_events_kept_pct": round(
                    _shared / len(_fixed_keys) * 100,
                    1,
                ),
                "new_vs_fixed": len(_keys - _fixed_keys),
                "fixed_missing": len(_fixed_keys - _keys),
            }
        )

    window_method_summary = pl.DataFrame(_rows)

    window_method_summary.rename({
        "method": "Period type", "breakout_events": "Breakouts",
        "fixed_events_kept_pct": "Calendar-week breakouts kept (%)",
        "new_vs_fixed": "Additional breakouts", "fixed_missing": "Calendar-week breakouts lost",
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Rolling 7-day periods** keep a full week of listening together without tying it to a weekday. Five-day periods shorten that timescale without a clear benefit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Active-day requirement

    One intense listening day can dominate a whole period. Requiring listening on **1, 2, or 3 different days** shows the effect of asking for more repeated listening.
    """)
    return


@app.cell
def _(
    detect_first_breakouts,
    daily_artist_listening,
    daily_user_listening,
    user_limits,
):
    rolling_7_2day_breakout_events, rolling_7_2day_warmup_established = detect_first_breakouts(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=7,
        min_active_days=2,
    )

    rolling_7_3day_breakout_events, rolling_7_3day_warmup_established = detect_first_breakouts(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=7,
        min_active_days=3,
    )
    return (
        rolling_7_2day_breakout_events,
        rolling_7_2day_warmup_established,
        rolling_7_3day_breakout_events,
        rolling_7_3day_warmup_established,
    )


@app.cell
def _(
    pl,
    rolling_7_2day_breakout_events,
    rolling_7_2day_warmup_established,
    rolling_7_3day_breakout_events,
    rolling_7_3day_warmup_established,
    rolling_7_breakout_events,
    rolling_7_warmup_established,
):
    _base_keys = set(
        rolling_7_breakout_events.select("user_id", "artist_id").iter_rows()
    )

    _inputs = [
        (
            "1+ active day",
            rolling_7_breakout_events,
            rolling_7_warmup_established,
        ),
        (
            "2+ active days",
            rolling_7_2day_breakout_events,
            rolling_7_2day_warmup_established,
        ),
        (
            "3+ active days",
            rolling_7_3day_breakout_events,
            rolling_7_3day_warmup_established,
        ),
    ]

    _rows = []
    for _rule, _events, _warmup in _inputs:
        _keys = set(_events.select("user_id", "artist_id").iter_rows())
        _shared = len(_base_keys & _keys)
        _rows.append(
            {
                "active_days_required_per_period": _rule,
                "breakout_events": _events.height,
                "first_year_established": _warmup.height,
                "events_kept_vs_1day_pct": round(
                    _shared / len(_base_keys) * 100,
                    1,
                ),
                "median_active_days_across_two_periods": float(
                    _events.select(
                        (
                            pl.col("first_active_days")
                            + pl.col("second_active_days")
                        ).median()
                    ).item()
                ),
            }
        )

    active_day_rule_summary = pl.DataFrame(_rows)

    active_day_rule_summary.rename({
        "active_days_required_per_period": "Listening days required per period",
        "breakout_events": "Breakouts", "first_year_established": "Artists established in the first year",
        "events_kept_vs_1day_pct": "One-day-rule breakouts kept (%)",
        "median_active_days_across_two_periods": "Listening days across both periods (median)",
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each important period requires listening on **at least 2 days**. This excludes periods driven by a single day while allowing less frequent listening than the 3-day rule.

    ## 2. Breakout events

    A breakout is the first pair of important, non-overlapping **7-day periods**, each with listening on **at least 2 days**. It is confirmed the morning after the second period ends.
    """)
    return


@app.cell
def _(artists, rolling_7_2day_breakout_events):
    working_breakout_events = rolling_7_2day_breakout_events.join(artists, on="artist_id", how="left", validate="m:1")
    return (working_breakout_events,)


@app.cell
def _(pl, rolling_7_2day_warmup_established, working_breakout_events):
    working_breakout_summary = pl.DataFrame(
        {
            "measure": [
                "Breakout events",
                "Artists established in first year",
            ],
            "artists": [
                working_breakout_events.height,
                rolling_7_2day_warmup_established.height,
            ],
        }
    )

    working_breakout_summary.rename({"measure": "Measure", "artists": "Artists"})
    return


@app.cell
def _(pl, working_breakout_events):
    working_breakouts_by_year = (
        working_breakout_events
        .with_columns(
            pl.col("breakout_period_start").dt.year().alias("year")
        )
        .group_by("year")
        .agg(pl.len().alias("breakout_events"))
        .sort("year")
    )

    working_breakouts_by_year.rename({"year": "Year", "breakout_events": "Breakouts"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Event list

    Each artist appears once, with the start of their first breakout and the date it becomes confirmed.
    """)
    return


@app.cell
def _(pl, working_breakout_events):
    working_breakout_artist_list = (
        working_breakout_events
        .with_columns(
            pl.col("breakout_period_start").dt.year().alias("year")
        )
        .select(
            "year",
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_important_period_end",
            "second_important_period_start",
            "confirmation_date",
        )
        .sort("year", "breakout_period_start", "artist")
    )

    working_breakout_artist_list.rename({
        "year": "Year", "artist": "Artist", "breakout_period_start": "Breakout starts",
        "first_important_period_end": "First period ends",
        "second_important_period_start": "Second period starts", "confirmation_date": "Confirmed",
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confirmation near the end of the data

    A breakout is confirmed as soon as both important periods are complete. It needs no further listening or waiting time. A second period that is still unfinished cannot confirm a breakout.

    The latest confirmed breakout for each user shows the timing between the second period ending and confirmation.
    """)
    return


@app.cell
def _(pl, working_breakout_events):
    latest_confirmed_breakouts = (
        working_breakout_events
        .sort("confirmation_date", "artist_id")
        .group_by("user_id", maintain_order=True).last()
        .with_columns(
            (pl.col("second_important_period_start") + pl.duration(days=6)).alias("second_period_end")
        )
        .select("user_id", "canonical_name", "second_period_end", "confirmation_date")
    )
    latest_confirmed_breakouts.rename({
        "user_id": "User", "canonical_name": "Artist",
        "second_period_end": "Second period ends", "confirmation_date": "Confirmed",
    })
    return (latest_confirmed_breakouts,)


@app.cell
def _(DEFAULT_HORIZON_DAYS, DEFAULT_RECENCY_DAYS):
    selected_recency_days = DEFAULT_RECENCY_DAYS
    selected_horizon_days = DEFAULT_HORIZON_DAYS
    return selected_horizon_days, selected_recency_days


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Daily eligibility and prediction window

    One listen is enough to start scoring an artist the next morning. Each scoring date uses listening through the previous day.

    Artists leave when their latest listen falls outside the eligibility window. Another listen brings them back the next morning, with their earlier history still available. Their first confirmed breakout permanently removes them from eligibility, starting on the confirmation morning.

    The six combinations compare **30, 60, or 90 days of eligibility** with a prediction window of **30 or 60 days**. Predictions cover tomorrow through the final day of that window, inclusive. The results measure available chances to score a breakout, rather than model performance.

    ### Dates used

    The comparison starts after the first year plus 21 days. This allows enough time to identify artists already established near the start of this listening history. It is a limit for this historical analysis, not a required waiting period for every user.

    Dates use UTC. Listening is treated as complete through the day before the latest recorded listen. All six options use the same scoring dates, with a full 60 days afterward to observe the outcome. Breakouts in the timing comparison also have their full preceding 60 days inside those scoring dates.

    These tables use one user's history. Counts will depend on each user's listening habits.
    """)
    return


@app.cell
def _(pl, user_limits, working_breakout_events):
    comparison_limits = (
        user_limits.select(
            "user_id",
            "score_start",
            (pl.col("last_complete_date") + pl.duration(days=1)).alias("known_confirmation_through"),
        )
        .with_columns(
            (pl.col("known_confirmation_through") - pl.duration(days=60)).alias("score_end")
        )
    )
    comparison_events = (
        working_breakout_events.join(comparison_limits, on="user_id", validate="m:1")
        .filter(
            (pl.col("confirmation_date") >= pl.col("score_start") + pl.duration(days=60))
            & (pl.col("confirmation_date") <= pl.col("score_end") + pl.duration(days=1))
        )
    )
    comparison_dates = (
        comparison_limits.join(
            comparison_events.group_by("user_id").agg(pl.len().alias("breakouts")),
            on="user_id", how="left",
        )
        .with_columns(pl.col("breakouts").fill_null(0))
        .rename({
            "user_id": "User", "score_start": "First scoring day",
            "score_end": "Last scoring day", "known_confirmation_through": "Confirmations known through",
            "breakouts": "Breakouts in timing comparison",
        })
    )
    comparison_dates
    return comparison_events, comparison_limits


@app.cell
def _(build_listening_gaps, daily_artist_listening, pl, user_limits,
      rolling_7_2day_warmup_established, working_breakout_events):
    listening_gaps = build_listening_gaps(
        daily_artist_listening, working_breakout_events,
        rolling_7_2day_warmup_established, user_limits,
    ).with_columns((pl.col("known_confirmation_through") - pl.duration(days=60)).alias("score_end"))
    return (listening_gaps,)



@app.cell
def _(comparison_limits, dt, eligible_segments, listening_gaps, pl):
    eligible_by_recency = {}
    _pool_rows, _return_rows = [], []
    for _recency in [30, 60, 90]:
        _segments = eligible_segments(listening_gaps, _recency)
        eligible_by_recency[_recency] = _segments
        for _limit in comparison_limits.to_dicts():
            _user, _start, _end = _limit["user_id"], _limit["score_start"], _limit["score_end"]
            _days = max(0, (_end - _start).days + 1)
            _changes = [0] * (_days + 1)
            for _a, _b in _segments.filter(pl.col("user_id") == _user).select("eligible_start", "eligible_end").iter_rows():
                _changes[(_a - _start).days] += 1
                _changes[(_b - _start).days + 1] -= 1
            _count, _counts = 0, []
            for _change in _changes[:-1]:
                _count += _change
                _counts.append(_count)
            _counts = pl.Series(_counts, dtype=pl.Int64)
            _pool_rows.append({
                "user_id": _user, "recency_days": _recency,
                "median_artists": _counts.median(), "busy_day_artists": _counts.quantile(0.95),
                "total_scores": _counts.sum(), "scoring_days": _days,
            })
            _exits, _returns = 0, 0
            for _gap in listening_gaps.filter(pl.col("user_id") == _user).to_dicts():
                _expiry = _gap["date"] + dt.timedelta(days=_recency + 1)
                _next, _confirmation = _gap["next_listen"], _gap["confirmation_date"]
                if _start <= _expiry <= _end and (_next is None or _next >= _expiry) and (_confirmation is None or _confirmation > _expiry):
                    _exits += 1
                if _next is not None:
                    _back = _next + dt.timedelta(days=1)
                    if _expiry <= _next and _start <= _back <= _end and (_confirmation is None or _confirmation > _back):
                        _returns += 1
            _return_rows.append({"user_id": _user, "recency_days": _recency, "exits": _exits, "returns": _returns})
    pool_summary = pl.DataFrame(_pool_rows).join(pl.DataFrame(_return_rows), on=["user_id", "recency_days"])
    pool_summary.rename({
        "user_id": "User", "recency_days": "Keep for (days)", "median_artists": "Daily artists (median)",
        "busy_day_artists": "Daily artists (95th percentile)", "total_scores": "Total scores",
        "scoring_days": "Scoring days", "exits": "Times artists left", "returns": "Times artists came back",
    })
    return eligible_by_recency, pool_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All six options

    The pool counts above include every calendar day, even days with no listening. A busy day means the 95th percentile: only about 5% of days have more artists.

    "Before the start" means before the first important period. Scoring after that can still catch a breakout while it develops. The median early days includes breakouts with no early days. The median days to confirmation uses breakouts with at least one chance to score.

    "Confirmed within the horizon" is the share of all eligible scores followed by confirmation within 30 or 60 days. It measures how common the target is among eligible scores.
    """)
    return


@app.cell
def _(comparison_events, comparison_limits, eligible_by_recency, pl, pool_summary):
    _rows, _timing_rows = [], []
    event_opportunities = {}
    for _recency, _segments in eligible_by_recency.items():
        for _horizon in [30, 60]:
            _positive = (
                _segments.filter(pl.col("confirmation_date").is_not_null())
                .with_columns(
                    pl.max_horizontal(
                        "eligible_start", pl.col("confirmation_date") - pl.duration(days=_horizon),
                    ).alias("positive_start")
                )
                .filter(pl.col("positive_start") <= pl.col("eligible_end"))
                .with_columns(
                    ((pl.col("eligible_end") - pl.col("positive_start")).dt.total_days() + 1).alias("positive_days")
                )
            )
            _event_days = (
                _positive.join(
                    comparison_events.select("user_id", "artist_id", "breakout_period_start", "first_important_period_end"),
                    on=["user_id", "artist_id"], how="inner", validate="m:1",
                )
                .with_columns(
                    ((pl.min_horizontal("eligible_end", pl.col("breakout_period_start") - pl.duration(days=1))
                      - pl.col("positive_start")).dt.total_days() + 1).clip(lower_bound=0).alias("before_days"),
                    ((pl.min_horizontal("eligible_end", "first_important_period_end")
                      - pl.max_horizontal("positive_start", "breakout_period_start")).dt.total_days() + 1).clip(lower_bound=0).alias("during_days"),
                    ((pl.col("eligible_end") - pl.max_horizontal(
                        "positive_start", pl.col("first_important_period_end") + pl.duration(days=1),
                    )).dt.total_days() + 1).clip(lower_bound=0).alias("after_days"),
                )
                .group_by("user_id", "artist_id")
                .agg(pl.col("positive_start").min(), pl.col("positive_days", "before_days", "during_days", "after_days").sum())
            )
            _opportunities = (
                comparison_events.join(_event_days, on=["user_id", "artist_id"], how="left", validate="1:1")
                .with_columns(pl.col("positive_days", "before_days", "during_days", "after_days").fill_null(0))
                .with_columns((pl.col("confirmation_date") - pl.col("positive_start")).dt.total_days().alias("lead_days"))
            )
            event_opportunities[(_recency, _horizon)] = _opportunities
            for _user in comparison_limits["user_id"]:
                _user_events = _opportunities.filter(pl.col("user_id") == _user)
                _total = pool_summary.filter((pl.col("user_id") == _user) & (pl.col("recency_days") == _recency))["total_scores"].item()
                _positive_days = _positive.filter(pl.col("user_id") == _user)["positive_days"].sum()
                _rows.append({
                    "user_id": _user, "recency_days": _recency, "horizon_days": _horizon,
                    "breakouts_compared": _user_events.height,
                    "breakouts_with_a_chance": _user_events.filter(pl.col("positive_days") > 0).height,
                    "breakouts_with_early_chance": _user_events.filter(pl.col("before_days") > 0).height,
                    "median_early_days": _user_events["before_days"].median(),
                    "median_lead_days": _user_events["lead_days"].median(),
                    "positive_scores": _positive_days,
                    "positive_pct": round(_positive_days / _total * 100, 2) if _total else None,
                })
                _total_event_days = _user_events["positive_days"].sum()
                _timing_rows.append({
                    "user_id": _user, "recency_days": _recency, "horizon_days": _horizon,
                    **{_name: round(_user_events[_column].sum() / _total_event_days * 100, 1) if _total_event_days else None
                       for _name, _column in [("Before the start (%)", "before_days"), ("During the first period (%)", "during_days"), ("After the first period (%)", "after_days")]},
                })
    six_option_summary = pl.DataFrame(_rows)
    scoring_timing = pl.DataFrame(_timing_rows)
    six_option_summary.drop("positive_scores").rename({
        "user_id": "User", "recency_days": "Keep for (days)", "horizon_days": "Predict ahead (days)",
        "breakouts_compared": "Breakouts compared", "breakouts_with_a_chance": "Could score before confirmation",
        "breakouts_with_early_chance": "Could score before the start", "median_early_days": "Early days (median)",
        "median_lead_days": "Days to confirmation (median)", "positive_pct": "Confirmed within the horizon (%)",
    })
    return event_opportunities, scoring_timing, six_option_summary


@app.cell
def _(scoring_timing):
    scoring_timing.rename({"user_id": "User", "recency_days": "Keep for (days)", "horizon_days": "Predict ahead (days)"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listening during the extra month

    Each count is a time the longer rule keeps an artist after the shorter rule would drop them. A return means another listen before the longer rule expires or the breakout is confirmed.

    Separate gaps can count the same artist more than once. The whole extra month must fit inside the scoring dates above. An artist with no listen during that month may still return later.
    """)
    return


@app.cell
def _(comparison_limits, dt, listening_gaps, pl):
    _rows = []
    for _user in comparison_limits["user_id"]:
        for _short, _long in [(30, 60), (60, 90)]:
            _gaps, _returns = 0, 0
            for _gap in listening_gaps.filter(pl.col("user_id") == _user).to_dicts():
                _start = _gap["date"] + dt.timedelta(days=_short + 1)
                _end = _gap["date"] + dt.timedelta(days=_long)
                _next, _confirmation = _gap["next_listen"], _gap["confirmation_date"]
                if _start < _gap["score_start"] or _end > _gap["score_end"] or (_next is not None and _next < _start) or (_confirmation is not None and _confirmation <= _start):
                    continue
                _gaps += 1
                _returns += _next is not None and _next <= _end and (_confirmation is None or _next < _confirmation)
            _rows.append({
                "User": _user, "Days compared": f"{_short} to {_long}", "Times an artist stayed longer": _gaps,
                "Listened again in the extra month": _returns,
                "Listened again (%)": round(_returns / _gaps * 100, 1) if _gaps else None,
            })
    extra_month_summary = pl.DataFrame(_rows)
    extra_month_summary
    return (extra_month_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Artist timelines

    The timelines show a first discovery, a return after a long gap, and an artist with no listening for 90 days. Each table includes the key dates and earlier listens for context. A listen only changes eligibility the next morning.

    The first two examples are the earliest matching breakouts in the comparison. The inactive example is the earliest 90-day gap for an artist with no confirmed breakout in the available history.
    """)
    return


@app.cell
def _(artists, comparison_events, daily_artist_listening, dt, first_artist_listens, listening_gaps, mo, pl):
    from bisect import bisect_left as _bisect_left

    _histories = {
        (_r["user_id"], _r["artist_id"]): _r["date"]
        for _r in daily_artist_listening.group_by("user_id", "artist_id").agg(pl.col("date").sort()).to_dicts()
    }
    _choices = []
    for _user in listening_gaps["user_id"].unique().sort():
        _events = comparison_events.filter(pl.col("user_id") == _user).join(
            first_artist_listens, on=["user_id", "artist_id"], validate="1:1",
        ).sort("confirmation_date", "artist_id")
        _discovery = _events.filter(pl.col("first_listen_date") + pl.duration(days=1) >= pl.col("breakout_period_start"))
        if _discovery.height:
            _row = _discovery.row(0, named=True)
            _choices.append(("First discovery", _row, _row["first_listen_date"]))
        _return = (
            _events.join(listening_gaps.select("user_id", "artist_id", "date", "next_listen"), on=["user_id", "artist_id"])
            .filter(
                ((pl.col("breakout_period_start") - pl.duration(days=1) - pl.col("date")).dt.total_days().is_between(61, 90))
                & (pl.col("next_listen") >= pl.col("breakout_period_start") - pl.duration(days=1))
            ).sort("confirmation_date", "artist_id")
        )
        if _return.height:
            _row = _return.row(0, named=True)
            _choices.append(("Back after a long gap", _row, _row["date"]))
        _inactive = (
            listening_gaps.filter(
                (pl.col("user_id") == _user) & pl.col("confirmation_date").is_null()
                & (pl.col("date") + pl.duration(days=31) >= pl.col("score_start"))
                & (pl.col("date") + pl.duration(days=91) <= pl.col("score_end"))
                & (pl.col("next_listen").is_null() | (pl.col("next_listen") > pl.col("date") + pl.duration(days=90)))
            ).join(artists, on="artist_id", validate="m:1").sort("date", "artist_id")
        )
        if _inactive.height:
            _row = _inactive.row(0, named=True)
            _choices.append(("No listening for 90 days", _row, _row["date"]))
    timeline_examples = []
    _views = []
    for _kind, _row, _last in _choices:
        _history = _histories[(_row["user_id"], _row["artist_id"])]
        _confirmation = _row["confirmation_date"]
        _end = _confirmation or _last + dt.timedelta(days=91)
        _notes = {}

        def _note(_date, _text, _end=_end, _notes=_notes):
            if _date <= _end:
                _notes.setdefault(_date, []).append(_text)

        _note(_last, "Listen")
        _note(_last + dt.timedelta(days=1), "Listen included in scoring")
        _next_index = _bisect_left(_history, _last) + 1
        _next = _history[_next_index] if _next_index < len(_history) else None
        if _next is not None:
            _note(_next, "Next listen")
            _note(_next + dt.timedelta(days=1), "Listen included in scoring")
        for _recency in [30, 60, 90]:
            _expiry = _last + dt.timedelta(days=_recency + 1)
            if _next is None or _expiry <= _next:
                _note(_expiry, f"Leaves with {_recency} days")
        if _confirmation is not None:
            _note(_row["breakout_period_start"], "Breakout starts")
            _note(_confirmation, "Breakout confirmed")
        _rows = []
        for _date in sorted(_notes):
            _position = _bisect_left(_history, _date) - 1
            _age = (_date - _history[_position]).days if _position >= 0 else None
            _rows.append({
                "Date": _date, "What happened": "; ".join(_notes[_date]),
                **{f"Keep for {_recency} days": "Yes" if _age is not None and _age <= _recency and (_confirmation is None or _date < _confirmation) else "No" for _recency in [30, 60, 90]},
            })
        _table = pl.DataFrame(_rows)
        timeline_examples.append({"kind": _kind, "user_id": _row["user_id"], "artist": _row["canonical_name"], "table": _table})
        _views.extend([mo.md(f"**{_kind}: {_row['canonical_name']}** ({_row['user_id']})"), mo.ui.table(_table, selection=None)])
    mo.vstack(_views)
    return (timeline_examples,)


@app.cell(hide_code=True)
def _(
    comparison_limits, eligible_by_recency, event_opportunities,
    latest_confirmed_breakouts, pl, selected_horizon_days, selected_recency_days,
    six_option_summary,
):
    assert (selected_recency_days, selected_horizon_days) in event_opportunities
    assert six_option_summary.filter(
        (pl.col("recency_days") == selected_recency_days)
        & (pl.col("horizon_days") == selected_horizon_days)
    ).height == comparison_limits.height
    assert latest_confirmed_breakouts.select(
        (pl.col("second_period_end") + pl.duration(days=1) == pl.col("confirmation_date")).all()
    ).item()
    assert comparison_limits.select(
        (pl.col("score_end") + pl.duration(days=60) <= pl.col("known_confirmation_through")).all()
    ).item()
    for _segments in eligible_by_recency.values():
        assert _segments.filter(
            (pl.col("eligible_start") <= pl.col("date"))
            | (pl.col("eligible_end") >= pl.col("confirmation_date"))
        ).is_empty()
    for (_recency, _horizon), _events in event_opportunities.items():
        assert _events.select(
            (pl.col("positive_days") == pl.col("before_days") + pl.col("during_days") + pl.col("after_days")).all()
        ).item()
        _larger_options = [(_r, _horizon) for _r in [30, 60, 90] if _r > _recency]
        if _horizon == 30:
            _larger_options.append((_recency, 60))
        for _option in _larger_options:
            _joined = _events.join(
                event_opportunities[_option], on=["user_id", "artist_id"], suffix="_larger", validate="1:1",
            )
            assert _joined.select(
                ((pl.col("positive_days_larger") >= pl.col("positive_days"))
                 & (pl.col("before_days_larger") >= pl.col("before_days"))).all()
            ).item()
    return


@app.cell(hide_code=True)
def _(mo, selected_horizon_days, selected_recency_days):
    mo.md(f"""
    ## 4. Selected target

    **Eligibility: {selected_recency_days} days. Prediction window: {selected_horizon_days} days.**

    An artist is eligible if they have at least one listen in the previous {selected_recency_days} days, ending yesterday, and their first breakout is not yet confirmed. The target is confirmation from tomorrow through {selected_horizon_days} days ahead, inclusive.

    A longer eligibility window keeps artists available through gaps in listening and leaves more of the filtering to the model. The {selected_horizon_days}-day prediction window gives more chances before breakout behaviour starts. The tradeoff is more scoring while artists are inactive and a prediction that covers the next two months.

    Artists can return after inactivity with their earlier history intact. First breakout confirmation ends eligibility permanently. The same rule applies to every user.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Daily target dataset

    Each row is an eligible artist on a scoring date. A target of **true** means the first breakout is confirmed within the following 60 days. **False** means those 60 days are complete without a confirmation. An empty target means the outcome is still unknown.

    Recent rows stay in the dataset. A confirmed outcome can already be true even when the full 60 days are not available. Training uses only rows with a full 60 days of follow-up.
    """)
    return





@app.cell
def _(
    daily_artist_listening, make_daily_targets, pl,
    rolling_7_2day_warmup_established, selected_horizon_days,
    selected_recency_days, user_limits, working_breakout_events,
):
    dataset_bounds = user_limits
    daily_targets = make_daily_targets(
        daily_artist_listening, working_breakout_events, rolling_7_2day_warmup_established,
        dataset_bounds, recency_days=selected_recency_days, horizon_days=selected_horizon_days,
    ).collect(engine="streaming")
    return daily_targets, dataset_bounds


@app.cell
def _(daily_targets, pl):
    _rows = []
    for _user in daily_targets["user_id"].unique().sort():
        _user_rows = daily_targets.filter(pl.col("user_id") == _user)
        for _label, _rows_to_count in [
            ("All eligible days", _user_rows),
            ("Full 60-day follow-up", _user_rows.filter(pl.col("has_full_horizon"))),
            ("Recent dates", _user_rows.filter(~pl.col("has_full_horizon"))),
        ]:
            _rows.append({
                "User": _user, "Rows": _label, "Total": _rows_to_count.height,
                "Yes": _rows_to_count["target_60d"].sum(),
                "No": _rows_to_count.filter(pl.col("target_60d").eq(False)).height,
                "Unknown": _rows_to_count["target_60d"].null_count(),
            })
    daily_target_summary = pl.DataFrame(_rows)
    daily_target_summary
    return (daily_target_summary,)


@app.cell
def _(artists, daily_targets, pl, timeline_examples):
    _names = [example["artist"] for example in timeline_examples]
    _examples = daily_targets.join(artists.select("artist_id", "canonical_name"), on="artist_id", validate="m:1").filter(pl.col("canonical_name").is_in(_names))
    daily_target_examples = (
        pl.concat([
            _examples.group_by("user_id", "artist_id", maintain_order=True).head(3),
            _examples.group_by("user_id", "artist_id", maintain_order=True).tail(3),
        ]).unique().sort("user_id", "canonical_name", "scoring_date")
        .select("user_id", "canonical_name", "scoring_date", "last_listen_date", "target_60d", "has_full_horizon")
    )
    daily_target_examples.rename({
        "user_id": "User", "canonical_name": "Artist", "scoring_date": "Scoring date",
        "last_listen_date": "Latest known listen", "target_60d": "Confirmed within 60 days",
        "has_full_horizon": "Full follow-up",
    })
    return


if __name__ == "__main__":
    app.run()

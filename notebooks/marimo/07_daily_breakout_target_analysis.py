import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    from collections import deque

    import marimo as mo
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return deque, dt, mo, paths, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Daily breakout target analysis

    The weekly breakout rule gave us a sensible first event definition. Before turning it into daily labels, we need to check two things:

    1. whether fixed Monday–Sunday weeks are affecting which breakouts we find;
    2. what the daily prediction should actually predict.

    We keep the settled ideas of **user-relative importance**, the **top 10% level**, the **24-month comparison history**, and **two important periods within three periods**.
    """)
    return


@app.cell
def _(paths, pl):
    listening_events = pl.scan_parquet(paths.curated_listening_events)
    artists = pl.scan_parquet(paths.curated_artists).collect()
    return artists, listening_events


@app.cell
def _(listening_events, pl):
    daily_user_listening = (
        listening_events
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "date")
        .agg(pl.len().alias("user_scrobbles"))
        .collect()
    )

    daily_artist_listening = (
        listening_events
        .with_columns(pl.col("listened_at").dt.date().alias("date"))
        .group_by("user_id", "artist_id", "date")
        .agg(pl.len().alias("artist_scrobbles"))
        .collect()
    )

    user_limits = (
        daily_user_listening
        .group_by("user_id")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_observed_date"),
        )
        .with_columns(
            (
                pl.col("last_observed_date") - pl.duration(days=1)
            ).alias("last_complete_date")
        )
    )

    first_artist_listens = (
        daily_artist_listening
        .group_by("user_id", "artist_id")
        .agg(pl.col("date").min().alias("first_listen_date"))
    )

    return (
        daily_artist_listening,
        daily_user_listening,
        first_artist_listens,
        user_limits,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Is the Monday–Sunday week boundary stable?

    The current rule groups listening into Monday–Sunday weeks. That boundary is arbitrary, so we first rebuild the same rule with each possible weekday as the start of the week.

    Nothing else changes in this check.
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
def _(
    build_fixed_week_events,
    daily_artist_listening,
    daily_user_listening,
    pl,
):
    fixed_weekly_history, fixed_weekly_importance, fixed_warmup_established, fixed_breakout_events = build_fixed_week_events(
        daily_user_listening,
        daily_artist_listening,
        start_weekday=0,
    )

    fixed_reference_summary = (
        fixed_breakout_events
        .with_columns(
            pl.col("breakout_period_start").dt.year().alias("year")
        )
        .group_by("year")
        .agg(pl.len().alias("breakout_events"))
        .sort("year")
    )

    fixed_reference_summary
    return (
        fixed_breakout_events,
        fixed_reference_summary,
        fixed_warmup_established,
        fixed_weekly_history,
        fixed_weekly_importance,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Monday-based reconstruction matches notebook 06. We can therefore use it as the reference for the boundary check.
    """)
    return


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
    _alignment_event_sets = {}
    for _weekday, _name in enumerate(_weekday_names):
        _, _, _, _events = build_fixed_week_events(
            daily_user_listening,
            daily_artist_listening,
            start_weekday=_weekday,
        )
        _keys = set(_events.select("user_id", "artist_id").iter_rows())
        _alignment_event_sets[_name] = _keys
        _shared = len(_monday_keys & _keys)
        _alignment_rows.append(
            {
                "week_starts": _name,
                "events": len(_keys),
                "shared_with_monday": _shared,
                "monday_events_kept_pct": round(
                    _shared / len(_monday_keys) * 100,
                    1,
                ),
                "new_vs_monday": len(_keys - _monday_keys),
                "missing_vs_monday": len(_monday_keys - _keys),
            }
        )

    weekday_alignment_summary = pl.DataFrame(_alignment_rows)
    weekday_alignment_events = _alignment_event_sets

    weekday_alignment_summary
    return weekday_alignment_events, weekday_alignment_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The event list changes noticeably when the week starts on a different day. Monday is therefore not a neutral boundary.

    A rolling window is worth testing because it keeps the same listening period without choosing one weekday as special.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fixed weeks vs rolling periods

    We compare:

    - **fixed 7-day weeks** — the current Monday–Sunday rule;
    - **rolling 7 days** — a 7-day period ending on every day;
    - **rolling 5 days** — a shorter sensitivity check.

    For rolling periods, the two important periods must not overlap. The second can start immediately after the first or after one period-sized gap. This keeps the same basic repetition idea as the weekly rule.
    """)
    return


@app.cell
def _(artists, deque, dt, pl):
    def build_rolling_events(
        daily_user,
        daily_artist,
        user_limits_data,
        period_days,
        min_active_days=1,
    ):
        _expanded_user = pl.concat(
            [
                daily_user.select(
                    "user_id",
                    (
                        pl.col("date") + pl.duration(days=_offset)
                    ).alias("window_end"),
                    "user_scrobbles",
                )
                for _offset in range(period_days)
            ]
        )

        _rolling_user = (
            _expanded_user
            .group_by("user_id", "window_end")
            .agg(pl.col("user_scrobbles").sum())
            .join(
                user_limits_data,
                on="user_id",
                how="left",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("window_end")
                    - pl.duration(days=period_days - 1)
                ).alias("window_start")
            )
            .filter(
                (pl.col("window_start") >= pl.col("first_date"))
                & (
                    pl.col("window_end")
                    <= pl.col("last_complete_date")
                )
            )
            .drop("first_date", "last_observed_date", "last_complete_date")
        )

        _expanded_artist = pl.concat(
            [
                daily_artist.select(
                    "user_id",
                    "artist_id",
                    "date",
                    (
                        pl.col("date") + pl.duration(days=_offset)
                    ).alias("window_end"),
                    "artist_scrobbles",
                )
                for _offset in range(period_days)
            ]
        )

        _windows = (
            _expanded_artist
            .group_by("user_id", "artist_id", "window_end")
            .agg(
                pl.col("artist_scrobbles").sum(),
                pl.col("date").n_unique().alias("active_days"),
            )
            .join(
                _rolling_user,
                on=["user_id", "window_end"],
                how="inner",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("window_end")
                    - pl.duration(days=period_days - 1)
                ).alias("window_start"),
                (
                    pl.col("artist_scrobbles")
                    / pl.col("user_scrobbles")
                    * 100
                )
                .round(2)
                .alias("artist_share_pct"),
            )
            .join(artists, on="artist_id", how="left", validate="m:1")
            .join(
                user_limits_data.select("user_id", "first_date"),
                on="user_id",
                how="left",
                validate="m:1",
            )
        )

        _cutoff_rows = []
        for _user_id in _windows["user_id"].unique().to_list():
            _user_windows = _windows.filter(
                pl.col("user_id") == _user_id
            )
            _first_date = _user_windows["first_date"][0]
            _candidate_min_end = _first_date + dt.timedelta(
                days=365 + period_days - 1
            )

            _by_date = (
                _user_windows
                .select(
                    "window_end",
                    (
                        pl.col("artist_share_pct") * 100
                    )
                    .round(0)
                    .cast(pl.Int16)
                    .alias("_share_cents"),
                )
                .group_by("window_end")
                .agg(pl.col("_share_cents").alias("_shares"))
                .sort("window_end")
            )

            _date_rows = list(_by_date.iter_rows(named=True))
            _hist = [0] * 10001
            _active_dates = deque()
            _total = 0
            _add_index = 0

            for _target_date in [
                _row["window_end"]
                for _row in _date_rows
                if _row["window_end"] >= _candidate_min_end
            ]:
                while (
                    _add_index < len(_date_rows)
                    and _date_rows[_add_index]["window_end"] < _target_date
                ):
                    _row = _date_rows[_add_index]
                    _counts = {}
                    for _value in _row["_shares"]:
                        _value = int(_value)
                        _counts[_value] = _counts.get(_value, 0) + 1
                    for _value, _count in _counts.items():
                        _hist[_value] += _count
                    _row_count = len(_row["_shares"])
                    _total += _row_count
                    _active_dates.append(
                        (_row["window_end"], _counts, _row_count)
                    )
                    _add_index += 1

                _lower_date = _target_date - dt.timedelta(days=730)
                while (
                    _active_dates
                    and _active_dates[0][0] < _lower_date
                ):
                    _, _counts, _row_count = _active_dates.popleft()
                    for _value, _count in _counts.items():
                        _hist[_value] -= _count
                    _total -= _row_count

                _rank = round(0.90 * (_total - 1))
                _seen = 0
                _cutoff_cents = 0
                for _value, _count in enumerate(_hist):
                    _seen += _count
                    if _seen > _rank:
                        _cutoff_cents = _value
                        break

                _cutoff_rows.append(
                    {
                        "user_id": _user_id,
                        "window_end": _target_date,
                        "top_10_cutoff": _cutoff_cents / 100,
                    }
                )

        _importance = (
            _windows
            .join(
                pl.DataFrame(_cutoff_rows),
                on=["user_id", "window_end"],
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

        _important = _importance.filter(
            pl.col("is_important")
            & (pl.col("active_days") >= min_active_days)
        )

        _anchors = (
            _important
            .join(
                user_limits_data.select(
                    "user_id",
                    "last_complete_date",
                ),
                on="user_id",
                how="left",
                validate="m:1",
            )
            .filter(
                pl.col("window_end")
                <= pl.col("last_complete_date")
                - pl.duration(days=2 * period_days)
            )
            .drop("last_complete_date")
        )

        _future = pl.concat(
            [
                _important.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("window_end")
                        - pl.duration(days=_days_ahead)
                    ).alias("window_end"),
                    pl.col("window_end").alias("_second_window_end"),
                )
                for _days_ahead in range(period_days, 2 * period_days + 1)
            ]
        )

        _candidates = (
            _anchors
            .join(
                _future,
                on=["user_id", "artist_id", "window_end"],
                how="inner",
                validate="1:m",
            )
            .sort("window_end", "_second_window_end")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
        )

        _warmup_cutoffs = (
            _windows
            .filter(
                pl.col("window_start")
                < pl.col("first_date") + pl.duration(days=365)
            )
            .group_by("user_id")
            .agg(
                pl.col("artist_share_pct")
                .quantile(0.90)
                .alias("_warmup_cutoff")
            )
        )

        _warmup_scored = (
            _windows
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

        _warmup_important = _warmup_scored.filter(
            pl.col("_warmup_important")
            & (pl.col("active_days") >= min_active_days)
        )

        _warmup_anchors = _warmup_important.filter(
            pl.col("window_start")
            < pl.col("first_date") + pl.duration(days=365)
        )

        _warmup_future = pl.concat(
            [
                _warmup_important.select(
                    "user_id",
                    "artist_id",
                    (
                        pl.col("window_end")
                        - pl.duration(days=_days_ahead)
                    ).alias("window_end"),
                    pl.col("window_end").alias("_second_window_end"),
                )
                for _days_ahead in range(period_days, 2 * period_days + 1)
            ]
        )

        _warmup_established = (
            _warmup_anchors
            .join(
                _warmup_future,
                on=["user_id", "artist_id", "window_end"],
                how="inner",
                validate="1:m",
            )
            .sort("window_end", "_second_window_end")
            .group_by("user_id", "artist_id", maintain_order=True)
            .first()
            .select(
                "user_id",
                "artist_id",
                "canonical_name",
                pl.col("window_start").alias("established_period_start"),
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
                pl.col("window_start").alias("breakout_period_start"),
                pl.col("window_end").alias("first_important_period_end"),
                (
                    pl.col("_second_window_end")
                    - pl.duration(days=period_days - 1)
                ).alias("second_important_period_start"),
                (
                    pl.col("_second_window_end") + pl.duration(days=1)
                ).alias("confirmation_date"),
                pl.col("artist_share_pct").alias("first_share_pct"),
                pl.col("active_days").alias("first_active_days"),
            )
        )

        _second_stats = _windows.select(
            "user_id",
            "artist_id",
            pl.col("window_end").alias("_second_window_end"),
            pl.col("active_days").alias("second_active_days"),
        )

        _events = (
            _events
            .join(
                _second_stats,
                on=["user_id", "artist_id", "_second_window_end"],
                how="left",
                validate="1:1",
            )
            .select(
                "user_id",
                "artist_id",
                "canonical_name",
                "breakout_period_start",
                "first_important_period_end",
                "second_important_period_start",
                "confirmation_date",
                "first_share_pct",
                "first_active_days",
                "second_active_days",
            )
            .sort("breakout_period_start")
        )

        return _windows, _importance, _warmup_established, _events

    return (build_rolling_events,)


@app.cell
def _(
    build_rolling_events,
    daily_artist_listening,
    daily_user_listening,
    user_limits,
):
    rolling_7_windows, rolling_7_importance, rolling_7_warmup_established, rolling_7_breakout_events = build_rolling_events(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=7,
    )

    rolling_5_windows, rolling_5_importance, rolling_5_warmup_established, rolling_5_breakout_events = build_rolling_events(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=5,
    )

    return (
        rolling_5_breakout_events,
        rolling_5_importance,
        rolling_5_warmup_established,
        rolling_5_windows,
        rolling_7_breakout_events,
        rolling_7_importance,
        rolling_7_warmup_established,
        rolling_7_windows,
    )


@app.cell
def _(
    fixed_breakout_events,
    fixed_warmup_established,
    fixed_weekly_importance,
    pl,
    rolling_5_breakout_events,
    rolling_5_importance,
    rolling_5_warmup_established,
    rolling_7_breakout_events,
    rolling_7_importance,
    rolling_7_warmup_established,
):
    _fixed_keys = set(
        fixed_breakout_events.select("user_id", "artist_id").iter_rows()
    )

    _method_inputs = [
        (
            "Fixed Monday–Sunday 7 days",
            fixed_breakout_events,
            fixed_warmup_established,
            fixed_weekly_importance,
        ),
        (
            "Rolling 7 days",
            rolling_7_breakout_events,
            rolling_7_warmup_established,
            rolling_7_importance,
        ),
        (
            "Rolling 5 days",
            rolling_5_breakout_events,
            rolling_5_warmup_established,
            rolling_5_importance,
        ),
    ]

    _rows = []
    for _method, _events, _warmup, _importance in _method_inputs:
        _keys = set(_events.select("user_id", "artist_id").iter_rows())
        _shared = len(_fixed_keys & _keys)
        _rows.append(
            {
                "method": _method,
                "breakout_events": len(_keys),
                "first_year_established": _warmup.height,
                "shared_with_fixed": _shared,
                "fixed_events_kept_pct": round(
                    _shared / len(_fixed_keys) * 100,
                    1,
                ),
                "new_vs_fixed": len(_keys - _fixed_keys),
                "fixed_missing": len(_fixed_keys - _keys),
                "median_top_10_cutoff_pct": round(
                    float(_importance["top_10_cutoff"].median()),
                    2,
                ),
            }
        )

    window_method_summary = pl.DataFrame(_rows)

    window_method_summary
    return (window_method_summary,)


@app.cell
def _(
    fixed_breakout_events,
    pl,
    rolling_7_breakout_events,
    weekday_alignment_events,
):
    _all_alignment_keys = set().union(*weekday_alignment_events.values())
    _all_seven_keys = set.intersection(*weekday_alignment_events.values())
    _rolling_keys = set(
        rolling_7_breakout_events.select("user_id", "artist_id").iter_rows()
    )

    rolling_7_alignment_support = pl.DataFrame(
        {
            "measure": [
                "Rolling 7-day events",
                "Also found by at least one fixed weekday alignment",
                "Not found by any fixed weekday alignment",
                "Events found under every fixed weekday alignment",
                "Stable-across-all events also found by rolling 7 days",
            ],
            "events": [
                len(_rolling_keys),
                len(_rolling_keys & _all_alignment_keys),
                len(_rolling_keys - _all_alignment_keys),
                len(_all_seven_keys),
                len(_rolling_keys & _all_seven_keys),
            ],
        }
    )

    _fixed_timing = fixed_breakout_events.select(
        "user_id",
        "artist_id",
        pl.col("canonical_name").alias("artist"),
        pl.col("breakout_period_start").alias("fixed_start"),
    )

    rolling_7_timing_changes = (
        rolling_7_breakout_events
        .select(
            "user_id",
            "artist_id",
            pl.col("breakout_period_start").alias("rolling_start"),
        )
        .join(
            _fixed_timing,
            on=["user_id", "artist_id"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("rolling_start") - pl.col("fixed_start")
            )
            .dt.total_days()
            .alias("date_change_days")
        )
        .with_columns(
            pl.col("date_change_days").abs().alias("abs_date_change_days")
        )
        .sort("abs_date_change_days", descending=True)
    )

    rolling_7_alignment_support
    return rolling_7_alignment_support, rolling_7_timing_changes


@app.cell
def _(pl, rolling_7_timing_changes):
    _total = rolling_7_timing_changes.height
    rolling_7_timing_stability = pl.DataFrame(
        {
            "date difference for shared artists": [
                "Within 7 days",
                "8–30 days",
                "More than 30 days",
            ],
            "artists": [
                rolling_7_timing_changes.filter(
                    pl.col("abs_date_change_days") <= 7
                ).height,
                rolling_7_timing_changes.filter(
                    pl.col("abs_date_change_days").is_between(8, 30)
                ).height,
                rolling_7_timing_changes.filter(
                    pl.col("abs_date_change_days") > 30
                ).height,
            ],
        }
    ).with_columns(
        (
            pl.col("artists") / _total * 100
        ).round(1).alias("artist_pct")
    )

    rolling_7_timing_stability
    return (rolling_7_timing_stability,)


@app.cell
def _(
    fixed_breakout_events,
    pl,
    rolling_5_breakout_events,
    rolling_7_breakout_events,
):
    _period_inputs = [
        ("Fixed 7-day weeks", fixed_breakout_events),
        ("Rolling 7 days", rolling_7_breakout_events),
        ("Rolling 5 days", rolling_5_breakout_events),
    ]

    _rows = []
    for _method, _events in _period_inputs:
        _two_single_day = _events.filter(
            (pl.col("first_active_days") == 1)
            & (pl.col("second_active_days") == 1)
        ).height
        _rows.append(
            {
                "method": _method,
                "events": _events.height,
                "median_active_days_across_two_periods": float(
                    _events.select(
                        (
                            pl.col("first_active_days")
                            + pl.col("second_active_days")
                        ).median()
                    ).item()
                ),
                "one_active_day_in_each_period": _two_single_day,
                "one_day_each_pct": round(
                    _two_single_day / _events.height * 100,
                    1,
                ),
            }
        )

    period_concentration_summary = pl.DataFrame(_rows)

    period_concentration_summary
    return (period_concentration_summary,)


@app.cell
def _(
    fixed_breakout_events,
    pl,
    rolling_7_breakout_events,
    rolling_7_timing_changes,
):
    rolling_7_added_artists = (
        rolling_7_breakout_events
        .join(
            fixed_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_important_period_end",
            "confirmation_date",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "artist")
    )

    rolling_7_missing_fixed_artists = (
        fixed_breakout_events
        .join(
            rolling_7_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
        )
        .sort("breakout_period_start", "artist")
    )

    rolling_7_material_date_changes = (
        rolling_7_timing_changes
        .filter(pl.col("abs_date_change_days") > 30)
        .select(
            "artist",
            "fixed_start",
            "rolling_start",
            "date_change_days",
        )
        .sort("date_change_days")
    )

    return (
        rolling_7_added_artists,
        rolling_7_material_date_changes,
        rolling_7_missing_fixed_artists,
    )


@app.cell(hide_code=True)
def _(
    mo,
    rolling_7_added_artists,
    rolling_7_material_date_changes,
    rolling_7_missing_fixed_artists,
):
    mo.vstack(
        [
            mo.md("### New events found by rolling 7 days"),
            rolling_7_added_artists,
            mo.md("### Shared artists whose first breakout moves by more than 30 days"),
            rolling_7_material_date_changes,
            mo.md("### Monday-week events not found by rolling 7 days"),
            rolling_7_missing_fixed_artists,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    mo,
    rolling_7_alignment_support,
    rolling_7_timing_stability,
    window_method_summary,
):
    _rolling_row = window_method_summary.filter(
        window_method_summary["method"] == "Rolling 7 days"
    ).row(0, named=True)
    _support = rolling_7_alignment_support.row(1, named=True)["events"]
    _stable = rolling_7_timing_stability.row(0, named=True)["artists"]

    mo.md(f"""
    The fixed week boundary is too influential to ignore. Rolling 7 days keeps the **same 7-day scale**, but removes the arbitrary weekday boundary.

    It finds **{_rolling_row['breakout_events']} events**. **{_support}** of them are also found under at least one fixed weekday alignment, so the larger event set is mostly exposing boundary-sensitive cases rather than creating completely new behaviour.

    For artists found by both Monday weeks and rolling 7 days, **{_stable}** keep essentially the same timing (within 7 days). A smaller group moves much earlier and needs direct review.

    The 5-day version changes the event set again without solving a separate product problem. **Rolling 7 days is therefore the stronger working event window.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Should an important period need more than one active day?

    Rolling windows remove the arbitrary weekday boundary, but they can still let a single intense listening day make a period important.

    We compare the same rolling 7-day rule with **1, 2, or 3 active listening days required in each important period**. The 2-day version is the main candidate; 3 days is only a stricter sensitivity check.
    """)
    return


@app.cell
def _(
    build_rolling_events,
    daily_artist_listening,
    daily_user_listening,
    user_limits,
):
    _, _, rolling_7_2day_warmup_established, rolling_7_2day_breakout_events = build_rolling_events(
        daily_user_listening,
        daily_artist_listening,
        user_limits,
        period_days=7,
        min_active_days=2,
    )

    _, _, rolling_7_3day_warmup_established, rolling_7_3day_breakout_events = build_rolling_events(
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
                "base_events_kept": _shared,
                "base_events_kept_pct": round(
                    _shared / len(_base_keys) * 100,
                    1,
                ),
                "removed_vs_base": len(_base_keys - _keys),
                "new_vs_base": len(_keys - _base_keys),
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

    active_day_rule_summary
    return (active_day_rule_summary,)


@app.cell
def _(
    pl,
    rolling_7_2day_breakout_events,
    rolling_7_breakout_events,
):
    rolling_7_removed_by_two_day_rule = (
        rolling_7_breakout_events
        .join(
            rolling_7_2day_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "artist")
    )

    rolling_7_added_by_two_day_rule = (
        rolling_7_2day_breakout_events
        .join(
            rolling_7_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "artist")
    )

    _base_dates = rolling_7_breakout_events.select(
        "user_id",
        "artist_id",
        pl.col("canonical_name").alias("artist"),
        pl.col("breakout_period_start").alias("base_start"),
    )

    rolling_7_two_day_date_changes = (
        rolling_7_2day_breakout_events
        .select(
            "user_id",
            "artist_id",
            pl.col("breakout_period_start").alias("two_day_start"),
        )
        .join(
            _base_dates,
            on=["user_id", "artist_id"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("two_day_start") - pl.col("base_start")
            )
            .dt.total_days()
            .alias("date_change_days")
        )
        .filter(pl.col("date_change_days") != 0)
        .select(
            "artist",
            "base_start",
            "two_day_start",
            "date_change_days",
        )
        .sort("date_change_days", descending=True)
    )

    return (
        rolling_7_added_by_two_day_rule,
        rolling_7_removed_by_two_day_rule,
        rolling_7_two_day_date_changes,
    )


@app.cell(hide_code=True)
def _(
    mo,
    rolling_7_added_by_two_day_rule,
    rolling_7_removed_by_two_day_rule,
    rolling_7_two_day_date_changes,
):
    mo.vstack(
        [
            mo.md("### 1 active day → 2 active days"),
            mo.md("#### Breakouts removed"),
            rolling_7_removed_by_two_day_rule,
            mo.md("#### Breakouts added"),
            rolling_7_added_by_two_day_rule,
            mo.md("#### Shared artists whose breakout date changes"),
            rolling_7_two_day_date_changes,
        ]
    )
    return


@app.cell
def _(
    pl,
    rolling_7_2day_breakout_events,
    rolling_7_3day_breakout_events,
):
    rolling_7_removed_by_three_day_rule = (
        rolling_7_2day_breakout_events
        .join(
            rolling_7_3day_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "artist")
    )

    rolling_7_added_by_three_day_rule = (
        rolling_7_3day_breakout_events
        .join(
            rolling_7_2day_breakout_events.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .select(
            pl.col("canonical_name").alias("artist"),
            "breakout_period_start",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "artist")
    )

    _two_day_dates = rolling_7_2day_breakout_events.select(
        "user_id",
        "artist_id",
        pl.col("canonical_name").alias("artist"),
        pl.col("breakout_period_start").alias("two_day_start"),
    )

    rolling_7_three_day_date_changes = (
        rolling_7_3day_breakout_events
        .select(
            "user_id",
            "artist_id",
            pl.col("breakout_period_start").alias("three_day_start"),
        )
        .join(
            _two_day_dates,
            on=["user_id", "artist_id"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("three_day_start") - pl.col("two_day_start")
            )
            .dt.total_days()
            .alias("date_change_days")
        )
        .filter(pl.col("date_change_days") != 0)
        .select(
            "artist",
            "two_day_start",
            "three_day_start",
            "date_change_days",
        )
        .sort("date_change_days", descending=True)
    )

    return (
        rolling_7_added_by_three_day_rule,
        rolling_7_removed_by_three_day_rule,
        rolling_7_three_day_date_changes,
    )


@app.cell(hide_code=True)
def _(
    mo,
    rolling_7_added_by_three_day_rule,
    rolling_7_removed_by_three_day_rule,
    rolling_7_three_day_date_changes,
):
    mo.vstack(
        [
            mo.md("### 2 active days → 3 active days"),
            mo.md("#### Breakouts removed"),
            rolling_7_removed_by_three_day_rule,
            mo.md("#### Breakouts added"),
            rolling_7_added_by_three_day_rule,
            mo.md("#### Shared artists whose breakout date changes"),
            rolling_7_three_day_date_changes,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **2+ active-day rule** is the working choice. It removes periods driven by a single listening day without requiring the stronger spread of the 3-day version.

    From here on, every result uses **rolling 7-day periods with at least 2 active listening days in each important period**.
    """)
    return


@app.cell
def _(
    rolling_7_2day_breakout_events,
):
    working_breakout_events = rolling_7_2day_breakout_events
    return (working_breakout_events,)


@app.cell
def _(
    pl,
    rolling_7_2day_warmup_established,
    working_breakout_events,
):
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

    working_breakout_summary
    return (working_breakout_summary,)


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

    working_breakouts_by_year
    return (working_breakouts_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final breakout list

    This is the working event set used for the daily-target analysis below.
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

    working_breakout_artist_list
    return (working_breakout_artist_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Daily prediction target

    The daily target is settled as **confirmed breakout**:

    > Will this artist become a confirmed first breakout within the next `H` days?

    We keep the first important period separately so we can later tell whether Sonora predicted the breakout before it started forming or only detected it while it was already developing.

    For this comparison only, an artist can be scored from the day after their first listen and then stays scoreable. The real inactivity / re-entry rule comes next.
    """)
    return


@app.cell
def _(
    first_artist_listens,
    pl,
    working_breakout_events,
):
    breakout_timing = (
        working_breakout_events
        .join(
            first_artist_listens,
            on=["user_id", "artist_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("first_listen_date") + pl.duration(days=1)
            ).alias("first_score_date")
        )
        .with_columns(
            (
                pl.col("first_score_date")
                < pl.col("breakout_period_start")
            ).alias("scoreable_before_first_period"),
            (
                pl.col("first_listen_date")
                .is_between(
                    pl.col("breakout_period_start"),
                    pl.col("first_important_period_end"),
                )
            ).alias("first_listen_inside_first_period"),
            (
                pl.col("confirmation_date")
                - pl.col("first_score_date")
            )
            .dt.total_days()
            .alias("days_from_first_score_to_confirmation"),
        )
    )

    _total = breakout_timing.height
    _before = breakout_timing.filter(
        pl.col("scoreable_before_first_period")
    ).height
    _inside = breakout_timing.filter(
        pl.col("first_listen_inside_first_period")
    ).height

    breakout_scoreability_summary = pl.DataFrame(
        {
            "measure": [
                "Breakout events",
                "Could be scored before first important period started",
                "Could not be scored before first important period started",
                "First-ever listen happened inside first important period",
            ],
            "events": [
                _total,
                _before,
                _total - _before,
                _inside,
            ],
        }
    ).with_columns(
        (
            pl.col("events") / _total * 100
        ).round(1).alias("event_pct")
    )

    breakout_scoreability_summary
    return breakout_scoreability_summary, breakout_timing


@app.cell
def _(breakout_timing, pl):
    first_listen_position_in_period = (
        breakout_timing
        .filter(pl.col("first_listen_inside_first_period"))
        .with_columns(
            (
                pl.col("first_listen_date")
                - pl.col("breakout_period_start")
            )
            .dt.total_days()
            .alias("day_in_first_period")
        )
        .group_by("day_in_first_period")
        .agg(pl.len().alias("events"))
        .with_columns(
            (pl.col("day_in_first_period") + 1).alias("period_day")
        )
        .select("period_day", "events")
        .sort("period_day")
    )

    first_listen_position_in_period
    return (first_listen_position_in_period,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some breakouts still cannot be predicted before their first important period because the artist has not been heard yet. Under the confirmed-breakout target, those artists can enter the watchlist as soon as they actually appear and still be predicted before confirmation.
    """)
    return


@app.cell
def _(breakout_timing, pl):
    _summary_rows = []
    _timing_rows = []

    for _horizon in [14, 30, 60]:
        _scored = (
            breakout_timing
            .with_columns(
                (
                    pl.col("confirmation_date")
                    - pl.duration(days=_horizon)
                ).alias("_window_start")
            )
            .with_columns(
                pl.max_horizontal(
                    "first_score_date",
                    "_window_start",
                ).alias("_positive_start")
            )
            .with_columns(
                (
                    pl.col("confirmation_date")
                    - pl.col("_positive_start")
                )
                .dt.total_days()
                .clip(lower_bound=0)
                .alias("_positive_days"),
                pl.when(
                    pl.col("_positive_start")
                    < pl.col("breakout_period_start")
                )
                .then(
                    (
                        pl.col("breakout_period_start")
                        - pl.col("_positive_start")
                    ).dt.total_days()
                )
                .otherwise(0)
                .alias("_before_first_period_days"),
                (
                    pl.min_horizontal(
                        "confirmation_date",
                        (
                            pl.col("first_important_period_end")
                            + pl.duration(days=1)
                        ),
                    )
                    - pl.max_horizontal(
                        "_positive_start",
                        "breakout_period_start",
                    )
                )
                .dt.total_days()
                .clip(lower_bound=0)
                .alias("_during_first_period_days"),
                (
                    pl.col("confirmation_date")
                    - pl.max_horizontal(
                        "_positive_start",
                        (
                            pl.col("first_important_period_end")
                            + pl.duration(days=1)
                        ),
                    )
                )
                .dt.total_days()
                .clip(lower_bound=0)
                .alias("_after_first_period_days"),
            )
        )

        _positive = _scored.filter(pl.col("_positive_days") > 0)
        _positive_days = int(_positive["_positive_days"].sum())
        _pre_days = int(_positive["_before_first_period_days"].sum())
        _events_before = _positive.filter(
            pl.col("_before_first_period_days") > 0
        ).height

        _summary_rows.append(
            {
                "horizon_days": _horizon,
                "events_with_positive_window": _positive.height,
                "event_coverage_pct": round(
                    _positive.height / breakout_timing.height * 100,
                    1,
                ),
                "events_with_pre_breakout_opportunity": _events_before,
                "pre_breakout_event_pct": round(
                    _events_before / breakout_timing.height * 100,
                    1,
                ),
                "positive_daily_rows": _positive_days,
                "positive_rows_before_first_period_pct": round(
                    _pre_days / _positive_days * 100,
                    1,
                ) if _positive_days else 0.0,
            }
        )

        _timing_rows.extend(
            [
                {
                    "horizon_days": _horizon,
                    "positive_timing": "Before first important period",
                    "positive_daily_rows": int(
                        _positive["_before_first_period_days"].sum()
                    ),
                },
                {
                    "horizon_days": _horizon,
                    "positive_timing": "During first important period",
                    "positive_daily_rows": int(
                        _positive["_during_first_period_days"].sum()
                    ),
                },
                {
                    "horizon_days": _horizon,
                    "positive_timing": "After first period, before confirmation",
                    "positive_daily_rows": int(
                        _positive["_after_first_period_days"].sum()
                    ),
                },
            ]
        )

    confirmation_horizon_summary = pl.DataFrame(_summary_rows)
    confirmation_positive_timing = (
        pl.DataFrame(_timing_rows)
        .with_columns(
            (
                pl.col("positive_daily_rows")
                / pl.col("positive_daily_rows").sum().over("horizon_days")
                * 100
            )
            .round(1)
            .alias("positive_row_pct")
        )
    )

    confirmation_horizon_summary
    return confirmation_horizon_summary, confirmation_positive_timing


@app.cell
def _(confirmation_positive_timing):
    confirmation_positive_timing
    return


@app.cell
def _(breakout_timing, pl):
    _late_entry = breakout_timing.filter(
        ~pl.col("scoreable_before_first_period")
    )

    late_entry_summary = pl.DataFrame(
        {
            "measure": [
                "Events not scoreable before first important period",
                "First score before second important period starts",
                "Minimum days from first score to confirmation",
                "Median days from first score to confirmation",
                "Maximum days from first score to confirmation",
            ],
            "value": [
                _late_entry.height,
                _late_entry.filter(
                    pl.col("first_score_date")
                    < pl.col("second_important_period_start")
                ).height,
                int(_late_entry["days_from_first_score_to_confirmation"].min()),
                int(_late_entry["days_from_first_score_to_confirmation"].median()),
                int(_late_entry["days_from_first_score_to_confirmation"].max()),
            ],
        }
    )

    late_entry_summary
    return (late_entry_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **14-day horizon** is mainly useful once a breakout is already developing. The **30- and 60-day horizons** create more opportunity to rank an artist before their first important period.

    We should not choose between 30 and 60 days yet. The real inactivity / re-entry rule determines which of those theoretical positive days would actually exist in the watchlist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Working decisions

    - Use **rolling 7-day periods** rather than fixed calendar weeks.
    - An important period must contain listening on **at least 2 different days**.
    - A breakout needs **two important, non-overlapping 7-day periods within the same short 21-day span**.
    - Use **confirmed breakout** as the daily prediction target.
    - Keep the first important period for lead-time evaluation.
    - Do not choose 30 vs 60 days until daily artist eligibility is applied.

    The next step is the inactivity / re-entry rule for the daily watchlist.
    """)
    return


if __name__ == "__main__":
    app.run()

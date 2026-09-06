"""First confirmed breakouts from completed rolling listening periods."""

import datetime as dt
from collections import deque

import polars as pl

from sonora.breakout.history import available_user_bounds, validate_user_bounds


def detect_first_breakouts(
    daily_user,
    daily_artist,
    user_bounds,
    period_days=7,
    min_active_days=2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return first events and artists already established during initial history.

    Bounds are per user. Only completed periods contribute; importance uses
    earlier period shares, rounded to two decimals, over the prior 730 days.
    """
    validate_user_bounds(user_bounds, period_days=period_days)
    if period_days < 1 or not 1 <= min_active_days <= period_days:
        raise ValueError("Invalid period length or active-day requirement")
    user_bounds = available_user_bounds(user_bounds)
    _expanded_user = pl.concat(
        [
            daily_user.select(
                "user_id",
                (pl.col("date") + pl.duration(days=_offset)).alias("window_end"),
                "user_scrobbles",
            )
            for _offset in range(period_days)
        ]
    )

    _rolling_user = (
        _expanded_user.group_by("user_id", "window_end")
        .agg(pl.col("user_scrobbles").sum())
        .join(
            user_bounds.select("user_id", "first_date", "last_complete_date"),
            on="user_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("window_end") - pl.duration(days=period_days - 1)).alias(
                "window_start"
            )
        )
        .filter(
            (pl.col("window_start") >= pl.col("first_date"))
            & (pl.col("window_end") <= pl.col("last_complete_date"))
        )
        .drop("first_date", "last_complete_date")
    )

    _expanded_artist = pl.concat(
        [
            daily_artist.select(
                "user_id",
                "artist_id",
                "date",
                (pl.col("date") + pl.duration(days=_offset)).alias("window_end"),
                "artist_scrobbles",
            )
            for _offset in range(period_days)
        ]
    )

    _windows = (
        _expanded_artist.group_by("user_id", "artist_id", "window_end")
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
            (pl.col("window_end") - pl.duration(days=period_days - 1)).alias(
                "window_start"
            ),
            (pl.col("artist_scrobbles") / pl.col("user_scrobbles") * 100)
            .round(2)
            .alias("artist_share_pct"),
        )
        .join(
            user_bounds.select("user_id", "first_date", "warmup_end"),
            on="user_id",
            how="left",
            validate="m:1",
        )
    )

    _cutoff_rows = []
    for _user_windows in _windows.partition_by("user_id"):
        _user_id = _user_windows["user_id"][0]
        _warmup_end = _user_windows["warmup_end"][0]
        _candidate_min_end = _warmup_end + dt.timedelta(days=period_days - 1)

        _by_date = (
            _user_windows.select(
                "window_end",
                (pl.col("artist_share_pct") * 100)
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
                _active_dates.append((_row["window_end"], _counts, _row_count))
                _add_index += 1

            _lower_date = _target_date - dt.timedelta(days=730)
            while _active_dates and _active_dates[0][0] < _lower_date:
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

    _importance = _windows.join(
        pl.DataFrame(
            _cutoff_rows,
            schema={
                "user_id": pl.String,
                "window_end": pl.Date,
                "top_10_cutoff": pl.Float64,
            },
        ),
        on=["user_id", "window_end"],
        how="inner",
        validate="m:1",
    ).with_columns(
        (pl.col("artist_share_pct") >= pl.col("top_10_cutoff")).alias("is_important")
    )

    _important = _importance.filter(
        pl.col("is_important") & (pl.col("active_days") >= min_active_days)
    )

    _future = pl.concat(
        [
            _important.select(
                "user_id",
                "artist_id",
                (pl.col("window_end") - pl.duration(days=_days_ahead)).alias(
                    "window_end"
                ),
                pl.col("window_end").alias("_second_window_end"),
            )
            for _days_ahead in range(period_days, 2 * period_days + 1)
        ]
    )

    _candidates = (
        _important.join(
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
        _windows.filter(pl.col("window_start") < pl.col("warmup_end"))
        .group_by("user_id")
        .agg(pl.col("artist_share_pct").quantile(0.90).alias("_warmup_cutoff"))
    )

    _warmup_scored = _windows.join(
        _warmup_cutoffs,
        on="user_id",
        how="left",
        validate="m:1",
    ).with_columns(
        (pl.col("artist_share_pct") >= pl.col("_warmup_cutoff")).alias(
            "_warmup_important"
        )
    )

    _warmup_important = _warmup_scored.filter(
        pl.col("_warmup_important") & (pl.col("active_days") >= min_active_days)
    )

    _warmup_anchors = _warmup_important.filter(
        pl.col("window_start") < pl.col("warmup_end")
    )

    _warmup_future = pl.concat(
        [
            _warmup_important.select(
                "user_id",
                "artist_id",
                (pl.col("window_end") - pl.duration(days=_days_ahead)).alias(
                    "window_end"
                ),
                pl.col("window_end").alias("_second_window_end"),
            )
            for _days_ahead in range(period_days, 2 * period_days + 1)
        ]
    )

    _warmup_established = (
        _warmup_anchors.join(
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
            pl.col("window_start").alias("established_period_start"),
        )
    )

    _events = _candidates.join(
        _warmup_established.select("user_id", "artist_id"),
        on=["user_id", "artist_id"],
        how="anti",
    ).with_columns(
        pl.col("window_start").alias("breakout_period_start"),
        pl.col("window_end").alias("first_important_period_end"),
        (pl.col("_second_window_end") - pl.duration(days=period_days - 1)).alias(
            "second_important_period_start"
        ),
        (pl.col("_second_window_end") + pl.duration(days=1)).alias("confirmation_date"),
        pl.col("artist_share_pct").alias("first_share_pct"),
        pl.col("active_days").alias("first_active_days"),
    )

    _second_stats = _windows.select(
        "user_id",
        "artist_id",
        pl.col("window_end").alias("_second_window_end"),
        pl.col("active_days").alias("second_active_days"),
    )

    _events = (
        _events.join(
            _second_stats,
            on=["user_id", "artist_id", "_second_window_end"],
            how="left",
            validate="1:1",
        )
        .select(
            "user_id",
            "artist_id",
            "breakout_period_start",
            "first_important_period_end",
            "second_important_period_start",
            "confirmation_date",
            "first_share_pct",
            "first_active_days",
            "second_active_days",
        )
        .sort("breakout_period_start", "user_id", "artist_id")
    )

    return _events, _warmup_established

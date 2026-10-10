import datetime as dt

import polars as pl
import pytest

from sonora.breakout import (
    score_listening_share_momentum,
    score_recent_active_days,
    score_recent_listening_share,
)

BASE = dt.date(2024, 1, 10)


def listened(day_offset, hour=12):
    return dt.datetime.combine(
        BASE + dt.timedelta(days=day_offset),
        dt.time(hour),
        tzinfo=dt.UTC,
    )


def events(rows):
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "listened_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def candidates(rows):
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
        },
        orient="row",
    )


def test_recent_share_uses_only_the_previous_seven_days():
    history = events(
        [
            ("u", "a", listened(-8)),
            ("u", "a", listened(-7)),
            ("u", "a", listened(-1)),
            ("u", "a", listened(0)),
            ("u", "b", listened(-1, 13)),
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_recent_listening_share(history, scoring)

    shares = dict(result.select("artist_id", "listening_share_7d").iter_rows())
    assert shares == {"a": 2 / 3, "b": 1 / 3}


def test_equal_share_is_broken_by_the_most_recent_listen():
    history = events(
        [
            ("u", "a", listened(-2, 10)),
            ("u", "b", listened(-1, 10)),
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_recent_listening_share(history, scoring)

    assert result["artist_id"].to_list() == ["b", "a"]
    assert result["score"].to_list() == [-1.0, -2.0]


def test_equal_share_and_recency_use_prior_scrobbles():
    history = events(
        [
            ("u", "a", listened(-8, 9)),
            ("u", "a", listened(-1, 10)),
            ("u", "b", listened(-1, 10)),
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_recent_listening_share(history, scoring)

    assert result["artist_id"].to_list() == ["a", "b"]
    assert result["prior_scrobbles"].to_list() == [2, 1]


def test_listening_share_is_isolated_per_user():
    history = events(
        [
            ("u1", "a", listened(-1)),
            ("u1", "b", listened(-1, 13)),
            ("u2", "a", listened(-1)),
            ("u2", "a", listened(-2)),
            ("u2", "b", listened(-1, 13)),
            ("u2", "b", listened(-2, 13)),
            ("u2", "b", listened(-3, 13)),
            ("u2", "b", listened(-4, 13)),
        ]
    )
    scoring = candidates(
        [
            ("u1", "a", BASE),
            ("u1", "b", BASE),
            ("u2", "a", BASE),
            ("u2", "b", BASE),
        ]
    )

    result = score_recent_listening_share(history, scoring)
    shares = {
        (user, artist): share
        for user, artist, share in result.select(
            "user_id", "artist_id", "listening_share_7d"
        ).iter_rows()
    }

    assert shares[("u1", "a")] == 0.5
    assert shares[("u1", "b")] == 0.5
    assert shares[("u2", "a")] == 2 / 6
    assert shares[("u2", "b")] == 4 / 6


def test_momentum_prefers_rising_share_over_higher_current_share():
    history = events(
        [
            *[("u", "a", listened(day, 9)) for day in [-7, -6, -5]],
            *[("u", "b", listened(day, 10)) for day in [-14, -13, -12, -11]],
            *[("u", "b", listened(day, 11)) for day in [-7, -6, -5, -4, -3]],
            *[
                ("u", "filler", listened(day, 12))
                for day in [-14, -13, -12, -11, -10, -9]
            ],
            *[("u", "filler", listened(day, 13)) for day in [-7, -6]],
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_listening_share_momentum(history, scoring)

    assert result["artist_id"].to_list() == ["a", "b"]
    values = {
        artist: (current, previous, momentum)
        for artist, current, previous, momentum in result.select(
            "artist_id",
            "listening_share_7d",
            "listening_share_previous_7d",
            "listening_share_momentum",
        ).iter_rows()
    }
    assert values["a"] == pytest.approx((3 / 10, 0.0, 3 / 10))
    assert values["b"] == pytest.approx((5 / 10, 4 / 10, 1 / 10))


def test_equal_momentum_uses_current_share_as_first_tie_break():
    history = events(
        [
            *[("u", "a", listened(day, 9)) for day in [-14, -13]],
            *[
                ("u", "filler", listened(day, 11))
                for day in [-12, -11, -10, -9, -8, -8]
            ],
            *[("u", "a", listened(day, 9)) for day in [-7, -6, -5, -4]],
            *[("u", "b", listened(day, 10)) for day in [-7, -6]],
            *[("u", "filler", listened(day, 11)) for day in [-5, -4]],
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_listening_share_momentum(history, scoring)

    assert result["artist_id"].to_list() == ["a", "b"]
    momentum = dict(result.select("artist_id", "listening_share_momentum").iter_rows())
    assert momentum["a"] == momentum["b"]


def test_active_days_prefers_repeated_listening_over_concentrated_volume():
    history = events(
        [
            ("u", "a", listened(-8, 8)),
            ("u", "a", listened(-1, 9)),
            ("u", "a", listened(-1, 10)),
            ("u", "a", listened(-1, 11)),
            ("u", "b", listened(-2, 9)),
            ("u", "b", listened(-1, 12)),
            ("u", "a", listened(0, 8)),
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_recent_active_days(history, scoring)

    assert result["artist_id"].to_list() == ["b", "a"]
    values = {
        artist: active_days
        for artist, active_days in result.select(
            "artist_id", "artist_active_days_7d"
        ).iter_rows()
    }
    assert values == {"b": 2, "a": 1}


def test_active_days_uses_current_share_as_first_tie_break():
    history = events(
        [
            ("u", "a", listened(-2, 9)),
            ("u", "a", listened(-1, 9)),
            ("u", "a", listened(-1, 10)),
            ("u", "b", listened(-2, 10)),
            ("u", "b", listened(-1, 11)),
        ]
    )
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])

    result = score_recent_active_days(history, scoring)

    assert result["artist_id"].to_list() == ["a", "b"]
    assert result["artist_active_days_7d"].to_list() == [2, 2]

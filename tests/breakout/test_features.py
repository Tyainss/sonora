import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sonora.breakout import build_breakout_features

BASE = dt.date(2024, 1, 10)
KEYS = ["user_id", "artist_id", "scoring_date"]


def listened(offset, hour=12):
    return dt.datetime.combine(
        BASE + dt.timedelta(days=offset), dt.time(hour), tzinfo=dt.UTC
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


def test_features_use_exact_windows_and_ignore_current_and_future_listening():
    history = events(
        [
            ("u", "a", listened(-8, 23)),
            ("u", "a", listened(-7, 0)),
            ("u", "a", listened(-1, 23)),
            ("u", "other", listened(-1)),
        ]
    )
    scoring = candidates([("u", "a", BASE)])
    result = build_breakout_features(history, scoring)
    assert result.columns == [*KEYS, "listening_share_7d", "previous_listening_count"]
    assert result.select("listening_share_7d", "previous_listening_count").row(0) == (
        2 / 3,
        1,
    )

    extended = pl.concat(
        [history, events([("u", "a", listened(0, 0)), ("u", "other", listened(1))])]
    )
    assert_frame_equal(build_breakout_features(extended.lazy(), scoring), result)


def test_every_candidate_day_gets_features_as_history_moves_between_windows():
    history = events([("u", "a", listened(-8)), ("u", "a", listened(-7))])
    scoring = candidates(
        [("u", "a", BASE + dt.timedelta(days=1)), ("u", "a", BASE)]
    ).with_columns(pl.lit(1).alias("target"))
    result = build_breakout_features(history, scoring).sort(KEYS)

    assert_frame_equal(result.select(KEYS), scoring.select(KEYS).sort(KEYS))
    assert result["listening_share_7d"].to_list() == [1.0, 0.0]
    assert result["previous_listening_count"].to_list() == [1, 2]
    assert "target" not in result.columns


def test_history_is_isolated_per_user_and_denominator_includes_other_artists():
    history = events(
        [
            ("u1", "a", listened(-8)),
            ("u1", "a", listened(-1)),
            ("u1", "other", listened(-1)),
            ("u2", "a", listened(-9)),
            ("u2", "a", listened(-8)),
            ("u2", "a", listened(-1)),
        ]
    )
    scoring = candidates([("u1", "a", BASE), ("u2", "a", BASE)])
    result = build_breakout_features(history, scoring).sort(KEYS)

    assert result["listening_share_7d"].to_list() == [0.5, 1.0]
    assert result["previous_listening_count"].to_list() == [1, 2]


def test_recent_only_artist_has_zero_previous_count():
    result = build_breakout_features(
        events([("u", "a", listened(-7, 0))]), candidates([("u", "a", BASE)])
    )
    assert result["previous_listening_count"].to_list() == [0]


def test_artists_with_identical_history_still_receive_separate_feature_rows():
    scoring = candidates([("u", "a", BASE), ("u", "b", BASE)])
    result = build_breakout_features(
        events([("u", "a", listened(-1)), ("u", "b", listened(-1))]), scoring
    )
    assert_frame_equal(result.select(KEYS).sort(KEYS), scoring.sort(KEYS))
    assert result["listening_share_7d"].to_list() == [0.5, 0.5]
    assert result["previous_listening_count"].to_list() == [0, 0]


def test_empty_candidates_return_typed_empty_features():
    result = build_breakout_features(events([]), candidates([]))
    assert result.is_empty()
    assert result.schema == {
        "user_id": pl.String,
        "artist_id": pl.String,
        "scoring_date": pl.Date,
        "listening_share_7d": pl.Float64,
        "previous_listening_count": pl.UInt32,
    }


def test_duplicate_candidate_rows_are_rejected():
    with pytest.raises(ValueError, match="duplicate rows"):
        build_breakout_features(
            events([("u", "a", listened(-1))]),
            candidates([("u", "a", BASE), ("u", "a", BASE)]),
        )


def test_candidate_without_prior_history_is_rejected():
    with pytest.raises(ValueError, match="prior listening history"):
        build_breakout_features(
            events([("u", "a", listened(0))]), candidates([("u", "a", BASE)])
        )


def test_non_utc_events_are_rejected():
    history = events([("u", "a", listened(-1))]).with_columns(
        pl.col("listened_at").dt.convert_time_zone("Europe/Lisbon")
    )
    with pytest.raises(ValueError, match="UTC timestamps"):
        build_breakout_features(history, candidates([("u", "a", BASE)]))
